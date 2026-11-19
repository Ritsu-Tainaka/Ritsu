"""Gemini REST istemcisi: yerel OCR ile okunan metinleri toplu olarak Türkçeye çevirir (yalnızca metin gönderilir)."""
import json, random, re, threading, time
import requests

API = "https://generativelanguage.googleapis.com/v1beta"


class GeminiError(Exception):
    def __init__(self, msg, permanent=False):
        super().__init__(msg)
        self.permanent = permanent


class Stopped(Exception):
    pass


class GeminiTimeout(GeminiError):
    """Yanıt süre sınırında gelmedi (genelde paket çok büyük ya da sunucu yoğun). Paket bölünerek yeniden denenir."""


STYLE_GUIDES = {
    "natural": (
        "Translation style: Natural & idiomatic Turkish. Adapt slang, idioms, jokes, and emotional nuances "
        "into fluent everyday Turkish as people actually speak, matching each character's personality and social distance (sen/siz)."
    ),
    "detailed": (
        "Translation style: Detailed & faithful Turkish. Preserve every nuance, subtext, cultural reference, "
        "honorific/form of address, and descriptive detail from the original English without oversimplifying or omitting meaning."
    ),
    "dramatic": (
        "Translation style: Expressive & dramatic Turkish. Use impactful, vivid, and atmospheric Turkish suited for "
        "action, fantasy, historical, or intense emotional scenes while keeping dialogue punchy."
    ),
}


TEXT_PROMPT = """You are a professional manga localizer (English -> Turkish).
Below is JSON: a list of speech bubbles / captions extracted by OCR from consecutive manga pages, in reading order.
Return ONLY JSON: {{"summary": "<one short English sentence on the story so far>",
 "new_terms": [{{"en": "..", "tr": ".."}}],
 "items": [{{"id": "<same id>", "tr": "<Turkish translation>"}}]}}
Rules:
- Keep every id. The OCR text may contain mistakes (wrong letters, split/broken words, punctuation glitches): reconstruct the intended English text from context before translating.
- Translate into natural, fluent Turkish matching each character's personality, register (sen/siz), and emotional tone across the scene. Keep it concise so it fits in a bubble.
- If an item is not real dialogue (noise, watermark, page number, scanlator credit), return "" for its tr.
- Items with "free": true are words drawn directly on the artwork (outside speech bubbles). Translate them if they are dialogue, thoughts, narration or a short exclamation a character says (e.g. "HE'S", "Whee!", "I remember him..."); a sentence may continue across neighbouring items, keep each part in its own item. If such an item is only a pure sound effect / onomatopoeia (e.g. BAM, THUD, SQUIRM, FWOOSH) return "" for it.
- Use correct Turkish characters (ç ğ ı İ ö ş ü). No explanations.{style}{glossary}{context}
INPUT:
{payload}"""


def _parse_json(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


def _retry_wait(r, attempt):
    """Bekleme: Retry-After (saniye) > yanıt gövdesindeki retryDelay ("37s") > üstel geri çekilme."""
    h = (r.headers.get("Retry-After") or "").strip()
    if h:
        try:
            return min(300.0, max(1.0, float(h)))
        except ValueError:
            pass
    try:
        for d in r.json().get("error", {}).get("details", []) or []:
            m = re.match(r"^\s*(\d+(?:\.\d+)?)s\s*$", str(d.get("retryDelay", "")))
            if m:
                return min(300.0, max(1.0, float(m.group(1)) + 1))
    except Exception:
        pass
    return min(90, 4 * 2 ** attempt + random.random() * 3)


def _bool(v, default=False):
    """Model bazen "false" gibi metin döndürür; bool("false") True olmasın."""
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "evet"):
            return True
        if s in ("false", "0", "no", "hayır", "hayir", ""):
            return False
        return default
    return default if v is None else bool(v)


class Gemini:
    def __init__(self, api_key, model, rpm=10, stop=None, log=None, img_side=2000,
                 thinking_level="medium", translation_style="natural", custom_prompt=""):
        self.key, self.model, self.img_side = api_key, model, int(img_side)
        self.thinking_level = str(thinking_level or "medium").strip().lower()
        self.translation_style = str(translation_style or "natural").strip().lower()
        self.custom_prompt = str(custom_prompt or "").strip()
        self.min_gap = 60.0 / max(1, rpm)
        self.stop = stop or threading.Event()
        self.log = log or (lambda m: None)
        self._last = 0.0
        self.read_timeout = 240

    def _thinking_config(self):
        lvl = self.thinking_level
        if lvl in ("", "auto", "default"):
            return None
        m = (self.model or "").lower()
        # "gemini-flash-lite-latest" gibi takma adlar da Gemini 3+ modellerine gider (ör. gemini-3.5-flash-lite);
        # ayar gönderilmezse model kendi varsayılanıyla çalışır ve yoğun saatlerde 503 alır. Desteklenmezse
        # _post() 400 alınca ayarı çıkarıp yeniden dener.
        if not (re.search(r"gemini-(?:3|4)", m) or m.endswith("-latest")):
            return None
        if lvl == "minimal" and "gemini-4" in m:
            lvl = "low"
        if lvl == "xhigh" and "gemini-4-hydrogen" not in m:
            lvl = "high"
        if lvl in ("minimal", "low", "medium", "high", "xhigh"):
            return {"thinkingLevel": lvl}
        return None

    def _gen_config(self):
        gc = {"responseMimeType": "application/json"}
        tc = self._thinking_config()
        if tc:
            gc["thinkingConfig"] = tc
        return gc

    def _extra_instructions(self):
        parts = []
        guide = STYLE_GUIDES.get(self.translation_style, STYLE_GUIDES["natural"])
        if guide:
            parts.append(f"- {guide}")
        if self.custom_prompt:
            parts.append(f"- Additional user translation instructions (follow strictly):\n  {self.custom_prompt}")
        return ("\n" + "\n".join(parts)) if parts else ""

    def _sleep(self, secs):
        end = time.time() + secs
        while time.time() < end:
            if self.stop.is_set():
                raise Stopped()
            time.sleep(min(0.25, max(0, end - time.time())))

    def _request(self, url, payload):
        """İsteği ayrı iş parçacığında gönderir; Durdur'a basılınca yanıtı beklemeden hemen çıkar
        (yarıda kalan istek arka planda kendi kendine biter, sonucu kullanılmaz)."""
        box = {}

        def go():
            try:
                box["r"] = requests.post(url, json=payload, timeout=(20, self.read_timeout),
                                         headers={"x-goog-api-key": self.key, "Content-Type": "application/json"})
            except Exception as e:
                box["e"] = e
        t = threading.Thread(target=go, daemon=True)
        t.start()
        t0 = time.time()
        next_note = 30
        while t.is_alive():
            if self.stop.is_set():
                raise Stopped()
            t.join(0.2)
            waited = time.time() - t0
            if waited >= next_note:
                self.log(f"    Gemini yanıtı bekleniyor... ({int(waited)} sn; büyük paketlerde 1-3 dk sürebilir)")
                next_note += 30
        if "e" in box:
            raise box["e"]
        return box["r"]

    def _post(self, payload):
        if not self.key:
            raise GeminiError("API anahtarı girilmemiş (Ayarlar).", permanent=True)
        url = f"{API}/models/{self.model}:generateContent"
        timeouts = 0
        for attempt in range(7):
            self._sleep(max(0.0, self._last + self.min_gap - time.time()))
            self._last = time.time()
            try:
                r = self._request(url, payload)
            except requests.Timeout:
                timeouts += 1
                if timeouts >= 2:
                    raise GeminiTimeout(f"Gemini {self.read_timeout} sn içinde yanıt vermedi.")
                self.log(f"Gemini {self.read_timeout} sn içinde yanıt vermedi, bir kez daha deneniyor...")
                continue
            except requests.RequestException as e:
                wait = min(60, 2 ** attempt + random.random())
                self.log(f"Ağ hatası ({e.__class__.__name__}), {wait:.0f} sn sonra tekrar...")
                self._sleep(wait)
                continue
            if r.status_code == 200:
                return r.json()
            msg = ""
            try:
                msg = r.json().get("error", {}).get("message", "")
            except Exception:
                msg = r.text[:200]
            if r.status_code == 400 and "thinkingConfig" in payload.get("generationConfig", {}):
                payload["generationConfig"].pop("thinkingConfig", None)
                continue
            if r.status_code in (400, 401, 403, 404):
                raise GeminiError(f"API hatası {r.status_code}: {msg}", permanent=True)
            if r.status_code == 429 and re.search(r"per ?day|daily", msg, re.I):
                raise GeminiError("Günlük ücretsiz kota doldu. Kota Pasifik saatiyle gece yarısı sıfırlanır "
                                  "(Türkiye'de yaklaşık 10:00-11:00). Kalan sayfalar yarın kaldığı yerden devam eder; "
                                  "ya da Ayarlar'dan daha yüksek günlük kotalı (Flash-Lite / 3.8 Flash) bir model seçin.", permanent=True)
            if r.status_code in (429, 500, 502, 503, 504):
                wait = _retry_wait(r, attempt)
                self.log(f"API {r.status_code} (hız sınırı/yoğunluk), {wait:.0f} sn bekleniyor...")
                self._sleep(wait)
                continue
            raise GeminiError(f"API hatası {r.status_code}: {msg}")
        raise GeminiError("API çok kez başarısız oldu (yeniden deneme bitti).")

    def list_models(self):
        r = requests.get(f"{API}/models?pageSize=100", headers={"x-goog-api-key": self.key}, timeout=30)
        if r.status_code != 200:
            raise GeminiError(f"Model listesi alınamadı ({r.status_code}).", permanent=True)
        return sorted(m["name"].split("/")[-1] for m in r.json().get("models", [])
                      if "generateContent" in m.get("supportedGenerationMethods", []))

    def translate_texts(self, items, glossary=(), context=(), free_ids=()):
        """items: [(id, text)] -> (dict id->tr, new_terms, summary). Yalnızca metin gönderilir (çok az token)."""
        gl = ""
        if glossary:
            gl = "\n- Glossary (use EXACTLY these Turkish forms):\n" + "\n".join(f"  {a} = {b}" for a, b in list(glossary)[:250])
        cx = ""
        if context:
            cx = "\n- Story so far:\n" + "\n".join(f"  * {c}" for c in context)
        st = self._extra_instructions()
        free_ids = set(free_ids or ())
        payload = json.dumps([dict(id=i, text=t, **({"free": True} if i in free_ids else {})) for i, t in items],
                             ensure_ascii=False)
        body = {"contents": [{"parts": [{"text": TEXT_PROMPT.format(style=st, glossary=gl, context=cx, payload=payload)}]}],
                "generationConfig": self._gen_config()}
        last = None
        for _ in range(2):
            data = self._post(body)
            try:
                text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
                d = _parse_json(text)
                out = {str(i["id"]): str(i.get("tr", "")).strip() for i in d.get("items", []) if isinstance(i, dict) and "id" in i}
                terms = [(str(t.get("en", "")), str(t.get("tr", ""))) for t in d.get("new_terms", []) or []
                         if isinstance(t, dict) and t.get("en") and t.get("tr")]
                return out, terms, str(d.get("summary", ""))
            except (KeyError, IndexError, ValueError, TypeError) as e:
                last = GeminiError(f"Yanıt okunamadı ({e}).")
        raise last

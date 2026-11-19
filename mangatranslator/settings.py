"""Ayarlar, uygulama klasörleri ve API anahtarı saklama."""
import json, os, sys
from pathlib import Path

APP_DIR = Path(os.environ.get("APPDATA") or (Path.home() / ".config")) / "MangaTranslator"
PROJECTS_DIR = APP_DIR / "projects"
USER_FONT_DIR = APP_DIR / "fonts"
SETTINGS_FILE = APP_DIR / "settings.json"
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
BUILTIN_FONT_DIR = BASE_DIR / "fonts"

DEFAULTS = {
    "model": "gemini-flash-lite-latest",
    "rpm": 10,                 # dakikada en fazla istek
    "render_long_side": 2200,  # PDF sayfası render boyutu (uzun kenar px)
    "jpeg_quality": 90,
    "translate_sfx": False,    # efekt yazıları (SFX) çevrilsin mi
    "auto_export": True,       # bölüm bitince PDF üret
    "export_cbz": False,       # PDF'e ek olarak .cbz de üret
    "eco_batch": 60,           # istek başına en fazla kutu sayısı
    "ocr_engine": "auto",      # OCR: "auto" (RapidOCR varsa o) | "rapid" | "tesseract"
    "inpaint": "opencv",       # desenli zeminde yazı silme: "opencv" (hızlı) | "manga_lama" (kaliteli, ~207 MB model)
    "erase_nearby": True,      # OCR'ın kaçırdığı yakın harfleri ("HE'S", "T...Y") de sil (modelsiz, hızlı)
    "text_detector_model": False,  # bunun için yazı bulucu model kullan (~95 MB; yavaş işlemcide sayfa başı dakikalar)
    "text_snap": True,         # kutuları sayfadaki gerçek yazıya hizala (silme alanı yazıya oturur)
    "tesseract_path": "",
    "rtl": True,               # okuma sırası sağdan sola (manga)
    "font_map": {},            # {stil: font dosyası}
}


def ensure_dirs():
    for d in (APP_DIR, PROJECTS_DIR, USER_FONT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load():
    ensure_dirs()
    s = dict(DEFAULTS)
    try:
        s.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    except Exception:
        pass
    return s


def save(s):
    ensure_dirs()
    tmp = SETTINGS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, SETTINGS_FILE)


def get_api_key_info():
    """-> (anahtar, kaynak). Öncelik: Ayarlar'da kaydedilen (Kimlik Bilgileri Yöneticisi / dosya), sonra ortam değişkeni."""
    try:
        import keyring
        k = keyring.get_password("MangaTranslator", "gemini")
        if k:
            return k, "Windows Kimlik Bilgileri Yöneticisi"
    except Exception:
        pass
    try:
        k = json.loads((APP_DIR / "key.json").read_text()).get("key", "")
        if k:
            return k, "yerel dosya (key.json)"
    except Exception:
        pass
    k = os.environ.get("GEMINI_API_KEY", "")
    return (k, "ortam değişkeni GEMINI_API_KEY") if k else ("", "yok")


def get_api_key():
    return get_api_key_info()[0].strip()


def delete_api_key():
    """Kayıtlı anahtarı (Kimlik Bilgileri Yöneticisi + yerel dosya) siler. Ortam değişkenine dokunmaz."""
    try:
        import keyring
        keyring.delete_password("MangaTranslator", "gemini")
    except Exception:
        pass
    try:
        (APP_DIR / "key.json").unlink()
    except OSError:
        pass


def set_api_key(key):
    """Windows Credential Manager (keyring) varsa orada, yoksa yerel dosyada saklar."""
    try:
        import keyring
        keyring.set_password("MangaTranslator", "gemini", key)
        try:
            (APP_DIR / "key.json").unlink()
        except OSError:
            pass
        return "keyring"
    except Exception:
        ensure_dirs()
        (APP_DIR / "key.json").write_text(json.dumps({"key": key}))
        return "dosya"

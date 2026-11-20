"""Çeviri kuyruğu: 1) yerel OCR 2) metinleri toplu çeviri (az istek) 3) silme + dizgi.
Her aşamanın sonucu sayfa bazlı diske yazılır; durdurup kaldığı yerden devam edilebilir."""
import json, os, re, threading, time
import numpy as np
from . import source, export, ocr
from .chapters import effective, fmt_num, sort_chapters, label as ch_label
from .gemini import Gemini, GeminiError, GeminiTimeout, Stopped
from .render import render_page


def atomic_write(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    os.replace(tmp, path)


def save_json(path, obj):
    atomic_write(path, json.dumps(obj, ensure_ascii=False))


def save_jpg(path, img, q):
    tmp = path.with_name(path.name + ".tmp")
    img.save(tmp, "JPEG", quality=q)
    os.replace(tmp, path)


def pending(r):
    """Henüz çevrilmemiş (ve 'çevirme' işaretlenmemiş) bölge."""
    return not str(r.get("tr", "")).strip() and not r.get("skip")


def relevant_terms(glossary, texts):
    """Sözlükten yalnızca bu metinlerde geçen terimler (istek kısa kalır, 250 terim sınırına takılmaz)."""
    blob = " ".join(texts).lower()
    out = []
    for a, b in glossary:
        a = str(a).strip()
        if a and re.search(r"(?<!\w)" + re.escape(a.lower()) + r"(?!\w)", blob):
            out.append((a, b))
    return out


def snap_regions(img, regions):
    """Kutuları hizala: her kutuyu (ve silme alanını) sayfadaki gerçek yazıya oturtur."""
    from . import bubbles, clean
    arr = np.asarray(img.convert("RGB"))
    for r in regions:
        bm = bubbles.mask(arr.shape, r["bubble"]) if r.get("bubble") else None
        nb = clean.snap_box(arr, r["box"], bm)
        if bm is not None:
            # Yazının bir kısmı balonun dışına taşıyorsa ("...TO HER." balon kenarının altında) balona
            # oturtulan kutu o kısmı kaybeder; taşan kısım ayrıca silinsin diye ilk kutu saklanır.
            H, W = arr.shape[:2]
            y0, x0, y1, x1 = r["box"]
            px = (int(x0 * W / 1000), int(y0 * H / 1000), int(np.ceil(x1 * W / 1000)), int(np.ceil(y1 * H / 1000)))
            sub = bm[px[1]:px[3], px[0]:px[2]]
            if sub.size and float((sub == 0).mean()) > 0.03:
                r["spill_box"] = list(r["box"])
        r["box"], r["clean_box"] = nb, list(nb)


def fallback_note(a):
    """render_page seçili silme modelini kullanamadıysa günlüğe yazılacak kısa uyarı (yoksa boş)."""
    fb = a.pop("inpaint_fallback", None) if isinstance(a, dict) else None
    if not fb:
        return ""
    from . import models
    name = models.inpaint_label(fb.get("kind", "")).split(" (")[0]
    where = f"{fb['n']} yerde" if fb.get("n") else "bu sayfada"
    why = f" Sebep: {fb['err']}" if fb.get("err") else ""
    return f"Uyarı: {name} {where} çalışmadı, o kısımlar OpenCV ile silindi.{why}"


class Pipeline:
    def __init__(self, project, cfg, api_key, stop, log=print, progress=lambda *a: None, client=None):
        self.p, self.cfg, self.stop, self.log, self.progress = project, cfg, stop, log, progress
        self.client = client or Gemini(
            api_key,
            cfg["model"],
            cfg.get("rpm", 10),
            stop,
            log,
            thinking_level=cfg.get("thinking_level", "medium"),
            translation_style=cfg.get("translation_style", "natural"),
            custom_prompt=cfg.get("custom_prompt", ""),
        )
        kind = cfg.get("inpaint", "opencv")
        if kind != "opencv":
            from . import models
            # Model burada YÜKLENMEZ (Başlat'a basınca pencere donmasın); ilk sayfada arka planda yüklenir.
            if not models.inpaint_ready(kind):
                log("Uyarı: " + models.inpaint_label(kind) + " seçili ama kullanılamıyor ("
                    + models.inpaint_status(kind) + "); OpenCV ile silinecek.")

    def ensure_text_detector(self):
        """Yazı bulucu model yoksa bir kez indirir (~95 MB). İnemezse uyarır, çeviri yine de sürer."""
        if not self.cfg.get("text_detector_model", False):
            return
        from . import models
        if models.text_det_ready():
            return
        self.log("Yazı bulucu model indiriliyor (~95 MB, yalnızca bir kez)...")
        last = [0]

        def prog(got, total):
            mb = got >> 20
            if mb - last[0] >= 10:
                last[0] = mb
                self.log(f"  {mb} / {total >> 20 if total else '?'} MB")
        try:
            models.download_text_det(prog, self.stop)
            self.log("Yazı bulucu model hazır.")
        except Exception as e:
            self.log(f"Uyarı: yazı bulucu model indirilemedi ({e}); yalnızca OCR kutuları silinecek.")

    def run(self, chapter_ids):
        self.ensure_text_detector()
        chs = sort_chapters([c for c in self.p.chapters() if c["id"] in set(chapter_ids)])
        total = sum(c["n_pages"] for c in chs)
        base = 0
        for ch in chs:
            if self.stop.is_set():
                break
            try:
                self.process(ch, base, total)
            except Stopped:
                self.p.update_chapter(ch["id"], status="durduruldu")
                self.log("İşlem kullanıcı tarafından durduruldu.")
                break
            except GeminiError as e:       # kalıcı hata (anahtar/model/kota): kuyruğu durdur
                self.p.update_chapter(ch["id"], status="hata")
                self.log(f"KRİTİK HATA: {e}")
                raise
            base += ch["n_pages"]
        return not self.stop.is_set()

    def _finish_status(self, cid, wd, n, failed_pages=0):
        """Bölüm durumunu sayfa dosyalarına bakarak yazar (çevrilemeyen balonlar 'kısmi' gösterilir)."""
        done = sum(1 for i in range(n) if (wd / f"p{i:04d}.jpg").exists())
        missing = 0
        for i in range(n):
            try:
                a = json.loads((wd / f"p{i:04d}.json").read_text(encoding="utf-8"))
                missing += sum(1 for r in a.get("regions", []) if pending(r))
            except Exception:
                pass
        parts = []
        if failed_pages:
            parts.append(f"{failed_pages} sayfa hatalı")
        if missing:
            parts.append(f"{missing} balon çevrilemedi")
        status = "çevrildi" if not parts and done == n else ("kısmi (" + ", ".join(parts) + ")" if parts else "kısmi")
        self.p.update_chapter(cid, done=done, failed=failed_pages, status=status)
        return done, missing, status

    def redraw(self, chapter_ids):
        """Kayıtlı çevirilerden (json) sayfaları yeniden çizer; API'ye istek atmaz, kota harcamaz."""
        chs = sort_chapters([c for c in self.p.chapters() if c["id"] in set(chapter_ids)])
        total = sum(c["n_pages"] for c in chs) or 1
        base, n_done = 0, 0
        for ch in chs:
            wd = self.p.work_dir(ch["id"])
            label = f"{ch_label(ch)} {fmt_num(effective(ch))}"
            self.log(f"[{label}] Yeniden çizim başlatıldı (API kotası kullanılmayacak)...")
            for i in range(ch["n_pages"]):
                if self.stop.is_set():
                    return False
                js = wd / f"p{i:04d}.json"
                self.progress(base + i, total, f"{label} — Sayfa {i + 1}/{ch['n_pages']}: Yazılar siliniyor ve Türkçe dizgi çiziliyor...")
                if not js.exists():
                    continue
                try:
                    a = json.loads(js.read_text(encoding="utf-8"))
                    img = source.load_page(ch, i, self.cfg.get("render_long_side", 2200))
                    out = render_page(img, a, self.cfg, self.stop)
                    note = fallback_note(a)
                    if note:
                        self.log(f"[{label}] Sayfa {i + 1}: {note}")
                    save_json(js, a)
                    save_jpg(wd / f"p{i:04d}.jpg", out, int(self.cfg.get("jpeg_quality", 90)))
                    n_done += 1
                except Stopped:
                    return False
                except Exception as e:
                    self.log(f"[{label}] Sayfa {i + 1} çizilemedi: {e}")
            if ch["status"] not in ("bekliyor", "durduruldu", "hata"):
                self._finish_status(ch["id"], wd, ch["n_pages"])
            if self.cfg.get("auto_export", True) and any((wd / f"p{i:04d}.jpg").exists() for i in range(ch["n_pages"])):
                self.progress(base + ch["n_pages"], total, f"{label} — PDF dosyası oluşturuluyor...")
                export.export_all(self.p, self.p.chapter(ch["id"]), self.cfg)
            base += ch["n_pages"]
        self.log(f"Tamamlandı: {n_done} sayfa yeniden çizildi.")
        return True

    def _translate_split(self, todo, gl, ctx, free):
        """Yanıt zaman aşımına uğrarsa paketi ikiye bölüp parça parça çevirir (takılı kalmaz)."""
        try:
            return self.client.translate_texts(todo, gl, ctx, free_ids=free)
        except GeminiTimeout:
            if len(todo) < 8:
                raise
            half = len(todo) // 2
            self.log(f"  ! Yanıt gecikti; {len(todo)} metin iki parçaya bölünüp yeniden gönderiliyor...")
            o1, t1, s1 = self._translate_split(todo[:half], gl, ctx, free)
            o2, t2, s2 = self._translate_split(todo[half:], gl, ctx, free)
            return {**o1, **o2}, list(t1) + list(t2), s2 or s1

    def translate_regions(self, regions_by_page, ctx=()):
        """regions_by_page: [(sayfa_no, [bölge, ...])]. Bekleyen bölgeleri çevirir, sonuçları bölgelere yazar.

        Gemini'nin bilerek boş döndürdüğü metin (gürültü/filigran) 'çevirme' olarak işaretlenir.
        Yanıtta hiç olmayan metinler bir kez daha istenir; yine gelmezse bekleyen kalır
        (sonraki çalıştırmada yeniden denenir, Metinler ekranında 'ÇEVRİLEMEDİ' görünür).
        Dönüş: (değişen sayfa numaraları, çevrilemeyen bölge sayısı, özet)."""
        items, ref = [], {}
        for i, regs in regions_by_page:
            for k, r in enumerate(regs):
                if pending(r):
                    key = f"{i}_{k}"
                    items.append((key, r.get("src", "")))
                    ref[key] = (i, r)
        if not items:
            return set(), 0, ""
        glossary = [(a, b) for a, b, _ in self.p.glossary()]
        changed, summary, todo = set(), "", items
        for attempt in range(2):
            gl = relevant_terms(glossary, [t for _, t in todo])
            free = {key for key, _ in todo if ref[key][1].get("free")}
            out, terms, summ = self._translate_split(todo, gl, list(ctx)[-3:], free)
            summary = summary or summ
            for en, tr in terms:
                self.p.add_auto_term(en, tr)
            missing = []
            for key, txt in todo:
                i, r = ref[key]
                if key in out:
                    r["tr"] = out[key]
                    r["skip"] = not out[key]
                    r.pop("missing", None)
                    changed.add(i)
                else:
                    missing.append((key, txt))
            todo = missing
            if not todo:
                break
            if attempt == 0:
                self.log(f"  ! {len(todo)} metin yanıtta yoktu, yeniden isteniyor...")
        for key, _ in todo:
            i, r = ref[key]
            r["missing"] = True
            changed.add(i)
        return changed, len(todo), summary

    def _start_text_masks(self, ch, n, failed, js, jp):
        """Çizilecek sayfaların yazı maskelerini ayrı iş parçacığında hesaplar (sonuç önbelleğe yazılır)."""
        if not self.cfg.get("text_detector_model", False):
            return None
        from . import models
        if not models.text_det_ready():
            return None
        pages = []
        for i in range(n):
            if i in failed or jp(i).exists() or not js(i).exists():
                continue
            try:
                a = json.loads(js(i).read_text(encoding="utf-8"))
            except Exception:
                continue
            if a.get("regions"):
                pages.append(i)
        if not pages:
            return None
        state = [0, len(pages)]
        side = self.cfg.get("render_long_side", 2200)

        def work():
            for i in pages:
                if self.stop.is_set():
                    return
                try:
                    img = source.load_page(ch, i, side)
                    models.text_mask(np.asarray(img.convert("RGB")))
                except Exception as e:
                    self.log(f"  ! Yazı bulucu sayfa {i + 1}: {e}")
                state[0] += 1
        t = threading.Thread(target=work, name="yazi-bulucu", daemon=True)
        t.start()
        return t, state

    def process(self, ch, base, total):
        """1) yerel OCR (API yok) 2) metinleri toplu çevir (az istek) 3) çiz."""
        p, cfg = self.p, self.cfg
        wd = p.work_dir(ch["id"])
        n = source.page_count(ch["src"], ch["kind"])
        if n != ch["n_pages"]:
            p.update_chapter(ch["id"], n_pages=n)
            ch["n_pages"] = n
        p.update_chapter(ch["id"], status="çevriliyor")
        label = f"{ch_label(ch)} {fmt_num(effective(ch))}"
        failed = set()
        js = lambda i: wd / f"p{i:04d}.json"
        jp = lambda i: wd / f"p{i:04d}.jpg"
        prog = lambda k, txt: self.progress(base + int(n * k / 3), total, f"{label} — {txt}")

        self.log(f"[{label}] Çeviri başlatıldı ({n} sayfa). Aşama 1/3: Yerel OCR metin taraması...")

        # 1) OCR
        for i in range(n):
            if self.stop.is_set():
                raise Stopped()
            if js(i).exists():
                continue
            prog(i / max(1, n), f"[1/3 OCR] Sayfa {i + 1}/{n} taranıyor (metinler ve balonlar bulunuyor)...")
            try:
                img = source.load_page(ch, i, cfg.get("render_long_side", 2200))
                if ocr.is_skipped_color_page(img):
                    save_json(js(i), dict(summary="", new_terms=[], regions=[], eco=True, color_skipped=True))
                    self.log(f"  • Sayfa {i + 1}/{n}: renkli sayfa, şimdilik çevrilmiyor (orijinal haliyle kalır).")
                    continue
                self.log(f"  … Sayfa {i + 1}/{n}: OCR çalışıyor (ilk sayfada modeller yüklenir, 1-2 dk sürebilir)...")
                regs = ocr.detect(img, cfg.get("tesseract_path", ""), cfg.get("rtl", True), cfg.get("ocr_engine", "auto"))
                if cfg.get("text_snap", True):
                    snap_regions(img, regs)
                save_json(js(i), dict(summary="", new_terms=[], regions=regs, eco=True,
                                         dropped=list(ocr.last_dropped)))
                self.log(f"  • Sayfa {i + 1}/{n}: OCR tamamlandı ({len(regs)} metin bölgesi bulundu).")
                if not getattr(self, "_ocr_named", False):
                    self._ocr_named = True
                    from . import models as _m
                    if _m._rapid and cfg.get("ocr_engine", "auto") != "tesseract":
                        self.log(f"  (OCR motoru: RapidOCR, {_m.rapid_kind()})")
                    elif _m._rapid is False or cfg.get("ocr_engine", "auto") == "tesseract":
                        self.log("  (OCR motoru: Tesseract)")
            except ocr.OCRError as e:
                raise GeminiError(str(e), permanent=True)
            except Exception as e:
                failed.add(i)
                self.log(f"  ! Sayfa {i + 1} OCR hatası: {e}")
            if self.stop.is_set():
                raise Stopped()

        # Yazı bulucu, Gemini yanıtı beklenirken arka planda sayfaları hazırlar (3. aşama beklemez)
        pre = self._start_text_masks(ch, n, failed, js, jp)

        # 2) Toplu çeviri
        pend = []
        for i in range(n):
            if i in failed or not js(i).exists():
                continue
            a = json.loads(js(i).read_text(encoding="utf-8"))
            if any(pending(r) for r in a["regions"]):
                pend.append((i, a))

        limit, ctx, batches, cur, cnt = int(cfg.get("eco_batch", 60)), [], [], [], 0
        for i, a in pend:
            k = sum(1 for r in a["regions"] if pending(r))
            if cur and cnt + k > limit:
                batches.append(cur)
                cur, cnt = [], 0
            cur.append((i, a))
            cnt += k
        if cur:
            batches.append(cur)

        if batches:
            self.log(f"[{label}] Aşama 2/3: Metinler Gemini'ye gönderiliyor ({len(batches)} paket)...")
        else:
            self.log(f"[{label}] Aşama 2/3: Çevrilecek yeni metin yok, çizime geçiliyor.")

        for bi, batch in enumerate(batches):
            if self.stop.is_set():
                raise Stopped()
            n_items = sum(1 for _, a in batch for r in a["regions"] if pending(r))
            p_start, p_end = batch[0][0] + 1, batch[-1][0] + 1
            prog(1 + bi / max(1, len(batches)), f"[2/3 Gemini Çeviri] Paket {bi + 1}/{len(batches)} (Sayfa {p_start}-{p_end}, {n_items} balon) çevriliyor...")
            self.log(f"  • Paket {bi + 1}/{len(batches)} (Sayfa {p_start}-{p_end}, {n_items} metin) Gemini'de çevriliyor...")
            try:
                changed, n_miss, summ = self.translate_regions([(i, a["regions"]) for i, a in batch], ctx)
            except GeminiError as e:
                if e.permanent:
                    raise
                for i, _ in batch:
                    failed.add(i)
                self.log(f"  ! Paket {bi + 1} atlandı: {e}")
                continue
            if summ:
                ctx.append(summ)
            if n_miss:
                self.log(f"  ! Paket {bi + 1}: {n_miss} metin çevrilemedi; orijinal kalacak, sonraki çalıştırmada yeniden denenecek.")
            for i, a in batch:
                save_json(js(i), a)
                if i in changed and jp(i).exists():
                    jp(i).unlink()          # yeni çeviri geldi: sayfa yeniden çizilsin

        # 3) Çizim (Temizleme + Dizgi)
        self.log(f"[{label}] Aşama 3/3: Orijinal yazılar siliniyor ve Türkçe metinler yerleştiriliyor...")
        if pre is not None and pre[0].is_alive():
            self.log(f"  … Yazı bulucu sayfaları hazırlıyor ({pre[1][0]}/{pre[1][1]})...")
            shown = pre[1][0]
            while pre[0].is_alive():
                if self.stop.is_set():
                    raise Stopped()
                pre[0].join(0.5)
                if pre[1][0] != shown:
                    shown = pre[1][0]
                    prog(2, f"[3/3] Yazı bulucu: sayfa {shown}/{pre[1][1]} hazırlandı...")
        for i in range(n):
            if self.stop.is_set():
                raise Stopped()
            if i in failed or jp(i).exists() or not js(i).exists():
                continue
            prog(2 + i / max(1, n), f"[3/3 Temizleme ve Dizgi] Sayfa {i + 1}/{n}: Yazılar siliniyor ve Türkçe yerleştiriliyor...")
            try:
                a = json.loads(js(i).read_text(encoding="utf-8"))
                img = source.load_page(ch, i, cfg.get("render_long_side", 2200))
                t0 = time.time()
                out = render_page(img, a, cfg, self.stop)
                note = fallback_note(a)
                save_json(js(i), a)
                save_jpg(jp(i), out, int(cfg.get("jpeg_quality", 90)))
                self.log(f"  • Sayfa {i + 1}/{n} çizildi ({time.time() - t0:.0f} sn).")
                if note:
                    self.log(f"  ! Sayfa {i + 1}: {note}")
            except Stopped:
                raise
            except Exception as e:
                failed.add(i)
                self.log(f"  ! Sayfa {i + 1} çizim hatası: {e}")

        if cfg.get("auto_export", True):
            prog(3, "Çıktı PDF/CBZ dosyası hazırlanıyor...")
            self.log(f"[{label}] PDF/CBZ çıktısı oluşturuluyor...")
            export.export_all(p, p.chapter(ch["id"]), cfg)
        done, missing, status = self._finish_status(ch["id"], wd, n, len(failed))
        extra = f", {missing} balon çevrilemedi" if missing else ""
        self.log(f"[{label}] Bölüm tamamlandı ({done}/{n} sayfa{extra}).")
        self.progress(base + n, total, f"{label} — Tamamlandı ({done}/{n} sayfa)")

"""Yerel ONNX modelleri (CPU): RapidOCR (yazı okuma) ve Manga-LaMa (yazı silme)."""
import hashlib
import os
import threading
import numpy as np
from . import settings as S

MODEL_DIR = S.APP_DIR / "models"
INPAINT_MODELS = {
    "manga_lama": {
        "label": "Manga-LaMa (CPU ONNX, manga için ince ayarlı, ~207 MB)",
        "url": "https://huggingface.co/mayocream/lama-manga-onnx/resolve/main/lama-manga.onnx",
        "sha256": None,
        "file": MODEL_DIR / "lama_manga.onnx",
    },
    # AOT (manga-image-translator'ın manga ile eğittiği model) ve MI-GAN, uygulamanın giriş düzenine
    # (image 0..1, mask 1 = silinecek, çıkış 0..1; 512x512) çevrilmiş ONNX dosyaları. İndirme adresi yok:
    # dosyalar models klasörüne doğrudan konur.
    "aot": {
        "label": "AOT (manga için eğitilmiş, LaMa'dan ~2,5 kat hızlı, ~23 MB) — önerilen",
        "url": None,
        "sha256": None,
        "file": MODEL_DIR / "aot_manga.onnx",
    },
    "migan": {
        "label": "MI-GAN (en hızlı, koyu çizgili zeminde beyaz leke bırakabilir, ~13 MB)",
        "url": None,
        "sha256": None,
        "file": MODEL_DIR / "migan.onnx",
    },
}

TEXT_DET = {
    "label": "Yazı bulucu (comic-text-detector, CPU ONNX, ~95 MB)",
    "url": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx",
    "sha256": "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f",
    "file": MODEL_DIR / "comic_text_detector.onnx",
}
# Yalnızca yazı maskesi (seg) çıkışını hesaplayan, gereksiz dalları ayıklanmış sürüm (aynı sonuç, ~2 kat hızlı).
# Varsa bu kullanılır; yoksa tam model.
TEXT_DET_SEG = MODEL_DIR / "comic_text_detector_seg.onnx"
TEXT_DET_SIZE = 1024
TEXT_MASK_CACHE = S.APP_DIR / "cache" / "textmask"

_lock = threading.Lock()
_det_lock = threading.Lock()
_det = None
_det_err = ""
_mask_mem = {}
_rapid = None
_rapid_err = ""
_inp_sessions = {}
_inp_errs = {}
_inp_failed = {}


def _ort_session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    # bir çekirdek arayüze bırakılır; model çalışırken Python kilidi (GIL) serbesttir, pencere donmaz
    so.intra_op_num_threads = max(1, min(8, (os.cpu_count() or 2) - 1))
    so.inter_op_num_threads = 1
    # Çok küçük (denormal) ara değerler CPU'yu 5-50 kat yavaşlatır; sıfır saymak sonucu değiştirmez.
    try:
        so.add_session_config_entry("session.set_denormal_as_zero", "1")
    except Exception:
        pass
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------- RapidOCR
def _pkg_ver(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "?"


_rapid_kind = None      # yüklenen RapidOCR: "yeni sürüm 3.10.0, PP-OCRv6" / "eski sürüm 1.4.4, PP-OCRv4"


def _make_rapid():
    # Yeni RapidOCR (3.x, PP-OCRv6) kuruluysa o kullanılır: manga sayfalarında kelimeleri çok daha az
    # yanlış okur (bitişik/yanlış kelime oranı ~%9 yerine ~%3). Yoksa ya da açılamazsa eski rapidocr_onnxruntime.
    global _rapid_kind
    new_err = None
    try:
        from rapidocr import RapidOCR
        eng = RapidOCR()
        _rapid_kind = f"yeni sürüm {_pkg_ver('rapidocr')}, PP-OCRv6"
        return eng
    except ImportError:
        pass
    except Exception as e:
        new_err = f"{e.__class__.__name__}: {e}"
    from rapidocr_onnxruntime import RapidOCR
    eng = RapidOCR()
    _rapid_kind = f"eski sürüm {_pkg_ver('rapidocr_onnxruntime')}, PP-OCRv4"
    if new_err:
        _rapid_kind += f"; yeni sürüm açılamadı: {new_err}"
    return eng


def rapid_kind():
    """Hangi RapidOCR'ın kullanıldığını (yüklendiyse) ya da kullanılacağını (yüklemeden) söyler."""
    if _rapid_kind:
        return _rapid_kind
    import importlib.util
    try:
        if importlib.util.find_spec("rapidocr") is not None:
            return f"yeni sürüm {_pkg_ver('rapidocr')}, PP-OCRv6"
        if importlib.util.find_spec("rapidocr_onnxruntime") is not None:
            return f"eski sürüm {_pkg_ver('rapidocr_onnxruntime')}, PP-OCRv4"
    except Exception:
        pass
    return ""


def rapid():
    global _rapid, _rapid_err
    with _lock:
        if _rapid is None:
            try:
                _rapid = _make_rapid()
            except Exception as e:
                _rapid = False
                _rapid_err = f"{e.__class__.__name__}: {e}"
        return _rapid or None


def rapid_installed():
    """RapidOCR kurulu mu? Modeli YÜKLEMEZ (Ayarlar penceresi donmasın diye)."""
    import importlib.util
    try:
        return any(importlib.util.find_spec(m) is not None for m in ("rapidocr_onnxruntime", "rapidocr"))
    except Exception:
        return False


def rapid_status(load=False):
    """load=False: yalnızca kurulu mu bakar. Model yüklemek (onnxruntime) Python'u saniyelerce kilitleyip
    arayüzü dondurabildiği için Ayarlar ekranı yüklemeden kontrol eder; model ilk OCR'da yüklenir."""
    if load:
        rapid()
    if _rapid:
        return f"hazır ({rapid_kind()})"
    if _rapid is False:
        return f"kullanılamıyor ({_rapid_err or 'kurulu değil'})"
    return f"kurulu, {rapid_kind()} (ilk OCR'da yüklenecek)" if rapid_installed() else "kurulu değil"


def _parse_rapid(res):
    if res is None:
        return []
    if hasattr(res, "boxes"):
        if res.boxes is None or res.txts is None:
            return []
        scores = res.scores if res.scores is not None else [1.0] * len(res.txts)
        return [(np.asarray(b).tolist(), str(t), float(s)) for b, t, s in zip(res.boxes, res.txts, scores)]
    result = res[0] if isinstance(res, tuple) else res
    if not result:
        return []
    out = []
    for item in result:
        try:
            out.append((np.asarray(item[0]).tolist(), str(item[1]), float(item[2])))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def rapid_lines(rgb):
    eng = rapid()
    if eng is None:
        return None
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    with _lock:
        try:
            res = eng(bgr, use_cls=False)
        except TypeError:
            res = eng(bgr)
    lines = []
    for box, txt, score in _parse_rapid(res):
        txt = txt.strip()
        if not txt:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        conf = float(score) * 100 if score <= 1 else float(score)
        ang = 0.0
        try:                                       # satırın eğimi (üst kenar): tabela/levha yazıları eğik olur
            (ax, ay), (bx, by) = box[0], box[1]
            ang = float(np.degrees(np.arctan2(by - ay, bx - ax)))
        except Exception:
            pass
        lines.append(dict(
            angle=ang,
            t=txt,
            x0=float(min(xs)),
            y0=float(min(ys)),
            x1=float(max(xs)),
            y1=float(max(ys)),
            conf=conf,
        ))
    return lines


def _first_text(x):
    if isinstance(x, str):
        return x
    if hasattr(x, "txts") and x.txts:
        return str(x.txts[0])
    if isinstance(x, (list, tuple)):
        for it in x:
            t = _first_text(it)
            if t:
                return t
    return None


def rapid_rec(rgb):
    eng = rapid()
    if eng is None:
        return None
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    with _lock:
        try:
            res = eng(bgr, use_det=False, use_cls=False, use_rec=True)
        except Exception:
            return None
    if isinstance(res, tuple):
        res = res[0]
    t = _first_text(res)
    return t.strip() if t else None


# ---------------------------------------------------------------- Yazı Silme (Inpainting)
def inpaint_spec(kind):
    return INPAINT_MODELS.get(kind)


def inpaint_choices():
    out = [("opencv", "OpenCV — hızlı, ek indirme yok (düz beyaz balonlar için)")]
    out += [(k, v["label"]) for k, v in INPAINT_MODELS.items()]
    return out


def inpaint_label(kind):
    if kind == "opencv":
        return "OpenCV"
    spec = inpaint_spec(kind)
    return spec["label"] if spec else kind


def inpaint_ready(kind="manga_lama"):
    spec = inpaint_spec(kind)
    return bool(spec and spec["file"].exists() and spec["file"].stat().st_size > 1_000_000)


def download_inpaint(kind="manga_lama", progress=None, stop=None):
    spec = INPAINT_MODELS.get(kind)
    if not spec or not spec.get("url"):
        raise ValueError(f"Bu model otomatik indirilmez; dosyayı şu konuma koyun: {spec['file'] if spec else kind}")
    out = _download(spec, progress, stop)
    with _lock:
        _inp_sessions.pop(kind, None)
        _inp_errs.pop(kind, None)
        _inp_failed.pop(kind, None)
    return out


def _download(spec, progress=None, stop=None):
    import requests
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = spec["file"]
    tmp = target.with_suffix(".part")
    h = hashlib.sha256()
    with requests.get(spec["url"], stream=True, timeout=90) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if stop is not None and stop.is_set():
                    raise RuntimeError("İndirme durduruldu.")
                f.write(chunk)
                h.update(chunk)
                got += len(chunk)
                if progress:
                    progress(got, total)
    if got < 1_000_000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("İndirilen model dosyası eksik veya bozuk.")
    if spec["sha256"] and h.hexdigest() != spec["sha256"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("İndirilen model doğrulanamadı (SHA-256 uyuşmuyor).")
    os.replace(tmp, target)
    return target, h.hexdigest()


# ---------------------------------------------------------------- Yazı bulucu (piksel düzeyinde yazı maskesi)
def text_det_ready():
    f = TEXT_DET["file"]
    return f.exists() and f.stat().st_size > 50_000_000


def download_text_det(progress=None, stop=None):
    global _det, _det_err
    out = _download(TEXT_DET, progress, stop)
    with _det_lock:
        _det, _det_err = None, ""
    return out


def text_det_status():
    if _det:
        return "hazır"
    if _det is False:
        return f"kullanılamıyor ({_det_err})"
    return "indirildi (ilk sayfada yüklenir)" if text_det_ready() else "indirilmemiş (ilk çeviride otomatik iner)"


def _det_session():
    global _det, _det_err
    with _det_lock:
        if _det is None:
            if not text_det_ready():
                return None
            try:
                f = TEXT_DET_SEG if (TEXT_DET_SEG.exists() and TEXT_DET_SEG.stat().st_size > 30_000_000) else TEXT_DET["file"]
                _det = _ort_session(f)
            except Exception as e:
                _det, _det_err = False, f"{e.__class__.__name__}: {e}"
        return _det or None


def text_mask(rgb):
    """Sayfadaki TÜM yazıların piksel maskesi (uint8 0/1, sayfa boyutunda) ya da None.
    OCR'ın okuyamadığı süslü harfleri de bulur; silme alanını tamamlamak için kullanılır.
    Sonuç diskte önbelleğe alınır (düzenleyicide yeniden çizim hızlı olsun)."""
    import cv2
    key = hashlib.md5(np.ascontiguousarray(rgb).tobytes()).hexdigest() + f"_{rgb.shape[1]}x{rgb.shape[0]}"
    if key in _mask_mem:
        return _mask_mem[key]
    path = TEXT_MASK_CACHE / f"{key}.png"
    if path.exists():
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is not None and m.shape == rgb.shape[:2]:
            m = (m > 127).astype(np.uint8)
            _remember(key, m)
            return m
    sess = _det_session()
    if sess is None:
        return None
    h, w = rgb.shape[:2]
    r = TEXT_DET_SIZE / max(h, w)
    nh, nw = max(1, int(round(h * r))), max(1, int(round(w * r)))
    can = np.zeros((TEXT_DET_SIZE, TEXT_DET_SIZE, 3), np.uint8)
    can[:nh, :nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    x = can.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    names = [o.name for o in sess.get_outputs()]
    with _det_lock:
        outs = sess.run(["seg"] if "seg" in names else None, {sess.get_inputs()[0].name: x})
    seg = None
    for o in outs:
        if o.ndim == 4 and o.shape[1] == 1:
            seg = o[0, 0]
    if seg is None:
        return None
    prob = cv2.resize(seg[:nh, :nw].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    m = (prob > 0.3).astype(np.uint8)
    try:
        TEXT_MASK_CACHE.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), m * 255)
    except Exception:
        pass
    _remember(key, m)
    return m


# ---------------------------------------------------------------- Satır bulucu (RapidOCR'ın kendi modeli, ~5 MB)
_line = None
_line_lock = threading.Lock()


def _line_model_path():
    import importlib.util
    from pathlib import Path
    for pkg in ("rapidocr", "rapidocr_onnxruntime"):
        try:
            spec = importlib.util.find_spec(pkg)
        except Exception:
            spec = None
        if spec is None or not spec.submodule_search_locations:
            continue
        d = Path(list(spec.submodule_search_locations)[0]) / "models"
        c = sorted(d.glob("*det*.onnx")) if d.is_dir() else []
        if c:
            return c[0]
    return None


def line_mask(rgb, side=1536):
    """RapidOCR'ın yazı bulma modeliyle sayfadaki yazı şeritleri (uint8 0/1) ya da None.
    Çok hafiftir (sayfa başı ~0,3 sn); koyu/desenli zeminde kaçak harfleri bulmak için kullanılır."""
    global _line
    import cv2
    with _line_lock:
        if _line is None:
            p = _line_model_path()
            try:
                _line = _ort_session(p) if p else False
            except Exception:
                _line = False
        sess = _line or None
    if sess is None:
        return None
    h, w = rgb.shape[:2]
    r = side / max(h, w)
    nh, nw = max(32, int(round(h * r / 32)) * 32), max(32, int(round(w * r / 32)) * 32)
    x = cv2.resize(np.ascontiguousarray(rgb[:, :, ::-1]), (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    with _line_lock:
        p = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    p = np.asarray(p).reshape(p.shape[-2], p.shape[-1])
    return cv2.resize((p > 0.3).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def text_mask_cached(rgb):
    """Önbellekte var mı (modeli çalıştırmadan)."""
    key = hashlib.md5(np.ascontiguousarray(rgb).tobytes()).hexdigest() + f"_{rgb.shape[1]}x{rgb.shape[0]}"
    return key in _mask_mem or (TEXT_MASK_CACHE / f"{key}.png").exists()


def _remember(key, m):
    if len(_mask_mem) >= 8:
        _mask_mem.pop(next(iter(_mask_mem)))
    _mask_mem[key] = m


def get_inpaint_session(kind="manga_lama"):
    spec = inpaint_spec(kind)
    if not spec:
        return None
    with _lock:
        if kind not in _inp_sessions:
            if not inpaint_ready(kind):
                _inp_errs[kind] = "model indirilmemiş"
                return None
            try:
                _inp_sessions[kind] = _ort_session(spec["file"])
            except Exception as e:
                _inp_sessions[kind] = False
                _inp_errs[kind] = f"{e.__class__.__name__}: {e}"
        return _inp_sessions[kind] or None


def _dim(d, default=512):
    return d if isinstance(d, int) and d > 0 else default


def _inpaint_profile(sess):
    inps = sess.get_inputs()
    if len(inps) == 1 and len(inps[0].shape) == 4 and inps[0].shape[1] == 4:
        return "rgbm"
    if len(inps) == 2:
        ms = [i for i in inps if "mask" in i.name.lower() or (len(i.shape) == 4 and i.shape[1] == 1)]
        if len(ms) == 1:
            im = [i for i in inps if i is not ms[0]][0]
            if len(im.shape) == 4 and im.shape[1] == 3:
                return "image_mask"
    return None


def inspect_inpaint(kind):
    sess = get_inpaint_session(kind)
    if sess is None:
        return "yüklenemedi: " + _inp_errs.get(kind, "bilinmiyor")
    ins = "; ".join(f"{i.name}{list(i.shape)}" for i in sess.get_inputs())
    outs = "; ".join(f"{o.name}{list(o.shape)}" for o in sess.get_outputs())
    prof = _inpaint_profile(sess) or "DESTEKLENMİYOR"
    return f"girişler: {ins} | çıkışlar: {outs} | düzen: {prof}"


def inpaint_status(kind="manga_lama"):
    if kind == "opencv":
        return "hazır (yerleşik)"
    if not inpaint_ready(kind):
        return "indirilmemiş" if kind in INPAINT_MODELS else "dosya bulunamadı"
    if kind not in _inp_sessions:          # yüklemeden durum ver (yükleme arayüzü dondurur)
        return "indirildi (ilk kullanımda yüklenir; doğrulamak için 'Test et')"
    if not _inp_sessions[kind]:
        return f"kullanılamıyor ({_inp_errs.get(kind, '')})"
    if kind in _inp_failed:
        return "hazır ama son çalışmada HATA verdi: " + _inp_failed[kind]
    return "hazır"


def run_inpaint(img, origin, mask, kind="manga_lama", _raise=False):
    import cv2
    sess = get_inpaint_session(kind)
    if sess is None:
        return False
    try:
        prof = _inpaint_profile(sess)
        if prof is None:
            raise RuntimeError("desteklenmeyen model girişleri: " + inspect_inpaint(kind))
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return True
        H, W = img.shape[:2]
        ox, oy = origin
        mx0, mx1 = ox + xs.min(), ox + xs.max() + 1
        my0, my1 = oy + ys.min(), oy + ys.max() + 1
        side = max(192, int(max(mx1 - mx0, my1 - my0) * 1.8) + 32)
        cx, cy = (mx0 + mx1) // 2, (my0 + my1) // 2
        x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
        x1, y1 = min(W, x0 + side), min(H, y0 + side)
        x0, y0 = max(0, x1 - side), max(0, y1 - side)
        crop = img[y0:y1, x0:x1]
        cm = np.zeros(crop.shape[:2], np.uint8)
        full = np.zeros((H, W), np.uint8)
        mh = min(mask.shape[0], H - oy)
        mw = min(mask.shape[1], W - ox)
        if mh > 0 and mw > 0:
            full[oy:oy + mh, ox:ox + mw] = mask[:mh, :mw] > 0
        cm[:] = full[y0:y1, x0:x1]

        inps = sess.get_inputs()
        im_inp = next((i for i in inps if len(i.shape) == 4 and i.shape[1] in (3, 4)), inps[0])
        shp = im_inp.shape
        th, tw = _dim(shp[2]), _dim(shp[3])
        im_res = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        m_res = (cv2.resize(cm, (tw, th), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
        feeds = {}
        if prof == "rgbm":
            feeds[inps[0].name] = np.concatenate([im_res.transpose(2, 0, 1), m_res[None]], axis=0)[None]
        else:
            for inp in inps:
                is_mask = "mask" in inp.name.lower() or (len(inp.shape) == 4 and inp.shape[1] == 1)
                feeds[inp.name] = m_res[None, None] if is_mask else im_res.transpose(2, 0, 1)[None]
        for inp in inps:
            if "float16" in inp.type:
                feeds[inp.name] = feeds[inp.name].astype(np.float16)
        with _lock:
            out = np.asarray(sess.run(None, feeds)[0], dtype=np.float32)
        if out.ndim == 4:
            out = out[0]
        if out.ndim != 3:
            raise RuntimeError(f"beklenmeyen çıkış şekli: {out.shape}")
        if out.shape[0] in (1, 3):
            out = out.transpose(1, 2, 0)
        if out.min() < -0.1:
            out = (out + 1.0) * 127.5
        elif out.max() <= 1.5:
            out = out * 255.0
        if not np.isfinite(out).all():
            raise RuntimeError("model geçersiz (NaN/sonsuz) değer üretti")
        out = np.clip(out, 0, 255).astype(np.uint8)
        out = cv2.resize(out, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
        sel = cm > 0
        crop[sel] = out[sel]
        _inp_failed.pop(kind, None)
        return True
    except Exception as e:
        _inp_failed[kind] = f"{e.__class__.__name__}: {e}"
        if _raise:
            raise
        return False


def selftest_inpaint(kind):
    import time
    if kind == "opencv":
        return True, "OpenCV yerleşik, test gerekmez."
    if not inpaint_ready(kind):
        return False, "model dosyası yok"
    t = time.time()
    try:
        yy, xx = np.mgrid[0:640, 0:640]
        img = np.stack([(xx * 255 // 639), (yy * 255 // 639), np.full_like(xx, 128)], axis=-1).astype(np.uint8)
        orig = img.copy()
        mask = np.zeros((60, 120), np.uint8)
        mask[:] = 1
        ok = run_inpaint(img, (260, 290), mask, kind, _raise=True)
        if not ok:
            return False, "model yüklenemedi: " + _inp_errs.get(kind, "")
        outside = np.ones((640, 640), bool)
        outside[290:350, 260:380] = False
        if (img[outside] != orig[outside]).any():
            return False, "model maske dışındaki pikselleri değiştirdi"
        diff = float(np.abs(img[290:350, 260:380].astype(int) - orig[290:350, 260:380].astype(int)).mean())
        return True, f"test geçti ({time.time() - t:.1f} sn; ort. değişim {diff:.0f})"
    except Exception as e:
        return False, f"{e.__class__.__name__}: {e}"

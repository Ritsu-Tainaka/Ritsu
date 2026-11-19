"""Kaynak tarama (klasör / PDF / CBZ) ve sayfa yükleme. Sayfalar tek tek (akış halinde) okunur."""
import hashlib, io, os, threading, zipfile
from pathlib import Path
from PIL import Image
from .chapters import extract_number, natural_key

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ARCH_EXT = {".cbz", ".zip", ".cbr"}      # .cbr yalnızca içeriği aslında ZIP ise okunur
_lock = threading.Lock()
_cache = {"path": None, "doc": None, "zpath": None, "zip": None}


def _img_files(d):
    return sorted((p for p in Path(d).iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT),
                  key=lambda p: natural_key(p.name))


def _zip_images(path):
    with zipfile.ZipFile(path) as z:
        names = [i.filename for i in z.infolist()
                 if not i.is_dir() and Path(i.filename).suffix.lower() in IMG_EXT
                 and "__MACOSX" not in i.filename and not Path(i.filename).name.startswith("._")]
    return sorted(names, key=natural_key)


def _sig(src, kind):
    h = hashlib.md5()
    files = _img_files(src) if kind == "dir" else [Path(src)]
    for p in files:
        st = p.stat()
        h.update(f"{p.name}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()


def page_count(src, kind):
    if kind == "dir":
        return len(_img_files(src))
    if kind == "cbz":
        return len(_zip_images(src))
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    with fitz.open(src) as d:
        return d.page_count


def describe_file(path, fallback_name=None):
    """Tek bir PDF/CBZ dosyasını bölüm sözlüğüne çevirir (hata olursa {'error': ...})."""
    p = Path(path)
    ext = p.suffix.lower()
    kind = "pdf" if ext == ".pdf" else "cbz"
    try:
        if kind == "cbz" and not zipfile.is_zipfile(p):
            return dict(error=f"{p.name}: RAR tabanlı .cbr desteklenmiyor (7-Zip ile .cbz/.zip'e çevirin).")
        n = page_count(str(p), kind)
        if n == 0:
            return dict(error=f"{p.name}: içinde sayfa (görsel) bulunamadı.")
        name = p.stem
        if fallback_name and extract_number(name)[0] is None:
            name = fallback_name
        num, conf = extract_number(name)
        return dict(src=str(p), kind=kind, name=name, auto_num=num, conf=conf, n_pages=n, sig=_sig(str(p), kind))
    except Exception as e:
        return dict(error=f"{p.name}: {e}")


def scan_root(root):
    """Ana klasörü gezer: görsel içeren her klasör, her PDF ve her CBZ/ZIP bir bölümdür."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort(key=natural_key)
        dp = Path(dirpath)
        if any(Path(f).suffix.lower() in IMG_EXT for f in filenames):
            try:
                n = page_count(str(dp), "dir")
                num, conf = extract_number(dp.name)
                out.append(dict(src=str(dp), kind="dir", name=dp.name or str(dp), auto_num=num, conf=conf,
                                n_pages=n, sig=_sig(str(dp), "dir")))
            except Exception as e:
                out.append(dict(error=f"{dp}: {e}"))
        books = sorted((f for f in filenames if Path(f).suffix.lower() == ".pdf" or Path(f).suffix.lower() in ARCH_EXT),
                       key=natural_key)
        for f in books:
            fb = dp.name if (len(books) == 1 and dp != Path(root)) else None   # dosya adında numara yoksa klasör adı
            out.append(describe_file(dp / f, fb))
    return out


def _open_pdf(path):
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    if _cache["path"] != path:
        if _cache["doc"] is not None:
            _cache["doc"].close()
        _cache["doc"], _cache["path"] = fitz.open(path), path
    return _cache["doc"]


def _open_zip(path):
    if _cache["zpath"] != path:
        if _cache["zip"] is not None:
            _cache["zip"].close()
        _cache["zip"], _cache["zpath"] = zipfile.ZipFile(path), path
    return _cache["zip"]


def _to_rgb(im):
    im.load()
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", im.size, "white")
        im = im.convert("RGBA")
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert("RGB")


def load_page(ch, idx, long_side=2200):
    """Bölümün idx. sayfasını RGB PIL görseli olarak döndürür."""
    if ch["kind"] == "dir":
        return _to_rgb(Image.open(_img_files(ch["src"])[idx]))
    if ch["kind"] == "cbz":
        with _lock:
            z = _open_zip(ch["src"])
            data = z.read(_zip_images(ch["src"])[idx])
        return _to_rgb(Image.open(io.BytesIO(data)))
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    with _lock:
        page = _open_pdf(ch["src"])[idx]
        r = page.rect
        zoom = long_side / max(r.width, r.height)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

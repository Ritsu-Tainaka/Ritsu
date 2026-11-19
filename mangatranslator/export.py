"""Bölüm PDF'i ve toplu ZIP çıktısı. Sayfalar tek tek okunur (düşük RAM)."""
import io, os, re, zipfile
from pathlib import Path
from PIL import Image
from . import source
from .chapters import effective, fmt_num, sort_chapters, label


def _safe(s):
    return re.sub(r'[<>:"/\\|?*]+', "", s).strip() or "manga"


def pdf_name(project, ch):
    base = f"{_safe(project.name)} - {label(ch)} {fmt_num(effective(ch)).zfill(3) if effective(ch) is not None else _safe(ch['name'])}"
    used = {c["out_pdf"] for c in project.chapters() if c["id"] != ch["id"] and c["out_pdf"]}
    name, i = f"{base}.pdf", 2
    while name in used:  # aynı numaralı bölümler birbirinin üstüne yazılmaz
        name, i = f"{base} ({i}).pdf", i + 1
    return name


def export_chapter(project, ch, cfg, only_translated=False):
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    wd = project.work_dir(ch["id"])
    doc = fitz.open()
    q = int(cfg.get("jpeg_quality", 90))
    for i in range(ch["n_pages"]):
        jp = wd / f"p{i:04d}.jpg"
        if only_translated and not jp.exists():
            continue
        if jp.exists():
            data = jp.read_bytes()
            w, h = Image.open(io.BytesIO(data)).size
        else:  # çevrilemeyen sayfa: orijinali koy
            im = source.load_page(ch, i, cfg.get("render_long_side", 2200))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=q)
            data, (w, h) = buf.getvalue(), im.size
        pw, ph = w * 72 / 150, h * 72 / 150
        page = doc.new_page(width=pw, height=ph)
        page.insert_image(fitz.Rect(0, 0, pw, ph), stream=data)
    if doc.page_count == 0:
        doc.close()
        raise RuntimeError("Bu bölümde henüz çevrilmiş sayfa yok.")
    name = pdf_name(project, ch)
    out = project.path / "out" / name
    tmp = out.with_suffix(".tmp")
    doc.save(str(tmp), garbage=3, deflate=True)
    doc.close()
    os.replace(tmp, out)
    project.update_chapter(ch["id"], out_pdf=name)
    return out


def export_cbz(project, ch, cfg, only_translated=False):
    """Çevrilmiş sayfaları .cbz (ZIP) olarak yazar: 001.jpg, 002.jpg, ..."""
    wd = project.work_dir(ch["id"])
    name = Path(pdf_name(project, ch)).with_suffix(".cbz").name
    out = project.path / "out" / name
    tmp = out.with_suffix(".tmp")
    q = int(cfg.get("jpeg_quality", 90))
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for i in range(ch["n_pages"]):
            jp = wd / f"p{i:04d}.jpg"
            if only_translated and not jp.exists():
                continue
            if jp.exists():
                z.write(jp, f"{i + 1:03d}.jpg")
            else:   # çevrilemeyen sayfa: orijinali koy
                buf = io.BytesIO()
                source.load_page(ch, i, cfg.get("render_long_side", 2200)).save(buf, "JPEG", quality=q)
                z.writestr(f"{i + 1:03d}.jpg", buf.getvalue())
    os.replace(tmp, out)
    return out


def export_all(project, ch, cfg, only_translated=False):
    """PDF (her zaman) + isteğe bağlı CBZ."""
    out = export_chapter(project, ch, cfg, only_translated)
    if cfg.get("export_cbz"):
        export_cbz(project, ch, cfg, only_translated)
    return out


def export_zip(project, zip_path):
    chs = [c for c in sort_chapters(project.chapters()) if c["out_pdf"] and (project.path / "out" / c["out_pdf"]).exists()]
    if not chs:
        raise RuntimeError("Dışa aktarılacak PDF yok.")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for c in chs:
            z.write(project.path / "out" / c["out_pdf"], c["out_pdf"])
            cb = Path(c["out_pdf"]).with_suffix(".cbz")
            if (project.path / "out" / cb).exists():
                z.write(project.path / "out" / cb, cb.name)
    return len(chs)

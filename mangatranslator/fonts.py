"""Yazı stili → Türkçe karakter destekli font eşleme. Hazır fontlar OFL lisanslıdır."""
from functools import lru_cache
from pathlib import Path
from PIL import ImageFont
from . import settings as S

# stil adı: (dosya, değişken-font ağırlığı)
BUILTIN = {
    "comic":       ("ComicRelief-Bold.ttf", None),      # standart kalın çizgi roman yazısı
    "comic_light": ("ComicRelief-Regular.ttf", None),   # ince/normal
    "shout":       ("Bangers-Regular.ttf", None),       # bağırma / vurgu
    "rounded":     ("Nunito.ttf", 800),                 # yumuşak, yuvarlak
    "serif":       ("Lora.ttf", 600),                   # anlatım kutuları
    "handwriting": ("PatrickHand-Regular.ttf", None),   # el yazısı
    "condensed":   ("Anton-Regular.ttf", None),         # dar, uzun başlık yazısı
}
STYLE_LABELS = {
    "comic": "Çizgi roman (kalın)", "comic_light": "Çizgi roman (ince)", "shout": "Bağırma",
    "rounded": "Yuvarlak", "serif": "Serif / anlatım", "handwriting": "El yazısı", "condensed": "Dar başlık",
}
TR_CHARS = "ğüşıöçĞÜŞİÖÇ"


def _find(fname):
    for d in (S.USER_FONT_DIR, S.BUILTIN_FONT_DIR):
        p = Path(d) / fname
        if p.exists():
            return p
    p = Path(fname)
    return p if p.exists() else None


@lru_cache(maxsize=64)
def has_turkish(path):
    try:
        from fontTools.ttLib import TTFont
        cmap = TTFont(str(path), lazy=True).getBestCmap()
        return all(ord(c) in cmap for c in TR_CHARS)
    except Exception:
        return True


def choices():
    """Ayarlarda seçilebilecek font dosyaları (hazır + kullanıcı)."""
    names = set()
    for d in (S.BUILTIN_FONT_DIR, S.USER_FONT_DIR):
        if Path(d).exists():
            names |= {p.name for p in Path(d).iterdir() if p.suffix.lower() in (".ttf", ".otf")}
    return sorted(names)


def resolve(style, font_map=None):
    """Stil adı → (font yolu, ağırlık). Seçilen font Türkçe harfleri içermiyorsa hazır fonta döner."""
    style = style if style in BUILTIN else "comic"
    want = (font_map or {}).get(style)
    if want:
        p = _find(want)
        if p and has_turkish(str(p)):
            return str(p), None
    fname, weight = BUILTIN[style]
    p = _find(fname) or _find(BUILTIN["comic"][0])
    return str(p), weight


@lru_cache(maxsize=256)
def load(path, size, weight=None):
    f = ImageFont.truetype(path, int(size))
    if weight:
        try:
            f.set_variation_by_axes([weight])
        except Exception:
            pass
    return f

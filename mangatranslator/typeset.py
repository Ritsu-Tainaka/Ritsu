"""Türkçe metni balona sığdırır ve orijinale en yakın font stiliyle çizer."""
import cv2
from PIL import Image, ImageDraw
from . import fonts
from .clean import to_px, grow_box


def tr_upper(s):
    return s.replace("i", "İ").replace("ı", "I").upper()


_VOW = set("aeıioöuüAEIİOÖUÜâîûÂÎÛ")
_APOS = {"'", "’"}


def syllables(word):
    """TDK kurallarına uygun Türkçe heceleme."""
    vs = [i for i, c in enumerate(word) if c in _VOW]
    if len(vs) < 2:
        return [word]
    cuts = []
    for a, b in zip(vs, vs[1:]):
        between = word[a + 1:b]
        special = [k for k, ch in enumerate(between) if ch in _APOS or ch == "-"]
        if special:
            cuts.append(a + 1 + special[-1] + 1)
            continue
        cons = len(between)
        cut = b if cons == 0 else b - 1
        cuts.append(cut)
    parts, prev = [], 0
    for c in cuts:
        parts.append(word[prev:c])
        prev = c
    parts.append(word[prev:])
    return [p for p in parts if p]


def _hyphenate(font, word, width, min_letters=6):
    """Satıra sığmayan kelimeyi TDK kurallarına göre hece sınırından böler."""
    if sum(c.isalpha() for c in word) < min_letters:
        return None
    sy = syllables(word)
    best = None
    for k in range(1, len(sy)):
        head, tail = "".join(sy[:k]), "".join(sy[k:])
        if sum(c.isalpha() for c in head) < 3 or sum(c.isalpha() for c in tail) < 3:
            continue
        sep = "" if head[-1] in _APOS or head.endswith("-") else "-"
        if font.getlength(head + sep) <= width:
            best = (head + sep, tail)
    return best


def _wrap(font, text, width, hyph=True):
    lines = []
    for para in text.split("\n"):
        cur = ""
        words = para.split()
        no_hyph = set()          # bir kelime en fazla bir kez bölünür (BAŞLA-DIĞI-MIZDAN olmasın)
        while words:
            word = words.pop(0)
            t = (cur + " " + word).strip()
            if font.getlength(t) <= width:
                cur = t
                continue
            if cur:
                lines.append(cur)
                cur = ""
                words.insert(0, word)
                continue
            if hyph and font.getlength(word) > width and id(word) not in no_hyph:
                hp = _hyphenate(font, word, width, min_letters=7)
                if hp:
                    lines.append(hp[0])
                    tail = hp[1]
                    words.insert(0, tail)
                    no_hyph.add(id(tail))
                    continue
            cur = word
        if cur:
            lines.append(cur)
    return lines


def _balance(font, text, width, n, hyph=True):
    """Aynı satır sayısını koruyarak en dar genişliği bulur (satırlar dengeli, elips balona uygun)."""
    base = _wrap(font, text, width, hyph)
    nh = sum(l.endswith(("-", "'", "’")) for l in base)
    lo, hi = 1, int(width)
    while lo < hi:
        mid = (lo + hi) // 2
        w = _wrap(font, text, mid, hyph)
        if (len(w) <= n and sum(l.endswith(("-", "'", "’")) for l in w) <= nh
                and max(font.getlength(l) for l in w) <= width):
            hi = mid
        else:
            lo = mid + 1
    out = _wrap(font, text, lo, hyph)
    return out if len(out) <= n and max(font.getlength(l) for l in out) <= width else base


def _break_long(font, lines, width):
    """Son çare: satır hala genişlikten büyükse önce TDK hece sınırından böler."""
    out = []
    for ln in lines:
        while font.getlength(ln) > width and len(ln) > 3:
            hp = _hyphenate(font, ln, width, min_letters=4)
            if hp:
                out.append(hp[0])
                ln = hp[1]
                continue
            i = len(ln) - 1
            while i > 1 and font.getlength(ln[:i] + "-") > width:
                i -= 1
            out.append(ln[:i] + "-")
            ln = ln[i:]
        out.append(ln)
    return out


def _best_size(text, path, weight, w, h, max_size, min_size, spacing, hyph):
    lo, hi, best = int(min_size), max(int(min_size), int(max_size)), None
    while lo <= hi:
        mid = (lo + hi) // 2
        f = fonts.load(path, mid, weight)
        lines = _wrap(f, text, w, hyph)
        fits = max(f.getlength(l) for l in lines) <= w and len(lines) * mid * spacing <= h
        if fits:
            best, lo = (mid, lines), mid + 1
        else:
            hi = mid - 1
    return best


def fit(text, path, weight, w, h, max_size, min_size=9, spacing=1.08):
    """Önce kelime bölmeden en büyük boyut; heceden bölmek yazıyı çok belirgin büyütüyorsa (%45+) onu seçer."""
    plain = _best_size(text, path, weight, w, h, max_size, min_size, spacing, False)
    hyph = _best_size(text, path, weight, w, h, max_size, min_size, spacing, True)
    if hyph and (not plain or hyph[0] >= plain[0] * 1.6):
        size, lines = hyph
        f = fonts.load(path, size, weight)
        return f, size, _balance(f, text, w, len(lines)), False
    if plain:
        size, lines = plain
        f = fonts.load(path, size, weight)
        return f, size, _balance(f, text, w, len(lines), hyph=False), False
    size = int(min_size)
    f = fonts.load(path, size, weight)
    return f, size, _break_long(f, _wrap(f, text, w), w), True


def _vertical(text, path, weight, w, h, max_size):
    """Kısa, tek kelimelik metni dikey (harf harf) yerleştirmek için en büyük boyut."""
    chars = [c for c in text if not c.isspace()]
    if h < 1.6 * w or len(text.split()) > 1 or not 2 <= len(chars) <= 10:
        return None
    best = None
    lo, hi = 6, int(max_size)
    while lo <= hi:
        mid = (lo + hi) // 2
        f = fonts.load(path, mid, weight)
        if max(f.getlength(c) for c in chars) <= w and len(chars) * mid * 0.98 <= h:
            best, lo = (mid, f), mid + 1
        else:
            hi = mid - 1
    return best


# ---------------------------------------------------------------- balon şekline göre dizgi
def bubble_zone(arr, bmask):
    """Balonun yazı yazılabilir iç bölgesi: kenardan biraz içeri, balonda kalan çizim/efekt
    mürekkebinden uzak. Silme sonrası kalan soluk gölgeler ve tek tük noktalar engel sayılmaz."""
    import numpy as np
    x, y, w, h = cv2.boundingRect(bmask)
    if w < 12 or h < 12:
        return None
    k = max(2, int(min(w, h) * 0.05))
    inner = cv2.erode(bmask, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    g = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    ink = ((g < 110) & (inner > 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    big = np.zeros(n, bool)
    big[1:] = st[1:, 4] >= max(30, (0.02 * min(w, h)) ** 2)
    ink = big[lab].astype(np.uint8)
    if ink.any():
        kk = max(3, k) | 1
        ink = cv2.dilate(ink, np.ones((kk, kk), np.uint8))
    return ((inner > 0) & (ink == 0))


def _near_text(zone, box, meas):
    """Balon sınırı sızıntılı bulunduysa (açık renkli zemin, alttan açık patlama balonu) bölge
    gereğinden büyür ve yazı balonun dışına kayar. Bölge, orijinal yazının çevresiyle ve ona bağlı
    parçayla sınırlanır."""
    import numpy as np
    H, W = zone.shape
    x0, y0, x1, y1 = box
    th = float(meas.get("text_h") or 0) or 0.012 * H
    mx = max(0.6 * (x1 - x0), 3 * th)
    my = max(0.6 * (y1 - y0), 3 * th)
    win = np.zeros_like(zone)
    a0, b0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
    a1, b1 = int(min(W, x1 + mx)), int(min(H, y1 + my))
    win[b0:b1, a0:a1] = True
    z = (zone & win).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(z, connectivity=4)
    if n <= 1:
        return None
    ix0, iy0, ix1, iy1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
    sub = lab[iy0:iy1, ix0:ix1]
    ids, cnt = np.unique(sub[sub > 0], return_counts=True)
    k = int(ids[cnt.argmax()]) if len(ids) else 1 + int(np.argmax(st[1:, 4]))
    return lab == k


def _band(zone, y0, y1, pad):
    """[y0, y1) satır bandında baştan sona boş olan en uzun yatay aralık."""
    import numpy as np
    H = zone.shape[0]
    y0, y1 = max(0, int(y0)), min(H, int(np.ceil(y1)))
    if y1 <= y0:
        return None
    col = zone[y0:y1].all(axis=0)
    if not col.any():
        return None
    d = np.diff(np.concatenate(([0], col.view(np.int8), [0])))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    i = int(np.argmax(en - st))
    a, b = st[i] + pad, en[i] - pad
    return (a, b) if b - a > 4 else None


def _wrap_var(font, words, widths, hyph):
    """Her satırın kendi genişliği olan açgözlü satır kırma; sığmazsa None."""
    words = list(words)
    out, i = [], 0
    no_hyph = set()
    while words:
        if i >= len(widths):
            return None
        w = widths[i]
        cur = ""
        while words:
            t = (cur + " " + words[0]).strip()
            if font.getlength(t) <= w:
                cur = t
                words.pop(0)
                continue
            if not cur:
                word = words[0]
                hp = _hyphenate(font, word, w, min_letters=7) if hyph and id(word) not in no_hyph else None
                if not hp:
                    return None
                cur = hp[0]
                words[0] = hp[1]
                no_hyph.add(id(words[0]))
            break
        out.append(cur)
        i += 1
    return out


def _try_layout(text, font, size, zone, cy, hyph, spacing=1.08):
    import numpy as np
    lh = size * spacing
    rows = np.flatnonzero(zone.any(axis=1))
    if len(rows) == 0:
        return None
    top, bot = rows[0], rows[-1] + 1
    words = text.split()
    total = font.getlength(text)
    pad = max(1, int(size * 0.08))
    maxw = int(zone.any(axis=0).sum())
    n0 = max(1, int(total // max(1, maxw)))
    for n in range(n0, int((bot - top) // lh) + 1):
        hgt = n * lh
        c0 = min(max(cy, top + hgt / 2), bot - hgt / 2)
        step = max(2.0, lh / 2)
        offs = [0.0]
        for j in range(1, 12):
            offs += [j * step, -j * step]
        for o in offs:
            c = c0 + o
            if c - hgt / 2 < top or c + hgt / 2 > bot:
                continue
            t0 = c - hgt / 2
            bands = [_band(zone, t0 + i * lh, t0 + i * lh + size * 1.0, pad) for i in range(n)]
            if any(b is None for b in bands):
                continue
            widths = [b[1] - b[0] for b in bands]
            lines = _wrap_var(font, words, widths, hyph)
            if lines is None:
                continue
            # aynı satır sayısıyla daha dengeli (dar) yerleşim
            lo, hi = 0.5, 1.0
            for _ in range(7):
                mid = (lo + hi) / 2
                l2 = _wrap_var(font, words, [w * mid for w in widths], hyph)
                if l2 is not None and len(l2) == len(lines) and \
                        sum(x.endswith("-") for x in l2) <= sum(x.endswith("-") for x in lines):
                    hi = mid
                else:
                    lo = mid
            l2 = _wrap_var(font, words, [w * hi for w in widths], hyph)
            if l2 is not None and len(l2) == len(lines):
                lines = l2
            placed = None
            n_used = len(lines)
            if n_used < n:                         # boş kalan satırlar olursa bloğu ortala
                t1 = c - n_used * lh / 2
                bs = [_band(zone, t1 + i * lh, t1 + i * lh + size, pad) for i in range(n_used)]
                if all(b is not None and font.getlength(ln) <= b[1] - b[0] for b, ln in zip(bs, lines)):
                    placed = [(ln, (b[0] + b[1]) / 2, t1 + i * lh) for i, (b, ln) in enumerate(zip(bs, lines))]
            if placed is None:
                placed = [(ln, (b[0] + b[1]) / 2, t0 + i * lh) for i, (b, ln) in enumerate(zip(bands, lines))]
            return placed
    return None


def fit_zone(text, path, weight, zone, cy, max_size, min_size):
    """Balonun gerçek şekline göre en büyük yazı boyu ve satır yerleşimi."""
    def best(hyph):
        lo, hi, res = int(min_size), max(int(min_size), int(max_size)), None
        while lo <= hi:
            mid = (lo + hi) // 2
            f = fonts.load(path, mid, weight)
            pl = _try_layout(text, f, mid, zone, cy, hyph)
            if pl:
                res, lo = (f, mid, pl), mid + 1
            else:
                hi = mid - 1
        return res
    plain = best(False)
    if plain and plain[1] >= 0.85 * max_size:
        return plain
    hy = best(True)
    if hy and (not plain or hy[1] >= plain[1] * 1.25):
        return hy
    return plain


def pick_style(r):
    """Gemini'nin söylediği stili, ölçülen kalınlıkla düzeltir."""
    st = r.get("font", "comic")
    wt = (r.get("meas") or {}).get("weight", 0)
    if wt:
        if st == "comic" and wt < 0.12:
            st = "comic_light"
        elif st == "comic_light" and wt > 0.24:
            st = "comic"
    if r.get("kind") == "narration" and r.get("font") not in fonts.BUILTIN:
        st = "serif"
    return st


def _ink_colors(r, meas, size):
    ink = tuple(meas.get("ink", (0, 0, 0))) if meas else (0, 0, 0)
    if r.get("color") == "white" and not meas:
        ink = (255, 255, 255)
    if max(ink) - min(ink) < 35:
        _lum = 0.3 * ink[0] + 0.59 * ink[1] + 0.11 * ink[2]
        if _lum < 115:
            ink = (0, 0, 0)
        elif _lum > 165:
            ink = (255, 255, 255)
    lum = 0.3 * ink[0] + 0.59 * ink[1] + 0.11 * ink[2]
    if meas and not meas.get("flat", True):
        bg = meas.get("bg", (255, 255, 255))
        bg_lum = 0.3 * bg[0] + 0.59 * bg[1] + 0.11 * bg[2]
        if abs(lum - bg_lum) < 75:
            ink = (255, 255, 255) if bg_lum < 128 else (0, 0, 0)
            lum = 0.3 * ink[0] + 0.59 * ink[1] + 0.11 * ink[2]
    stroke_fill = (0, 0, 0) if lum > 128 else (255, 255, 255)
    stroke = 0 if meas.get("flat", True) else max(2, round(size / 9))
    return tuple(ink), stroke_fill, stroke


def _paint(img, r, meas, f, size, placed):
    """placed: [(satır, merkez_x, üst_y)] sayfa koordinatında."""
    ink, stroke_fill, stroke = _ink_colors(r, meas, size)
    W, H = img.size
    m = int(size * 0.6) + stroke + 2
    wid = max(f.getlength(ln) for ln, _, _ in placed)
    lx0 = max(0, int(min(cx for _, cx, _ in placed) - wid / 2 - m))
    ly0 = max(0, int(min(y for _, _, y in placed) - m))
    lx1 = min(W, int(max(cx for _, cx, _ in placed) + wid / 2 + m))
    ly1 = min(H, int(max(y for _, _, y in placed) + size * 1.3 + m))
    if lx1 - lx0 < 2 or ly1 - ly0 < 2:
        return
    layer = Image.new("RGBA", (lx1 - lx0, ly1 - ly0), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for ln, cx, y in placed:
        d.text((cx - lx0, y - ly0), ln, font=f, fill=ink + (255,), anchor="ma",
               stroke_width=stroke, stroke_fill=stroke_fill)
    if r.get("slant"):
        sh = 0.22
        layer = layer.transform(layer.size, Image.AFFINE, (1, sh, -sh * layer.height / 2, 0, 1, 0), Image.BICUBIC)
    img.paste(layer, (lx0, ly0), layer)


def _size_limits(r, meas, H, aw, ah):
    th = meas.get("text_h") or 0
    hint = th / 0.72 if th >= 7 else ah * 0.5
    base = H * 0.012                                  # okunabilir yazı boyu (2200 px sayfada ~26 px)
    max_size = max(10, min(ah, max(hint * 1.25, base))) * float(r.get("size_mul", 1.0))
    min_readable = max(7, min(int(base * 0.6), int(aw / 4.5)))
    return max_size, min_readable


def _rect_layout(img, r, meas, box, text, path, weight):
    """Dikdörtgen alana dizgi (balonu bilinmeyen metinler); (font, boy, yerleşim, taştı mı) ya da None."""
    import numpy as np
    W, H = img.size
    x0, y0, x1, y1 = box
    if meas.get("flat"):
        x0, y0, x1, y1 = grow_box(np.asarray(img), box, meas.get("bg", [255, 255, 255]))
    else:
        bw, bh = x1 - x0, y1 - y0
        x0, y0, x1, y1 = max(0, x0 - bw // 12), max(0, y0 - bh // 12), min(W, x1 + bw // 12), min(H, y1 + bh // 12)
    pad_ratio = 0.02 if (x1 - x0) < 90 else 0.06
    mw, mh = (x1 - x0) * pad_ratio, (y1 - y0) * pad_ratio
    aw, ah = (x1 - x0) - 2 * mw, (y1 - y0) - 2 * mh
    if aw < 8 or ah < 8:
        return None
    max_size, min_readable = _size_limits(r, meas, H, aw, ah)
    f, size, lines, overflow = fit(text, path, weight, aw, ah, max_size, min_size=min_readable)
    if not meas.get("flat", True):
        _s = max(2, round(size / 9))
        if aw - 2 * _s >= 8 and ah - 2 * _s >= 8:
            f, size, lines, overflow = fit(text, path, weight, aw - 2 * _s, ah - 2 * _s, max_size, min_size=min_readable)
    lh = size * 1.08
    vert = _vertical(text, path, weight, aw, ah, max_size)
    if vert and vert[0] >= size * 1.3:
        size, f = vert
        lines = [c for c in text if not c.isspace()]
        lh, overflow = size * 0.98, False
    cx = (x0 + x1) / 2
    y = (y0 + y1) / 2 - len(lines) * lh / 2
    return f, size, [(ln, cx, y + i * lh) for i, ln in enumerate(lines)], overflow


def _bubble_solidity(bmask):
    cnts, _ = cv2.findContours(bmask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 1.0
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    return float(cv2.contourArea(c) / hull) if hull else 1.0


def _oval_zone(zone, box):
    """Dikenli/çizgili ya da kenarı açık balonlar için görünmez oval sınır: orijinal yazının
    çevresinden başlayıp boş alanın içinde kalabildiği kadar büyüyen elips. Dikenler, çizgiler ve
    balonun açık kenarı bu ovalin dışında kalır."""
    import numpy as np
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax0, ay0 = max(6.0, (x1 - x0) / 2 * 1.1), max(6.0, (y1 - y0) / 2 * 1.1)
    H, W = zone.shape
    best = None
    for k in np.arange(0.5, 2.01, 0.05):
        m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, (int(cx), int(cy)), (int(ax0 * k), int(ay0 * k)), 0, 0, 360, 1, -1)
        a = int(m.sum())
        if not a:
            continue
        inside = float((zone & (m > 0)).sum()) / a
        if inside < 0.97:
            break
        best = m > 0
    return None if best is None else (zone & best)


def _zone_layout(img, r, meas, box, text, path, weight, bmask):
    import numpy as np
    W, H = img.size
    zone = bubble_zone(np.asarray(img), bmask)
    if zone is not None and zone.any():
        zone = _near_text(zone, box, meas)
    if zone is None or not zone.any():
        return None
    ys, xs = np.nonzero(zone)
    zw, zh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    max_size, min_readable = _size_limits(r, meas, H, zw * 0.9, zh * 0.9)
    bcy = (box[1] + box[3]) / 2
    cy = 0.4 * float(ys.mean()) + 0.6 * min(max(bcy, ys.min()), ys.max())
    res = fit_zone(text, path, weight, zone, cy, max_size, min_readable)
    if True:                                          # dikenli/çizgili/açık balonlar için oval sınır dene
        oz = _oval_zone(zone, box)
        if oz is not None and oz.any():
            oys = np.nonzero(oz)[0]
            ores = fit_zone(text, path, weight, oz, float(oys.mean()), max_size, min_readable)
            if ores and (not res or ores[1] >= 0.95 * res[1]):
                return ores
    return res


def draw_region(img, r, font_map=None, bmask=None, guess=False):
    """bmask: metnin balonu (yazı balonun gerçek şekline göre dizilir).
    guess=True: balon sonradan tahmin edildi; balon şekli ancak yazıyı küçültmüyorsa kullanılır."""
    W, H = img.size
    meas = r.get("meas") or {}
    box = to_px(r["box"], W, H)
    text = " ".join(r["tr"].split())
    if r.get("caps", True):
        text = tr_upper(text)
    path, weight = fonts.resolve(pick_style(r), font_map)
    z = _zone_layout(img, r, meas, box, text, path, weight, bmask) if bmask is not None else None
    if z is not None and not guess:
        _paint(img, r, meas, *z)
        return False
    rect = _rect_layout(img, r, meas, box, text, path, weight)
    if z is not None and (rect is None or z[1] >= 0.9 * rect[1]):
        _paint(img, r, meas, *z)
        return False
    if rect is None:
        return False
    f, size, placed, overflow = rect
    _paint(img, r, meas, f, size, placed)
    return overflow

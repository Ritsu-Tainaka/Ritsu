"""Yerel CPU metin algılama: RapidOCR (önerilen) veya Tesseract."""
import os
import re
import shutil
import statistics
import numpy as np
import cv2
from PIL import Image

WIN_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

SKIP_COLOR_PAGES = True
COLOR_SAT_RATIO = 0.18
ONLY_NARRATION_OUTSIDE_BUBBLES = True
LIGHT_T = 200
DARK_T = 65
KEEP_PLAIN_BACKGROUND_TEXT = True
KEEP_UNCLOSED_BUBBLES = True
SFX_H = 2.2
BUBBLE_TOL = 0.7
KEEP_STYLIZED_DIALOGUE = True
TILT_SKIP_DEG = 10.0      # bu açıdan eğik satırlar tabela/efekt sayılır
STYLIZED_MIN_H = 1.25
STYLIZED_MAX_H = 8.0
KEEP_SCREENTONE_DIALOGUE = True
DEBUG_FILTER = False
BOX_FILL = 0.86
IGNORE_TEXT = [r"#\w+", r"@\w+", r"https?://", r"\bwww\."]

UNCLOSED_MAX_H = 2.15
UNCLOSED_MIN_RING = 0.52
SIGN_MAX_WORDS = 8

_SIGN_RE = re.compile(
    r"\b(welcome\s+to|class\s*\d|room\s*\d|\d+\s*(st|nd|rd|th)\s*floor|no\s+entry|staff\s+only)\b"
    r"|cosp[l1i]ay\s+caf[eé]"
    r"|^\W*\d+\s*[-–]\s*\d+\W*$",
    re.I,
)

_TITLE_RE = re.compile(r"^\W*(chapter|chap|ch|episode|ep|vol(?:ume)?)\s*\.?\s*\d", re.I)


last_dropped = []          # son detect() çağrısında atlanan balon dışı metinler (tanı için json'a yazılır)


def _drop(text, why):
    if text and any(c.isalpha() for c in text) and len(last_dropped) < 60:
        last_dropped.append(dict(src=text[:120], why=why))


def _real_words(text):
    return [w for w in re.findall(r"[A-Za-z][A-Za-z']*", text) if sum(c.isalpha() for c in w) >= 2]


def _is_sfx_like(text):
    words = _real_words(text)
    if not words:
        return True
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    all_caps = (sum(c.isupper() for c in letters) / len(letters)) >= 0.85
    if all_caps and len(words) <= 2:
        return True
    return False


def _looks_like_sign(text):
    return len(text.split()) <= SIGN_MAX_WORDS and bool(_SIGN_RE.search(text))


def _is_rectangular_bubble(poly):
    pts = np.asarray(poly, dtype=np.float32)
    if len(pts) < 4:
        return False
    _, _, w, h = cv2.boundingRect(pts)
    if w <= 0 or h <= 0:
        return False
    area = cv2.contourArea(pts)
    return (area / float(w * h)) >= 0.88


def _is_color_page(rgb):
    s = 600.0 / max(rgb.shape[:2])
    small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    return float(((sat > 75) & (val > 40)).mean()) > COLOR_SAT_RATIO


def _components(gray):
    _, l_lab, l_st, _ = cv2.connectedComponentsWithStats((gray > LIGHT_T).astype(np.uint8), connectivity=4)
    _, d_lab, d_st, _ = cv2.connectedComponentsWithStats((gray < DARK_T).astype(np.uint8), connectivity=4)
    return (l_lab, l_st), (d_lab, d_st)


def _is_narration_box(gray, comps, box):
    (l_lab, l_st), (d_lab, d_st) = comps
    H, W = gray.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 <= x0 or y1 <= y0:
        return False

    sub = l_lab[y0:y1, x0:x1]
    ids, cnt = np.unique(sub[sub > 0], return_counts=True)
    if len(ids) > 0 and cnt.max() >= 0.78 * cnt.sum():
        k = ids[cnt.argmax()]
        cx, cy, cw, ch, _ = l_st[k]
        m = 4
        if (m < cx and m < cy and cx + cw < W - m and cy + ch < H - m
                and cw * ch <= 0.45 * W * H
                and cx <= x0 + m and cy <= y0 + m and cx + cw >= x1 - m and cy + ch >= y1 - m):
            comp = (l_lab[cy:cy + ch, cx:cx + cw] == k).astype(np.uint8)
            cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cs and cv2.contourArea(max(cs, key=cv2.contourArea)) >= BOX_FILL * cw * ch:
                t = 4
                strips = [
                    gray[max(cy - t, 0):cy, cx:cx + cw].min(axis=0, initial=255),
                    gray[cy + ch:min(cy + ch + t, H), cx:cx + cw].min(axis=0, initial=255),
                    gray[cy:cy + ch, max(cx - t, 0):cx].min(axis=1, initial=255),
                    gray[cy:cy + ch, cx + cw:min(cx + cw + t, W)].min(axis=1, initial=255),
                ]
                if sum(float((s < 120).mean()) >= 0.55 for s in strips if s.size) >= 3:
                    return True

    dsub = d_lab[y0:y1, x0:x1]
    dids, dcnt = np.unique(dsub[dsub > 0], return_counts=True)
    if len(dids) > 0 and dcnt.max() >= 0.72 * dcnt.sum():
        k = dids[dcnt.argmax()]
        cx, cy, cw, ch, _ = d_st[k]
        m = 4
        if (m < cx and m < cy and cx + cw < W - m and cy + ch < H - m
                and 400 < cw * ch <= 0.40 * W * H
                and cx <= x0 + m and cy <= y0 + m and cx + cw >= x1 - m and cy + ch >= y1 - m):
            comp = (d_lab[cy:cy + ch, cx:cx + cw] == k).astype(np.uint8)
            cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cs and cv2.contourArea(max(cs, key=cv2.contourArea)) >= BOX_FILL * cw * ch:
                return True

    return False


def _is_unclosed_bubble(gray, comps, box, text, num_lines, line_h, med_h):
    if _looks_like_sign(text) or _TITLE_RE.search(text) or _is_sfx_like(text):
        return False
    if len(_real_words(text)) < 3:
        return False
    if med_h > 0 and line_h > UNCLOSED_MAX_H * med_h:
        return False
    (l_lab, l_st), _ = comps
    H, W = gray.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 <= x0 or y1 <= y0:
        return False
    sub = l_lab[y0:y1, x0:x1]
    ids, cnt = np.unique(sub[sub > 0], return_counts=True)
    if len(ids) == 0 or cnt.max() < 0.80 * cnt.sum():
        return False
    k = ids[cnt.argmax()]
    cx, cy, cw, ch, area = l_st[k]
    edge_touches = int(cx <= 3) + int(cy <= 3) + int(cx + cw >= W - 3) + int(cy + ch >= H - 3)
    if edge_touches >= 2:
        return False
    if area > 0.25 * W * H or area < 1.70 * (x1 - x0) * (y1 - y0):
        return False
    if float(area) / max(1, cw * ch) < 0.45:
        return False
    p = int(max(10, 0.50 * line_h))
    a0, b0, a1, b1 = max(x0 - p, 0), max(y0 - p, 0), min(x1 + p, W), min(y1 + p, H)
    ring = np.ones((b1 - b0, a1 - a0), bool)
    ring[y0 - b0:y1 - b0, x0 - a0:x1 - a0] = False
    ring_px = gray[b0:b1, a0:a1][ring]
    if ring_px.size == 0:
        return False
    return float((ring_px > LIGHT_T).mean()) >= UNCLOSED_MIN_RING


def _is_screentone_dialogue(gray, box, text, num_lines, line_h, med_h):
    if _looks_like_sign(text) or _TITLE_RE.search(text) or _is_sfx_like(text):
        return False
    if num_lines < 2 or len(_real_words(text)) < 4:
        return False
    if med_h > 0 and not (0.70 * med_h <= line_h <= 1.35 * med_h):
        return False
    H, W = gray.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 <= x0 or y1 <= y0:
        return False
    p = int(max(10, 0.5 * line_h))
    a0, b0, a1, b1 = max(x0 - p, 0), max(y0 - p, 0), min(x1 + p, W), min(y1 + p, H)
    ring = np.ones((b1 - b0, a1 - a0), bool)
    ring[y0 - b0:y1 - b0, x0 - a0:x1 - a0] = False
    ring_px = gray[b0:b1, a0:a1][ring]
    if ring_px.size == 0:
        return False
    return float((ring_px > 165).mean()) >= 0.78


def _is_stylized_dialogue(text, line_h, med_h):
    if _looks_like_sign(text) or _TITLE_RE.search(text) or _is_sfx_like(text):
        return False
    if med_h <= 0 or not (STYLIZED_MIN_H * med_h <= line_h <= STYLIZED_MAX_H * med_h):
        return False
    words = {w.lower() for w in _real_words(text)}
    if len(words) < 3:
        return False
    return bool(re.search(r"[?!.,…]|\w'\w", text))


def _is_screen_text(text, confs):
    """Telefon ekranı, mesaj, not gibi balon dışındaki küçük harfli cümleler ("Meet you at the gate.",
    "exhibit at Hakata Station"). Ses efektleri ve vurgulu iç sesler hep büyük harf yazılır; karışık
    harfli, en az iki kelimelik, net okunan satırlar okunması gereken yazıdır."""
    if _looks_like_sign(text) or _TITLE_RE.search(text):
        return False
    words = _real_words(text)
    letters = [c for c in text if c.isalpha()]
    if len(words) < 2 or len(letters) < 8:
        return False
    if sum(c.islower() for c in letters) < 0.5 * len(letters):
        return False
    return bool(confs) and min(confs) >= 85


def _is_short_dialogue(text):
    """Balon dışındaki kısa replik/iç ses ("HE'S", "Whee!", "I remember him..."): kesme işaretli
    kısaltma ya da konuşma noktalaması taşır. Ses efekti olup olmadığına Gemini karar verir
    (bölge 'free' işaretlenir; efektse boş döner ve sayfaya dokunulmaz)."""
    if _looks_like_sign(text) or _TITLE_RE.search(text):
        return False
    words = _real_words(text)
    if not words:
        return False
    if re.search(r"[A-Za-z]'[A-Za-z]", text):
        return True
    if not re.search(r"(\.\.|…|[?!,])\W*$|^\W*(\.\.|…)", text.strip()):
        return False
    if len(words) >= 2:
        return True
    w = words[0]
    # tek kelimelik ünlem: büyük harfli tek kelimeler genelde ses efektidir (BAM!, THUD!)
    return not w.isupper() and len(w) >= 3


def _is_bogus_bubble(poly):
    """Balon bulucu bazen yüz/saç gibi beyaz çizim alanlarını balon sanar (çok girintili, yüzlerce
    köşeli şekil). Gerçek balonlar (sivri 'şok' balonları dahil) daha düzgündür."""
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    n = len(pts)
    if n < 100:
        return False
    hull = cv2.contourArea(cv2.convexHull(pts))
    sol = cv2.contourArea(pts) / hull if hull > 0 else 0
    return n >= 300 or sol < 0.80


def _attach_leftovers(regs, leftovers, W, H, med_h=0):
    """Tek başına atlanan parçaları ("EVEN", "I") hemen bitişiğindeki balon dışı yazıya ekler:
    "EVEN" + "THOUGH IT'S AUTUMN..." tek cümle olarak çevrilir ve silinir."""
    for (x0, y0, x1, y1, text) in sorted(leftovers, key=lambda t: t[1]):
        lh = y1 - y0
        if med_h and lh > 1.8 * med_h:                 # büyük başlık/efekt ("Contents"): cümleye ait değil
            continue
        best, bgap = None, 1e9
        for r in regs:
            if r.get("bubble") is not None:
                continue
            ry0, rx0, ry1, rx1 = r["box"][0] * H / 1000, r["box"][1] * W / 1000, r["box"][2] * H / 1000, r["box"][3] * W / 1000
            ov = min(x1, rx1) - max(x0, rx0)
            if ov < 0.3 * min(x1 - x0, rx1 - rx0):
                continue
            gap = max(ry0 - y1, y0 - ry1, 0)
            if gap <= 1.2 * max(lh, 1) and gap < bgap:
                best, bgap = r, gap
        if best is None:
            continue
        before = y1 <= best["box"][0] * H / 1000 + 1
        best["src"] = _join([text, best["src"]] if before else [best["src"], text])
        best["box"] = [min(best["box"][0], int(y0 / H * 1000)), min(best["box"][1], int(x0 / W * 1000)),
                       max(best["box"][2], int(y1 / H * 1000) + 1), max(best["box"][3], int(x1 / W * 1000) + 1)]
        last_dropped[:] = [d for d in last_dropped if d["src"] != text[:120]]


def _is_plain_background_text(gray, comps, box, text):
    if _TITLE_RE.search(text) or _looks_like_sign(text) or _is_sfx_like(text):
        return False
    if len(_real_words(text)) < 3:
        return False
    (l_lab, l_st), _ = comps
    H, W = gray.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 <= x0 or y1 <= y0:
        return False
    sub = l_lab[y0:y1, x0:x1]
    ids, cnt = np.unique(sub[sub > 0], return_counts=True)
    if len(ids) == 0 or cnt.max() < 0.8 * cnt.sum():
        return False
    k = ids[cnt.argmax()]
    cx, cy, cw, ch, area = l_st[k]
    m = 3
    page_like = cx <= m or cy <= m or cx + cw >= W - m or cy + ch >= H - m or area > 0.25 * W * H
    if not page_like:
        return False
    p = int(max(12, 0.8 * (y1 - y0)))
    a0, b0, a1, b1 = max(x0 - p, 0), max(y0 - p, 0), min(x1 + p, W), min(y1 + p, H)
    ring = np.ones((b1 - b0, a1 - a0), bool)
    ring[y0 - b0:y1 - b0, x0 - a0:x1 - a0] = False
    return float((gray[b0:b1, a0:a1][ring] > LIGHT_T).mean()) >= 0.96


class OCRError(Exception):
    pass


def _setup(path=""):
    try:
        import pytesseract
    except ImportError:
        raise OCRError("pytesseract kurulu değil (pip install pytesseract).")
    cands = [path, shutil.which("tesseract")] + WIN_PATHS
    for c in cands:
        if c and os.path.exists(c):
            pytesseract.pytesseract.tesseract_cmd = c
            return pytesseract
    raise OCRError("Tesseract bulunamadı.")


def _run(pt, g, scale, conf_min=45):
    d = pt.image_to_data(g, lang="eng", config="--oem 1 --psm 11", output_type=pt.Output.DICT)
    out = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        try:
            conf = float(d["conf"][i])
        except Exception:
            conf = -1
        if not t or conf < conf_min or not any(c.isalnum() for c in t):
            continue
        x, y, w, h = (d["left"][i] / scale, d["top"][i] / scale, d["width"][i] / scale, d["height"][i] / scale)
        if h < 6:
            continue
        out.append(dict(t=t, x0=x, y0=y, x1=x + w, y1=y + h, conf=conf))
    return out


def _iou(a, b):
    ix = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
    iy = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
    if ix <= 0 or iy <= 0:
        return 0
    inter = ix * iy
    return inter / ((a["x1"] - a["x0"]) * (a["y1"] - a["y0"]) + (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]) - inter)


def _words(img, path):
    pt = _setup(path)
    g = img.convert("L")
    scale = 1.0
    if max(g.size) < 2600:
        scale = 2.0 if max(g.size) < 1800 else 1.5
        g = g.resize((int(g.width * scale), int(g.height * scale)), Image.LANCZOS)
    try:
        words = _run(pt, g, scale)
        inv = _run(pt, Image.fromarray(255 - np.asarray(g)), scale, conf_min=60)
    except Exception as e:
        raise OCRError(f"Tesseract çalışmadı: {e}")
    for w in inv:
        clash = [o for o in words if _iou(w, o) > 0.2]
        if not clash:
            words.append(w)
        elif all(w["conf"] > o["conf"] + 15 for o in clash):
            words = [o for o in words if o not in clash] + [w]
    return words


def _lines(words):
    words = sorted(words, key=lambda w: (w["y0"] + w["y1"]) / 2)
    lines = []
    for w in words:
        h = w["y1"] - w["y0"]
        cy = (w["y0"] + w["y1"]) / 2
        for ln in lines:
            lh = ln["y1"] - ln["y0"]
            same_row = abs(cy - (ln["y0"] + ln["y1"]) / 2) < 0.5 * max(h, lh)
            gap = max(w["x0"], ln["x0"]) - min(w["x1"], ln["x1"])
            if same_row and -0.3 * h < gap < 1.6 * h:
                ln["ws"].append(w)
                ln["x0"], ln["y0"] = min(ln["x0"], w["x0"]), min(ln["y0"], w["y0"])
                ln["x1"], ln["y1"] = max(ln["x1"], w["x1"]), max(ln["y1"], w["y1"])
                break
        else:
            lines.append(dict(ws=[w], x0=w["x0"], y0=w["y0"], x1=w["x1"], y1=w["y1"]))
    changed = True
    while changed:
        changed = False
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i], lines[j]
                h = max(a["y1"] - a["y0"], b["y1"] - b["y0"])
                row = abs((a["y0"] + a["y1"]) / 2 - (b["y0"] + b["y1"]) / 2) < 0.5 * h
                gap = max(a["x0"], b["x0"]) - min(a["x1"], b["x1"])
                if row and -0.3 * h < gap < 1.6 * h:
                    a["ws"] += b["ws"]
                    a["x0"], a["y0"] = min(a["x0"], b["x0"]), min(a["y0"], b["y0"])
                    a["x1"], a["y1"] = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
                    del lines[j]
                    changed = True
                    break
            if changed:
                break
    for ln in lines:
        ln["text"] = " ".join(w["t"] for w in sorted(ln["ws"], key=lambda w: w["x0"]))
    return lines


def _group(lines):
    n = len(lines)
    par = list(range(n))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            a, b = lines[i], lines[j]
            lh = min(a["y1"] - a["y0"], b["y1"] - b["y0"])
            gap = max(a["y0"], b["y0"]) - min(a["y1"], b["y1"])
            ox = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
            wmin = min(a["x1"] - a["x0"], b["x1"] - b["x0"])
            if gap < 0.9 * lh and ox > 0.25 * wmin and 0.6 < (a["y1"] - a["y0"]) / (b["y1"] - b["y0"]) < 1.7:
                par[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(lines[i])
    return list(groups.values())


def _join(lines):
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and ln[:1].isalpha():
            out = out[:-1] + ln
        else:
            out = (out + " " + ln).strip()
    return out


def _bubble_text(pt, rgb, b):
    x, y, w, h = b["bbox"]
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [(b["poly"] - [x, y]).astype(np.int32)], 1)
    m = cv2.erode(m, np.ones((5, 5), np.uint8))
    g = cv2.cvtColor(rgb[y:y + h, x:x + w], cv2.COLOR_RGB2GRAY)
    g = np.where(m > 0, g, 255).astype(np.uint8)
    sc = 2.0 if max(w, h) < 600 else 1.0
    im = Image.fromarray(g)
    if sc != 1.0:
        im = im.resize((int(w * sc), int(h * sc)), Image.LANCZOS)
    im = Image.fromarray(np.pad(np.asarray(im), 20, constant_values=255))
    d = pt.image_to_data(im, lang="eng", config="--oem 1 --psm 6", output_type=pt.Output.DICT)
    rows, box = {}, None
    for i, t in enumerate(d["text"]):
        t = t.strip()
        try:
            conf = float(d["conf"][i])
        except Exception:
            conf = -1
        if not t or conf < 20 or not any(c.isalnum() or c in "!?" for c in t):
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        rows.setdefault(key, []).append((d["left"][i], t))
        bx0, by0 = (d["left"][i] - 20) / sc + x, (d["top"][i] - 20) / sc + y
        bx1, by1 = bx0 + d["width"][i] / sc, by0 + d["height"][i] / sc
        box = (bx0, by0, bx1, by1) if box is None else (min(box[0], bx0), min(box[1], by0), max(box[2], bx1), max(box[3], by1))
    text = _join(" ".join(t for _, t in sorted(ws)) for _, ws in sorted(rows.items()))
    if sum(c.isalpha() for c in text) < 2 or box is None:
        return None
    return text, box


def engine_name(engine="auto"):
    from . import models
    if engine in ("auto", "rapid") and models.rapid():
        return "rapid"
    if engine == "rapid":
        raise OCRError("RapidOCR yüklenemedi: " + models.rapid_status())
    return "tesseract"


def _extend_poly(poly, rects, shape, tol):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.asarray(poly, np.int32)], 1)
    for x0, y0, x1, y1 in rects:
        m[max(0, int(y0) - 3):int(y1) + 3, max(0, int(x0) - 3):int(x1) + 3] = 1
    k = int(2 * tol + 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.asarray(poly)
    return cv2.approxPolyDP(max(cnts, key=cv2.contourArea), 1.2, True).reshape(-1, 2)


def _split_sfx(words, page_med):
    if not words:
        return words, [], []
    if page_med <= 0:
        page_med = statistics.median(w["y1"] - w["y0"] for w in words) or 1
    big = [w for w in words if (w["y1"] - w["y0"]) > SFX_H * page_med and not re.search(r"[!?]", w["t"])]
    if not big:
        return words, [], []
    ids = {id(w) for w in big}
    normal = [w for w in words if id(w) not in ids]
    stylized, rejected = [], []
    for g in _group(_lines([dict(w) for w in big])):
        g = sorted(g, key=lambda l: l["y0"])
        text = " ".join(l["text"] for l in g).strip()
        lh = statistics.median(l["y1"] - l["y0"] for l in g)
        letters = [c for c in text if c.isalpha()]
        heading = (len(letters) >= 6 and sum(c.islower() for c in letters) >= 0.5 * len(letters)
                   and not _is_sfx_like(text))          # "Contents" gibi büyük başlık (efektler büyük harf olur)
        if KEEP_STYLIZED_DIALOGUE and (heading or _is_stylized_dialogue(text, lh, page_med) or _is_short_dialogue(text)):
            stylized.append(g)
        else:
            _drop(text, "büyük-yazı/efekt")
            rejected.append((min(l["x0"] for l in g), min(l["y0"] for l in g),
                             max(l["x1"] for l in g), max(l["y1"] for l in g), text))
    return normal, stylized, rejected


def _bubble_groups(lns):
    n = len(lns)
    if n < 2:
        return [lns] if lns else []
    mh = statistics.median(l["y1"] - l["y0"] for l in lns) or 1
    par = list(range(n))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            a, b = lns[i], lns[j]
            vgap = max(a["y0"], b["y0"]) - min(a["y1"], b["y1"])
            hgap = max(a["x0"], b["x0"]) - min(a["x1"], b["x1"])
            if (vgap <= 1.0 * mh and hgap < 0) or (vgap < -0.4 * mh and hgap <= 0.8 * mh):
                par[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(lns[i])
    return list(groups.values())


def _row_sort(lines):
    if not lines:
        return lines
    mh = statistics.median(l["y1"] - l["y0"] for l in lines) or 1
    return sorted(lines, key=lambda l: (round(((l["y0"] + l["y1"]) / 2) / (0.6 * mh)), l["x0"]))


def _is_bare_title(text):
    """Yalnızca "Chapter 10" gibi çıplak bölüm numarası (çevrilmez). Arkasında başlık varsa çevrilir."""
    if not _TITLE_RE.search(text):
        return False
    rest = re.sub(r"^\W*(chapter|chap|ch|episode|ep|vol(?:ume)?)\s*\.?\s*[\d.]+\W*", "", text, flags=re.I)
    return len(_real_words(rest)) < 2


def _merge_two(a, b):
    first, second = sorted((a, b), key=lambda r: (r["box"][0], r["box"][1]))
    out = dict(a if (a.get("bubble") is not None or len(a.get("src", "")) >= len(b.get("src", ""))) else b)
    out["src"] = _join([first.get("src", ""), second.get("src", "")])
    out["box"] = [min(a["box"][0], b["box"][0]), min(a["box"][1], b["box"][1]),
                  max(a["box"][2], b["box"][2]), max(a["box"][3], b["box"][3])]
    out["bubble"] = a.get("bubble") if a.get("bubble") is not None else b.get("bubble")
    out["caps"] = bool(a.get("caps")) and bool(b.get("caps"))
    out["free"] = bool(a.get("free")) and bool(b.get("free"))
    return out


def _dedup_regions(regs):
    """Aynı balondaki ya da üst üste binen metinleri TEK bölgede birleştirir.
    (Eskiden biri atılıyordu: atılan yazı silinmeden kalıyor, aynı balona iki metin üst üste çiziliyordu.)"""
    regs = list(regs)
    changed = True
    while changed and len(regs) > 1:
        changed = False
        for i in range(len(regs)):
            for j in range(i + 1, len(regs)):
                a, b = regs[i], regs[j]
                ba, bb = a.get("bubble"), b.get("bubble")
                if ba is not None and bb is not None:
                    hit = ba == bb                      # aynı balon
                else:
                    ay0, ax0, ay1, ax1 = a["box"]
                    by0, bx0, by1, bx1 = b["box"]
                    inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
                    small = max(1, min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0)))
                    acx, acy, bcx, bcy = (ax0 + ax1) / 2, (ay0 + ay1) / 2, (bx0 + bx1) / 2, (by0 + by1) / 2
                    hit = (inter / small > 0.35 or (ax0 <= bcx <= ax1 and ay0 <= bcy <= ay1)
                           or (bx0 <= acx <= bx1 and by0 <= acy <= by1))
                if hit:
                    regs[i] = _merge_two(a, b)
                    del regs[j]
                    changed = True
                    break
            if changed:
                break
    return regs


def is_skipped_color_page(img):
    """Renkli sayfalar şimdilik çevrilmez (SKIP_COLOR_PAGES)."""
    return SKIP_COLOR_PAGES and _is_color_page(np.asarray(img.convert("RGB")))


def detect(img, tesseract_path="", rtl=True, engine="auto"):
    from . import bubbles
    W, H = img.size
    last_dropped.clear()
    rgb = np.asarray(img.convert("RGB"))
    if SKIP_COLOR_PAGES and _is_color_page(rgb):
        return []

    eng = engine_name(engine)

    gray_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    comps = _components(gray_full) if ONLY_NARRATION_OUTSIDE_BUBBLES else None
    regs, taken, leftovers = [], [], []
    if eng == "rapid":
        from . import models
        words = models.rapid_lines(rgb) or []
        # Eğik yazılar (tabela, levha, afiş, ses efekti) çizimin parçasıdır: balon yazıları düz dizilir.
        # Bunlar çevrilmez ve silinmez; eğik bir tabelaya düz Türkçe yazı oturtmak çizimi bozar.
        # Telefon/ekran yazıları da eğik durabilir: küçük harfli ve normal boydaki satırlar korunur.
        kept = []
        all_med = statistics.median(w["y1"] - w["y0"] for w in words) if words else 0
        for w in words:
            lt = [c for c in w["t"] if c.isalpha()]
            screenish = ((not lt or (len(lt) >= 3 and sum(c.islower() for c in lt) >= 0.5 * len(lt)))
                         and (w["y1"] - w["y0"]) <= 2.0 * all_med)
            if abs(w.get("angle", 0.0)) >= TILT_SKIP_DEG and not screenish:
                _drop(w["t"], "eğik-yazı")
            else:
                kept.append(w)
        words = kept
        page_med = statistics.median(w["y1"] - w["y0"] for w in words) if words else 0
        rest = list(words)
        bubs = bubbles.find(rgb, boxes=[(w["x0"], w["y0"], w["x1"], w["y1"]) for w in words])
        owner, orphan = {}, set()
        bogus = {bi for bi, b in enumerate(bubs) if _is_bogus_bubble(b["poly"])}
        polys = [b["poly"].astype(np.int32) for b in bubs]
        for w in words:
            c = (float((w["x0"] + w["x1"]) / 2), float((w["y0"] + w["y1"]) / 2))
            # İç içe bölgelerde (balonu çevreleyen açık renkli zemin de "balon" sanılabilir) en küçüğü seçilir.
            hits = [bi for bi, poly in enumerate(polys) if cv2.pointPolygonTest(poly, c, False) >= 0]
            if hits:
                owner[id(w)] = min(hits, key=lambda bi: bubs[bi]["area"])
        for w in words:
            if id(w) in owner:
                continue
            c = (float((w["x0"] + w["x1"]) / 2), float((w["y0"] + w["y1"]) / 2))
            tol = max(8.0, BUBBLE_TOL * (w["y1"] - w["y0"]))
            best, bd = None, -1e9
            for bi, poly in enumerate(polys):
                bx, by, bw, bh = bubs[bi]["bbox"]
                if not (bx - tol <= w["x0"] and w["x1"] <= bx + bw + tol and by - tol <= w["y0"] and w["y1"] <= by + bh + tol):
                    continue
                d = cv2.pointPolygonTest(poly, c, True)
                if d >= -tol and d > bd:
                    best, bd = bi, d
            if best is not None and any(owner.get(id(v)) == best for v in words if id(v) in owner):
                owner[id(w)] = best
                orphan.add(id(w))
        # Balonun içine düşen dev efekt yazısı (BA-BMP) balon metnine karışmasın: efekt ayrımına gitsin
        for bi in set(owner.values()):
            ws = [w for w in words if owner.get(id(w)) == bi]
            normal_h = [w["y1"] - w["y0"] for w in ws if (w["y1"] - w["y0"]) <= SFX_H * page_med]
            if not normal_h:
                continue
            for w in ws:
                if (w["y1"] - w["y0"]) > SFX_H * page_med and not re.search(r"[!?]", w["t"]):
                    owner.pop(id(w), None)
        for bi, b in enumerate(bubs):
            poly = b["poly"].astype(np.int32)
            is_rect = _is_rectangular_bubble(poly)
            mine = [w for w in words if owner.get(id(w)) == bi]
            rest = [w for w in rest if w not in mine]
            if not mine:
                continue
            lns = [dict(ws=[w], text=w["t"], x0=w["x0"], y0=w["y0"], x1=w["x1"], y1=w["y1"]) for w in mine]
            groups = [g for g in _bubble_groups(lns) if sum(c.isalpha() for l in g for c in l["text"]) >= 2]
            if not groups:
                continue
            gboxes = [(min(l["x0"] for l in g), min(l["y0"] for l in g), max(l["x1"] for l in g), max(l["y1"] for l in g))
                      for g in groups]
            cells = [b["poly"]] if len(groups) == 1 else [None] * len(groups)
            fake = bi in bogus                    # çizim alanı: balon gibi davranma, yazıyı serbest metin say
            if fake:
                cells = [None] * len(groups)
            for g, (x0, y0, x1, y1), cell in zip(groups, gboxes, cells):
                orp = [l for l in g if id(l["ws"][0]) in orphan]
                if orp and cell is not None:
                    cell = _extend_poly(cell, [(l["x0"], l["y0"], l["x1"], l["y1"]) for l in orp], (H, W),
                                        max(8.0, BUBBLE_TOL * statistics.median(l["y1"] - l["y0"] for l in orp)))
                text = _join(l["text"] for l in _row_sort(g))
                letters = [c for c in text if c.isalpha()]
                if len(letters) < 2:
                    continue
                if is_rect and (_looks_like_sign(text) or _is_bare_title(text)):
                    continue
                if fake and (_looks_like_sign(text) or _TITLE_RE.search(text)):
                    continue
                regs.append(dict(
                    box=[int(y0 / H * 1000), int(x0 / W * 1000), int(y1 / H * 1000) + 1, int(x1 / W * 1000) + 1],
                    kind="speech", font="comic", slant=False, caps=sum(c.isupper() for c in letters) / len(letters) > 0.8,
                    color="black", src=text, tr="", skip=False, size_mul=1.0,
                    bubble=None if fake else bubbles.to_norm(cell if cell is not None else b["poly"], W, H),
                    **({"free": True} if fake else {})))

        words, stylized, leftovers = _split_sfx(rest, page_med)
        for g in stylized:
            x0, y0 = min(l["x0"] for l in g), min(l["y0"] for l in g)
            x1, y1 = max(l["x1"] for l in g), max(l["y1"] for l in g)
            text = _join(l["text"] for l in g)
            letters = [c for c in text if c.isalpha()]
            regs.append(dict(
                box=[int(y0 / H * 1000), int(x0 / W * 1000), int(y1 / H * 1000) + 1, int(x1 / W * 1000) + 1],
                kind="speech", font="comic", slant=False,
                caps=sum(c.isupper() for c in letters) / max(1, len(letters)) > 0.8,
                color="black", src=text, tr="", skip=False, size_mul=1.0, bubble=None, free=True))
    else:
        pt = _setup(tesseract_path)
        for b in bubbles.find(rgb):
            try:
                res = _bubble_text(pt, rgb, b)
            except Exception as e:
                raise OCRError(f"Tesseract çalışmadı: {e}")
            if not res:
                continue
            text, (x0, y0, x1, y1) = res
            letters = [c for c in text if c.isalpha()]
            if _is_rectangular_bubble(b["poly"]) and (_looks_like_sign(text) or _is_bare_title(text)):
                continue
            regs.append(dict(
                box=[int(y0 / H * 1000), int(x0 / W * 1000), int(y1 / H * 1000) + 1, int(x1 / W * 1000) + 1],
                kind="speech", font="comic", slant=False, caps=sum(c.isupper() for c in letters) / len(letters) > 0.8,
                color="black", src=text, tr="", skip=False, size_mul=1.0, bubble=bubbles.to_norm(b["poly"], W, H)))
            taken.append(b)
        words = _words(img, tesseract_path)
        if taken:
            def inside(w):
                cx, cy = (w["x0"] + w["x1"]) / 2, (w["y0"] + w["y1"]) / 2
                return any(cv2.pointPolygonTest(b["poly"].astype(np.int32), (float(cx), float(cy)), False) >= 0 for b in taken)
            words = [w for w in words if not inside(w)]

    groups = _group(_lines(words))
    hs = [ln["y1"] - ln["y0"] for g in groups for ln in g]
    med_h = statistics.median(hs) if hs else 0

    for g in groups:
        g.sort(key=lambda l: l["y0"])
        text = " ".join(l["text"] for l in g).strip()
        letters = [c for c in text if c.isalpha()]
        x0, y0 = min(l["x0"] for l in g), min(l["y0"] for l in g)
        x1, y1 = max(l["x1"] for l in g), max(l["y1"] for l in g)
        piece = (x0, y0, x1, y1, text) if letters and len(text.split()) <= 2 else None
        if len(letters) < 2:
            if piece:
                leftovers.append(piece)
            continue
        if not any(sum(c.isalpha() for c in w["t"]) >= 3 and w["conf"] >= 70 for l in g for w in l["ws"]):
            if not (any(w["conf"] >= 60 for l in g for w in l["ws"]) and _is_short_dialogue(text)):
                _drop(text, "düşük-güven")
                if piece:
                    leftovers.append(piece)
                continue
        lh = statistics.median(l["y1"] - l["y0"] for l in g)

        if comps is not None:
            bx = (x0, y0, x1, y1)
            why = None
            if _TITLE_RE.search(text) and not _is_bare_title(text):
                why = "başlık"                          # "Chapter 10: <başlık>" çevrilir
            elif _is_narration_box(gray_full, comps, bx):
                why = "kutu"
            elif KEEP_UNCLOSED_BUBBLES and _is_unclosed_bubble(gray_full, comps, bx, text, len(g), lh, med_h):
                why = "açık-balon"
            elif KEEP_SCREENTONE_DIALOGUE and _is_screentone_dialogue(gray_full, bx, text, len(g), lh, med_h):
                why = "iç-ses"
            elif KEEP_STYLIZED_DIALOGUE and _is_stylized_dialogue(text, lh, med_h):
                why = "stilize"
            elif KEEP_PLAIN_BACKGROUND_TEXT and _is_plain_background_text(gray_full, comps, bx, text):
                why = "düz-zemin"
            elif KEEP_STYLIZED_DIALOGUE and _is_short_dialogue(text):
                why = "kısa-replik"
            elif KEEP_STYLIZED_DIALOGUE and _is_screen_text(text, [w["conf"] for l in g for w in l["ws"]]):
                why = "ekran-yazısı"
            if not why:
                _drop(text, "balon-dışı")
                if piece:
                    leftovers.append(piece)
                continue

        caps = sum(c.isupper() for c in letters) / len(letters) > 0.8
        shout = med_h and lh > 1.7 * med_h and text.rstrip().endswith("!")
        regs.append(dict(
            box=[int(y0 / H * 1000), int(x0 / W * 1000), int(y1 / H * 1000) + 1, int(x1 / W * 1000) + 1],
            kind="shout" if shout else "speech", font="shout" if shout else "comic",
            slant=False, caps=bool(caps), color="black", src=_join(l["text"] for l in g), tr="", skip=False, size_mul=1.0,
            bubble=None, free=bool(comps is not None and why in ("stilize", "kısa-replik", "düz-zemin", "ekran-yazısı"))))

    _attach_leftovers(regs, leftovers, W, H, med_h)
    regs = [r for r in regs if not any(re.search(p, r["src"], re.I) for p in IGNORE_TEXT)]
    regs = [r for r in regs if not (r.get("bubble") is None and (_looks_like_sign(r["src"]) or _is_bare_title(r["src"])))]
    regs = _dedup_regions(regs)
    for r in regs:
        r["clean_box"] = list(r["box"])
    regs.sort(key=lambda r: (round(r["box"][0] / 150), -r["box"][1] if rtl else r["box"][1]))
    return regs
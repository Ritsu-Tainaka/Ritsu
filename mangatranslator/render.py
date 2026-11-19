"""Bir sayfayı analiz sonucundan üretir (tamamen yerel, kota harcamaz)."""
import cv2
import numpy as np
from PIL import Image
from . import bubbles, clean, typeset


def _looks_like_sfx(r):
    if r.get("kind") in ("sfx", "sound"):
        return True
    tr = (r.get("tr") or "").strip()
    if tr.startswith("*") and tr.endswith("*") and len(tr) > 2:
        return True
    return False


def active(r, translate_sfx):
    if r.get("skip") or not r.get("tr", "").strip():
        return False
    if not translate_sfx and _looks_like_sfx(r):
        return False
    return True


def _group_masks(items, shape, grow=1.3, max_side=1100):
    """items: [(x0, y0, bool maske)]. Birbirine yakın maskeleri tek pencerede birleştirir; birleşik pencere
    tek tek pencerelerden çok büyümüyorsa (model girdisi küçültülünce ayrıntı kaybolmasın) birleştirir."""
    H, W = shape
    boxes = []
    for x0, y0, m in items:
        ys, xs = np.nonzero(m)
        if not len(xs):
            continue
        boxes.append([x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1, [(x0, y0, m)]])
    side = lambda b: max(b[2] - b[0], b[3] - b[1])
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                u = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                su = max(u[2] - u[0], u[3] - u[1])
                if su <= max_side and su <= grow * max(side(a), side(b)) + 24:
                    boxes[i] = u + [a[4] + b[4]]
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    out = []
    for x0, y0, x1, y1, parts in boxes:
        m = np.zeros((y1 - y0, x1 - x0), bool)
        for px, py, pm in parts:
            h, w = pm.shape
            sy0, sx0 = max(py, y0), max(px, x0)
            sy1, sx1 = min(py + h, y1), min(px + w, x1)
            if sy1 > sy0 and sx1 > sx0:
                m[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] |= pm[sy0 - py:sy1 - py, sx0 - px:sx1 - px]
        out.append(((x0, y0), m))
    return out


def _needs_model(arr, org, m):
    """LaMa her çağrıda 512x512'lik bir görüntü işler (yavaş işlemcide onlarca saniye). Çok küçük alanlar
    ve çevresi düz/az desenli alanlar OpenCV ile aynı kalitede silinir; model yalnızca gerçekten desenli
    (tram, çizim) zeminde çalışır."""
    ys, xs = np.nonzero(m)
    if not len(xs):
        return False
    if max(xs.max() - xs.min(), ys.max() - ys.min()) < 64 and len(xs) < 2500:
        return False
    x0, y0 = org
    H, W = arr.shape[:2]
    p = 12
    a0, b0 = max(0, x0 - p), max(0, y0 - p)
    a1, b1 = min(W, x0 + m.shape[1] + p), min(H, y0 + m.shape[0] + p)
    big = np.zeros((b1 - b0, a1 - a0), np.uint8)
    big[y0 - b0:y0 - b0 + m.shape[0], x0 - a0:x0 - a0 + m.shape[1]] = m
    ring = cv2.dilate(big, np.ones((2 * p + 1, 2 * p + 1), np.uint8)).astype(bool) & ~big.astype(bool)
    if ring.sum() < 30:
        return True
    gray = cv2.cvtColor(arr[b0:b1, a0:a1], cv2.COLOR_RGB2GRAY)
    return float(gray[ring].std()) >= 14


def _clean_spill(arr, r, bmask, inpaint):
    """Balonun dışına taşan yazı parçasını (balon kenarının ötesindeki son kelime, ör. "HER.") siler.
    Taşan harfler balonun beyazına bağlı beyaz bir konturun içinde durur: beyaz alan harf kalınlığı
    kadar kapatılınca harfler içeride kalır, çizimin geri kalanı (tram, saç) dışarıda kalır."""
    H, W = arr.shape[:2]
    x0, y0, x1, y1 = clean.to_px(r["spill_box"], W, H)
    th = float((r.get("meas") or {}).get("text_h") or 0) or 0.012 * H
    pad = int(th)
    x0, y0, x1, y1 = max(0, int(x0) - pad), max(0, int(y0) - pad), min(W, int(x1) + pad), min(H, int(y1) + pad)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    roi = arr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    white = (gray > 200).astype(np.uint8)
    n, lab = cv2.connectedComponents(white, connectivity=4)
    bm = bmask[y0:y1, x0:x1] > 0
    if n < 2:
        return
    ov = np.bincount(lab[bm].ravel(), minlength=n)
    ov[0] = 0
    if ov.max() == 0:
        return
    comp = (lab == int(ov.argmax())).astype(np.uint8)
    k = max(5, int(0.8 * th)) | 1
    closed = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ink = (closed > 0) & (comp == 0) & (gray < 160)
    # yalnızca balonun dışına (ya da kenarına) düşen harfler; balonun içi zaten temizlendi
    near_out = cv2.dilate((~bm).astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    ink &= near_out
    # Balon kenarı, diken çizgileri ve panel çerçevesi uzun, birbirine bağlı çizgilerdir; harfler ise
    # kendi başına duran küçük lekeler. Harf boyunu aşan ya da kırpılan koyu parçalar korunur.
    dark = (gray < 170).astype(np.uint8)
    far = cv2.distanceTransform((~bm).astype(np.uint8), cv2.DIST_L2, 3) > max(4.0, 0.3 * th)
    nd, dl, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    hh, ww = dark.shape
    okc = np.zeros(nd, bool)
    cand, seed = [], []
    for c in np.unique(dl[ink]):
        if c == 0:
            continue
        cx, cy, cw, ch = (int(v) for v in st[c, :4])
        if not (cw <= 1.6 * th and ch <= 1.6 * th and cx > 0 and cy > 0 and cx + cw < ww and cy + ch < hh):
            continue
        pts = np.column_stack(np.nonzero(dl == c)).astype(np.float32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        if max(rw, rh) > 4.5 * max(1.0, min(rw, rh)) and max(rw, rh) > 0.6 * th:
            continue                                   # ince uzun çizgi: diken/tarama, harf değil
        if len(pts) < 0.3 * cw * ch:
            continue                                   # eğik tek çizgi kutusunu doldurmaz; harfler doldurur
        cand.append((c, cx, cy, cw, ch))
        if float(far[dl == c].mean()) >= 0.3:          # balon kenarına yapışık değil, gerçekten dışarıda
            seed.append(c)
    if not seed or len(cand) > 12:                     # çok sayıda benzer parça = çizim deseni
        return
    # Aynı satırdaki komşu harfler ("HER." içindeki "R.") balonun kenarında kalsa da kelimeyle gider.
    box = {c: (cx, cy, cw, ch) for c, cx, cy, cw, ch in cand}
    acc, grown = set(seed), True
    while grown:
        grown = False
        for c, cx, cy, cw, ch in cand:
            if c in acc:
                continue
            for a in list(acc):
                ax, ay, aw, ah = box[a]
                gap = max(ax - (cx + cw), cx - (ax + aw), 0)
                vo = min(ay + ah, cy + ch) - max(ay, cy)
                if gap <= 0.6 * th and vo >= 0.3 * min(ah, ch):
                    acc.add(c)
                    grown = True
                    break
    # Tek başına kalan parça (bir diken, bir gölge çizgisi) kelime sayılmaz: en az iki harf yan yana olmalı.
    def near(a, c):
        ax, ay, aw, ah = box[a]
        cx, cy, cw, ch = box[c]
        gap = max(ax - (cx + cw), cx - (ax + aw), 0)
        return gap <= 0.6 * th and min(ay + ah, cy + ch) - max(ay, cy) >= 0.3 * min(ah, ch)
    acc = {a for a in acc if any(c != a and near(a, c) for c in acc)}
    if not acc:
        return
    okc[list(acc)] = True
    ink &= okc[dl] | (dark == 0)
    if ink.sum() < 0.05 * th * th:
        return
    m = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) & closed
    if inpaint is not None:
        inpaint(arr, (x0, y0), m.astype(bool))
        return
    bg = np.median(roi[comp > 0], axis=0) if comp.any() else np.array([255, 255, 255])
    roi[m > 0] = bg.astype(roi.dtype)


def render_page(img, analysis, cfg, stop=None):
    """stop: threading.Event; Durdur'a basılınca kutular arasında hemen çıkılır (Stopped)."""
    arr = np.array(img.convert("RGB"))
    if analysis.get("bubbles_v") != bubbles.VERSION:
        for r in analysis.get("regions", []):
            if not analysis.get("eco") or r.get("bubble"):
                r.pop("bubble", None)
        analysis["bubbles_v"] = bubbles.VERSION

    all_regs = [r for r in analysis.get("regions", []) if not r.get("skip") and r.get("tr", "").strip()]
    try:
        bubbles.attach(arr, all_regs)
    except Exception:
        pass
    regs = [r for r in all_regs if active(r, cfg.get("translate_sfx"))]
    # Aynı balona düşen birden çok metin (eski kayıtlar): her biri balonun tamamına yazılırsa üst üste biner;
    # bu durumda her metin kendi kutusuna yazılır.
    seen = {}
    for r in regs:
        if r.get("bubble"):
            k = str(r["bubble"])
            seen[k] = seen.get(k, 0) + 1
    shared = {k for k, v in seen.items() if v > 1}
    bm = lambda r: (bubbles.mask(arr.shape, r["bubble"]) if r.get("bubble") and str(r["bubble"]) not in shared else None)

    inp_kind = cfg.get("inpaint", "opencv")
    has_ml_inp = False
    if inp_kind != "opencv":
        from . import models
        has_ml_inp = bool(models.get_inpaint_session(inp_kind))

    def check():
        if stop is not None and stop.is_set():
            from .gemini import Stopped
            raise Stopped()

    # Yazı bulucu model (isteğe bağlı, yavaş işlemcilerde dakikalar sürebilir): yalnızca açıksa çalışır.
    # Kapalıysa ya da sonuç yoksa modelsiz yöntem kullanılır (açık zemindeki harf boyutlu mürekkep).
    tmask = None
    if cfg.get("text_detector_model", False) and regs:
        try:
            from . import models
            tmask = models.text_mask(np.asarray(img.convert("RGB")))
        except Exception:
            tmask = None
    keep = [r["box"] for r in analysis.get("regions", []) if r not in regs and r.get("box")]
    zones = None
    if tmask is None and cfg.get("erase_nearby", True) and regs:
        try:
            from . import models
            zones = models.line_mask(np.asarray(img.convert("RGB")))
        except Exception:
            zones = None

    # Hız: model tabanlı silme (LaMa) her kutu için ayrı ayrı değil, sayfa sonunda yakın alanlar
    # birleştirilerek toplu çalıştırılır (aynı kalite, çok daha az model çağrısı).
    deferred = []

    def defer(im, org, m):
        if m is not None and m.any():
            deferred.append((org[0], org[1], m.astype(bool)))
        return True

    # Balonsuz kaydedilmiş metinler (ör. balon dışındaki bir parçayla birleşmiş replik): yazı içinde
    # bulunduğu balonun şekline göre dizilsin diye balon burada (silmeden önce) eşleştirilir.
    # Yalnızca dizgide kullanılır; silme alanı değişmez.
    tbm = {}
    loose = [r for r in regs if not r.get("bubble")]
    if loose:
        try:
            Hh, Ww = arr.shape[:2]
            cands = bubbles.find(arr)
            try:                                       # çizgili (fırça darbeli) kenarlı balonlar
                cands += bubbles._find_hatched(arr, [clean.to_px(r["box"], Ww, Hh) for r in loose], cands)
            except Exception:
                pass
            centers = [((r["box"][1] + r["box"][3]) * Ww / 2000, (r["box"][0] + r["box"][2]) * Hh / 2000) for r in regs]
            for r in loose:
                x0, y0, x1, y1 = clean.to_px(r["box"], Ww, Hh)
                c = ((x0 + x1) / 2, (y0 + y1) / 2)
                best = None
                for b in cands:
                    poly = b["poly"].astype(np.int32)
                    tol = 0.25 * min(x1 - x0, y1 - y0)
                    if cv2.pointPolygonTest(poly, (float(c[0]), float(c[1])), True) < -tol:
                        continue
                    m = np.zeros((Hh, Ww), np.uint8)
                    cv2.fillPoly(m, [poly], 1)
                    if m[int(y0):int(y1), int(x0):int(x1)].mean() < 0.6:
                        continue
                    others = sum(cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0 for px, py in centers)
                    if others > 1:                         # balonda başka metin de var
                        continue
                    if best is None or b["area"] < best[0]:
                        best = (b["area"], m, bool(b.get("hatched")))
                if best is not None:
                    tbm[id(r)] = best[1:]
        except Exception:
            tbm = {}

    for r in regs:
        check()
        inpaint = defer if has_ml_inp else None
        b = bm(r)
        if b is None and not r.get("bubble") and tbm.get(id(r), (None, False))[1]:
            b = tbm[id(r)][0]                      # çizgili balon: kenar darbeleri silinmesin
        r["meas"] = clean.clean_region(arr, r.get("clean_box") or r["box"], b, inpaint) or {}
        if b is not None and r.get("spill_box"):
            try:
                _clean_spill(arr, r, b, inpaint)
            except Exception:
                pass
        if cfg.get("erase_nearby", True):
            try:
                clean.erase_detected(arr, r.get("clean_box") or r["box"], tmask, r["meas"], b, inpaint, keep, zones)
            except Exception:
                pass
    if deferred:
        from . import models
        for org, m in _group_masks(deferred, arr.shape[:2]):
            check()
            ok = False
            if _needs_model(arr, org, m):
                try:
                    ok = models.run_inpaint(arr, org, m, kind=inp_kind)
                except Exception:
                    ok = False
            if not ok:                                    # model çalışmazsa OpenCV ile sil
                x0, y0 = org
                roi = arr[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
                roi[:] = cv2.inpaint(np.ascontiguousarray(roi), m.astype(np.uint8), 4, cv2.INPAINT_TELEA)

    out = Image.fromarray(arr)
    for r in regs:
        check()
        try:
            b, guess = bm(r), False
            if b is None and not r.get("bubble"):
                t = tbm.get(id(r))
                if t is not None:
                    b, guess = t[0], not t[1]      # çizgili balon kesin bulunur: tahmin sayılmaz
            r["overflow"] = bool(typeset.draw_region(out, r, cfg.get("font_map"), b, guess))
        except Exception as e:
            r["overflow"] = True
            r["error"] = str(e)
    return out
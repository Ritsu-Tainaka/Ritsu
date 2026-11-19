"""Orijinal yazıyı siler (düz balon: renkle doldurma, desenli zemin: inpainting) ve yazı özelliklerini ölçer."""
import cv2
import numpy as np


def to_px(box, W, H, pad=0.0):
    y0, x0, y1, x1 = box
    bw, bh = (x1 - x0) * W / 1000, (y1 - y0) * H / 1000
    px0 = max(0, int(x0 * W / 1000 - bw * pad)); px1 = min(W, int(x1 * W / 1000 + bw * pad))
    py0 = max(0, int(y0 * H / 1000 - bh * pad)); py1 = min(H, int(y1 * H / 1000 + bh * pad))
    return px0, py0, px1, py1


def clean_region(img, box, bmask=None, inpaint=None):
    """img: RGB np.ndarray (yerinde değiştirilir). Ölçümleri döndürür."""
    H, W = img.shape[:2]
    X0, Y0, X1, Y1 = to_px(box, W, H, pad=0.30)      # geniş arama bölgesi
    cx0, cy0, cx1, cy1 = to_px(box, W, H, pad=0.07)   # çekirdek (kutunun kendisi + küçük taşma payı)
    if X1 - X0 < 4 or Y1 - Y0 < 4:
        return None
    roi = img[Y0:Y1, X0:X1]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    core = (cx0 - X0, cy0 - Y0, cx1 - X0, cy1 - Y0)
    inner = None
    if bmask is not None:
        bx, by, bw_, bh_ = cv2.boundingRect(bmask)
        e = max(2, int(min(bw_, bh_) * 0.02))
        inner = cv2.erode(bmask, np.ones((2 * e + 1, 2 * e + 1), np.uint8))[Y0:Y1, X0:X1]

    # Arka plan rengini yazıların bulunduğu çekirdekten değil, çevresindeki bölgeden ölç
    ring = np.ones((h, w), bool)
    ring[max(0, core[1] - 2):min(h, core[3] + 2), max(0, core[0] - 2):min(w, core[2] + 2)] = False
    if inner is not None:
        ring = (inner > 0) & ring
        if ring.sum() < 20:
            ring = inner > 0
    else:
        ring = np.zeros((h, w), bool)
        ring[max(0, core[1]):core[3], max(0, core[0]):core[2]] = True
    bg_src = gray[ring] if ring.sum() >= 20 else gray.ravel()
    if inner is None:
        mid = bg_src[(bg_src > 80) & (bg_src < 250)]
        if mid.size >= 0.15 * bg_src.size:
            bg_src = mid
    hist = cv2.GaussianBlur(np.bincount(bg_src, minlength=256).astype(np.float32).reshape(-1, 1), (1, 9), 0).ravel()
    bg = int(hist.argmax())

    if inner is None:
        # Balonsuz yazı: noktalama ve ince kenarların da yakalanması için eşik 70'e çekildi
        mask = ((gray.astype(np.int16) < bg - 70) if bg > 128 else (gray.astype(np.int16) > bg + 70)).astype(np.uint8)
    else:
        mask = (np.abs(gray.astype(np.int16) - bg) > 45).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

    def inside_core(x, y, bw, bh):
        return x >= core[0] and y >= core[1] and x + bw <= core[2] and y + bh <= core[3]

    def touches(x, y, bw, bh):
        return x <= 0 or y <= 0 or x + bw >= w or y + bh >= h

    # 1. geçiş: çekirdekteki harflerden tipik harf boyunu bul
    amin, hmin = (6, 4) if inner is not None else (16, 6)
    base = [i for i in range(1, n) if st[i][4] >= amin and st[i][3] >= hmin and inside_core(*st[i][:4])
            and st[i][4] < 0.5 * w * h and not (st[i][2] >= 0.95 * (core[2] - core[0]) and st[i][3] >= 0.6 * (core[3] - core[1]))]
    med = float(np.median([st[i][3] for i in base])) if base else 0.0
    heights, strokes = [], []
    flat_lab = lab.ravel()
    dtmax = np.zeros(n, np.float32)
    np.maximum.at(dtmax, flat_lab, dt.ravel())
    if inner is not None:
        inside = np.bincount(flat_lab, weights=(inner.ravel() > 0), minlength=n)
    sel = np.zeros(n, bool)
    ink_sel = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if inner is not None:
            if inside[i] / max(1, area) < 0.85:           # çoğu balonun dışında: çerçeve/çizim
                continue
        elif touches(x, y, bw, bh) or area > 0.25 * w * h:
            continue
        if inner is None:                                   # balonsuz: parçanın merkezi kutunun (+%12) içinde olmalı
            mx, my = 0.12 * (core[2] - core[0]), 0.12 * (core[3] - core[1])
            if med:                                         # çok geniş tek satırlarda pay, yandaki sayıyı
                mx, my = min(mx, 1.2 * med), min(my, 1.2 * med)   # (içindekiler sayfa no.) yutmasın
            if not (core[0] - mx <= x + bw / 2 <= core[2] + mx and core[1] - my <= y + bh / 2 <= core[3] + my):
                continue
        if med and bh > 1.8 * med:
            # büyük ilk harf / vurgulu harf: kutunun içindeyse yine silinir (yoksa "S B", "T Y" gibi kalıntı kalır)
            if not (bh <= 3.5 * med and inside_core(x, y, bw, bh) and area < 0.15 * w * h):
                continue
        sel[i] = True
        if bh >= 4 and area >= 6 and (not med or bh >= 0.45 * med):
            ink_sel[i] = True
            heights.append(bh)
            strokes.append(2 * float(dtmax[i]) / bh)

    keep = sel[lab].astype(np.uint8)
    if inner is None and keep.sum() > 0.55 * max(1, (core[2] - core[0]) * (core[3] - core[1])):
        keep[:] = 0; sel[:] = False; ink_sel[:] = False; heights = []
    ink_mask = ink_sel[lab] if ink_sel.any() else keep.astype(bool)
    bgc_est = np.median(roi[ring], axis=0).astype(int).tolist() if ring.sum() else [bg, bg, bg]
    bg_lum = 0.3 * bgc_est[0] + 0.59 * bgc_est[1] + 0.11 * bgc_est[2]

    if keep.sum() == 0 or (not heights and inpaint is not None and inner is None):
        if inpaint is not None and inner is None:
            m = np.zeros((h, w), np.uint8)
            m[max(0, core[1]):core[3], max(0, core[0]):core[2]] = 1
            try:
                if inpaint(img, (X0, Y0), cv2.dilate(m, np.ones((5, 5), np.uint8))):
                    def_ink = [255, 255, 255] if bg_lum < 128 else [0, 0, 0]
                    return dict(flat=False, bg=bgc_est, ink=def_ink, text_h=0.0, weight=0.0, wiped=True)
            except Exception:
                pass
        if keep.sum() == 0 and inner is None:
            m = np.zeros((h, w), np.uint8)
            m[max(0, core[1]):core[3], max(0, core[0]):core[2]] = 1
            roi[:] = cv2.inpaint(np.ascontiguousarray(roi), cv2.dilate(m, np.ones((5, 5), np.uint8)), 4, cv2.INPAINT_TELEA)
            return dict(flat=False, bg=bgc_est, ink=[255, 255, 255] if bg_lum < 128 else [0, 0, 0],
                        text_h=0.0, weight=0.0, wiped=True)
        if keep.sum() == 0:
            return None

    prelim_bg = (keep == 0) if inner is None else ((keep == 0) & (inner > 0))
    flat_thr = 14 if inner is not None else 8
    is_flat_est = prelim_bg.sum() > 20 and float(gray[prelim_bg].std()) < flat_thr
    med_h = np.median(heights) if heights else 6
    if is_flat_est or inner is not None:
        k = max(3, int(med_h * 0.16))
    elif ink_sel.any():
        inv = (keep == 0).astype(np.uint8)
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        cap = max(60, 6 * med_h * med_h)
        holes = np.zeros((h, w), np.uint8)
        for j in range(1, n2):
            x_, y_, bw_, bh_, a_ = st2[j]
            if not touches(x_, y_, bw_, bh_) and a_ <= cap:
                holes[lab2 == j] = 1
        keep = np.maximum(keep, holes)
        # Beyaz dış konturlu/gölgeli harf kenarlarını da tamamen silmek için genişletme payı artırıldı
        k = max(4, int(med_h * 0.18))
    else:
        k = max(4, int(med_h * 0.28))
    big = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))

    bgmask = (big == 0) if inner is None else ((big == 0) & (inner > 0))
    sel_bg = bgmask.copy()
    sel_bg[:, :max(0, core[0])] = False; sel_bg[:, core[2]:] = False
    sel_bg[:max(0, core[1]), :] = False; sel_bg[core[3]:, :] = False
    if sel_bg.sum() < 20:
        sel_bg = bgmask
    flat = sel_bg.sum() > 20 and float(gray[sel_bg].std()) < flat_thr

    ink_px = roi[ink_mask]
    if len(ink_px):
        lum_px = 0.3 * ink_px[:, 0] + 0.59 * ink_px[:, 1] + 0.11 * ink_px[:, 2]
        core_px = ink_px[lum_px <= np.percentile(lum_px, 25)] if bg_lum >= 128 else ink_px[lum_px >= np.percentile(lum_px, 75)]
        ink = np.median(core_px if len(core_px) else ink_px, axis=0).astype(int).tolist()
    else:
        ink = [0, 0, 0] if bg_lum >= 128 else [255, 255, 255]
    if max(ink) - min(ink) < 35:
        ink_lum = 0.3 * ink[0] + 0.59 * ink[1] + 0.11 * ink[2]
        if ink_lum < 115:
            ink = [0, 0, 0]
        elif ink_lum > 165:
            ink = [255, 255, 255]

    if flat:
        roi[big.astype(bool)] = np.median(roi[sel_bg], axis=0).astype(np.uint8)
    else:
        done = False
        if inpaint is not None:
            try:
                done = bool(inpaint(img, (X0, Y0), big))
            except Exception:
                done = False
        if not done:
            roi[:] = cv2.inpaint(np.ascontiguousarray(roi), big, 4, cv2.INPAINT_TELEA)
    bgc = np.median(roi[sel_bg], axis=0).astype(int).tolist() if sel_bg.sum() else bgc_est
    return dict(flat=bool(flat), bg=bgc, ink=ink,
                text_h=float(np.median(heights)) if heights else 0.0,
                weight=float(np.median(strokes)) if strokes else 0.0)


def grow_box(img, box_px, bg, tol=28, max_grow=0.45):
    """Temizlenmiş düz balon içinde kutuyu, çerçeveye çarpana kadar büyütür (Türkçe daha uzun olduğu için)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box_px
    bw, bh = x1 - x0, y1 - y0
    lim_x, lim_y = int(bw * max_grow), int(bh * max_grow)
    bgv = np.array(bg, dtype=np.int16)

    def ok(strip):
        if strip.size == 0:
            return False
        d = np.abs(strip.astype(np.int16) - bgv).mean(axis=-1)
        return d.mean() < tol and int((d > 70).sum()) < 2

    g = {"l": 0, "r": 0, "t": 0, "b": 0}
    for _ in range(max(lim_x, lim_y) // 2 + 1):
        moved = False
        if g["l"] < lim_x and x0 - 2 > 0 and ok(img[y0:y1, x0 - 2:x0]): x0 -= 2; g["l"] += 2; moved = True
        if g["r"] < lim_x and x1 + 2 < W and ok(img[y0:y1, x1:x1 + 2]): x1 += 2; g["r"] += 2; moved = True
        if g["t"] < lim_y and y0 - 2 > 0 and ok(img[y0 - 2:y0, x0:x1]): y0 -= 2; g["t"] += 2; moved = True
        if g["b"] < lim_y and y1 + 2 < H and ok(img[y1:y1 + 2, x0:x1]): y1 += 2; g["b"] += 2; moved = True
        if not moved:
            break
    return x0, y0, x1, y1

def snap_box(img, box, bubble_mask=None):
    """Kutuyu sayfadaki gerçek yazıya hizalar (kutuları hizala özelliği).

    img: RGB np.ndarray, box: [y0, x0, y1, x1] (0-1000). Kutu çevresinde (%30 pay) yazı pikselleri
    aranır; merkezi kutunun yakınında olan harf parçalarının dış sınırı yeni kutu olur.
    Güvenilir bir sonuç yoksa kutu olduğu gibi döner.
    """
    H, W = img.shape[:2]
    X0, Y0, X1, Y1 = to_px(box, W, H, pad=0.30)
    bx0, by0, bx1, by1 = to_px(box, W, H)
    if X1 - X0 < 8 or Y1 - Y0 < 8 or bx1 - bx0 < 4 or by1 - by0 < 4:
        return list(box)
    gray = cv2.cvtColor(np.ascontiguousarray(img[Y0:Y1, X0:X1]), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    allowed = np.ones((h, w), bool)
    if bubble_mask is not None:
        allowed = bubble_mask[Y0:Y1, X0:X1] > 0
        if allowed.sum() < 20:
            allowed = np.ones((h, w), bool)
    border = np.zeros((h, w), bool)
    border[:3, :] = border[-3:, :] = True
    border[:, :3] = border[:, -3:] = True
    ref = gray[allowed & border] if (allowed & border).sum() >= 20 else gray[allowed]
    bg = int(np.median(ref))
    ink = ((np.abs(gray.astype(np.int16) - bg) > 60) & allowed).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return list(box)
    cx0, cy0, cx1, cy1 = bx0 - X0, by0 - Y0, bx1 - X0, by1 - Y0
    mx, my = 0.10 * (cx1 - cx0), 0.10 * (cy1 - cy0)
    cand = []
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area < 4 or bh < 3:
            continue
        if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h:      # pay bölgesinin kenarına değen: çerçeve/çizim
            continue
        mxc, myc = x + bw / 2, y + bh / 2
        if not (cx0 - mx <= mxc <= cx1 + mx and cy0 - my <= myc <= cy1 + my):
            continue
        cand.append((x, y, bw, bh))
    if not cand:
        return list(box)
    med = float(np.median([c[3] for c in cand]))
    cand = [c for c in cand if c[3] <= 3.5 * med]                 # yazıdan çok büyük parçalar (çizim) atılır
    if not cand:
        return list(box)
    nx0 = min(c[0] for c in cand); ny0 = min(c[1] for c in cand)
    nx1 = max(c[0] + c[2] for c in cand); ny1 = max(c[1] + c[3] for c in cand)
    p = max(2, int(med * 0.15))
    nx0, ny0, nx1, ny1 = max(0, nx0 - p) + X0, max(0, ny0 - p) + Y0, min(w, nx1 + p) + X0, min(h, ny1 + p) + Y0
    old_a = max(1, (bx1 - bx0) * (by1 - by0))
    new_a = (nx1 - nx0) * (ny1 - ny0)
    if new_a < 0.15 * old_a or new_a > 2.2 * old_a:               # şüpheli sonuç: dokunma
        return list(box)
    # Yeni kutu yazının ortasını kaçırıyorsa (tramlı zeminde beyaz konturlu yazıda zeminin noktaları
    # "yazı" sanılıp yalnızca bir şerit seçilir) yanlış yere oturmuştur: dokunma.
    ocx, ocy = (bx0 + bx1) / 2, (by0 + by1) / 2
    if not (nx0 <= ocx <= nx1 and ny0 <= ocy <= ny1) or (ny1 - ny0) < 0.4 * (by1 - by0):
        return list(box)
    out = [int(ny0 * 1000 / H), int(nx0 * 1000 / W), int(np.ceil(ny1 * 1000 / H)), int(np.ceil(nx1 * 1000 / W))]
    out = [max(0, min(1000, v)) for v in out]
    if out[2] - out[0] < 6 or out[3] - out[1] < 6:
        return list(box)
    return out


def _shape_ok(st, i, w, h, lh):
    x, y, bw, bh, area = st[i]
    if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h:
        return False                                  # pencere kenarına değiyor: çizime/çerçeveye bağlı
    if bh > 4 * lh or bw > 4 * lh or area < 4:
        return False
    lo, hi = min(bw, bh), max(bw, bh)
    return not (hi > 2.5 * lh and hi > 6 * max(1, lo))  # ince uzun çizgi değil


def _isolated(lab, ok, n, lh):
    """Çizime yapışık parçaları ele: harf boyutlu parçanın hemen yanında büyük/çizgi mürekkep varsa
    (saç teli, kontur parçası) o parça çizimin parçasıdır, silinmez. Yazı harfleri boşlukta tek durur."""
    bad = ((lab > 0) & ~ok[lab]).astype(np.uint8)
    keep = ok[lab].astype(np.uint8)
    if not keep.any():
        return keep
    if bad.any():
        k = max(3, int(lh * 0.35)) | 1
        near = cv2.dilate(bad, np.ones((k, k), np.uint8)) > 0
        hit = np.unique(lab[near & (keep > 0)])
        ok = ok.copy()
        ok[hit[hit > 0]] = False
        keep = ok[lab].astype(np.uint8)
    return keep


def _ink_mask(roi, lh, zones=None, light_ink=False):
    """Yazı maskesi (modelsiz + isteğe bağlı satır bulucu):
    - açık zeminde: koyu, harf boyutlu parçalar;
    - koyu/desenli zeminde: yalnızca satır bulucunun 'yazı var' dediği şeritlerdeki, zeminden belirgin
      ayrışan harf boyutlu parçalar (zones yoksa bu zeminlere dokunulmaz).
    Çerçeveye/çizime bağlı, çok büyük ya da çizgi gibi ince uzun parçalar alınmaz."""
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    bg = float(np.median(gray))
    out = np.zeros((h, w), np.uint8)
    if bg >= 170:
        ink = (gray < min(110, bg - 80)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        ok = np.zeros(n, bool)
        for i in range(1, n):
            ok[i] = _shape_ok(st, i, w, h, lh)
        out |= _isolated(lab, ok, n, lh)
    elif light_ink and bg <= 90:                       # koyu zeminde beyaz yazı ("I remember him...")
        ink = (gray > max(150, bg + 90)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        ok = np.zeros(n, bool)
        for i in range(1, n):                          # tram noktaları harf boyunda değildir
            ok[i] = st[i][3] >= 0.6 * lh and _shape_ok(st, i, w, h, lh)
        out |= _isolated(lab, ok, n, lh)
    if zones is not None and zones.any():
        k = max(3, int(lh * 0.6)) | 1
        z = cv2.dilate(zones, np.ones((k, k), np.uint8)) > 0
        zb = float(np.median(gray[z])) if z.any() else bg
        dev = (np.abs(gray.astype(np.int16) - zb) > 70).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(dev, connectivity=8)
        inz = np.bincount(lab.ravel(), weights=z.ravel(), minlength=n)
        ok = np.zeros(n, bool)
        for i in range(1, n):
            ok[i] = inz[i] >= 0.8 * st[i][4] and _shape_ok(st, i, w, h, lh)
        out |= _isolated(lab, ok, n, lh)
    return out if out.any() else None


def erase_detected(img, box, tmask, meas=None, bmask=None, inpaint=None, keep_boxes=(), zones=None):
    """OCR'ın kaçırdığı harfleri (süslü "HE'S", büyük ilk harf, dağınık "T...Y") yazı bulucu maskesinden
    tamamlar: kutunun yakınındaki (yaklaşık bir satır yüksekliği) yazı parçaları da silinir.
    keep_boxes: çevrilmeyen (ör. ses efekti) bölgeler; onlara dokunulmaz."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = to_px(box, W, H)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return False
    lh = float((meas or {}).get("text_h") or 0) or max(8.0, 0.012 * H)
    e = int(1.9 * lh)
    zx0, zy0, zx1, zy1 = max(0, x0 - e), max(0, y0 - e), min(W, x1 + e), min(H, y1 + e)
    c = int(3 * lh)
    X0, Y0, X1, Y1 = max(0, zx0 - c), max(0, zy0 - c), min(W, zx1 + c), min(H, zy1 + c)
    if tmask is not None:
        tm = tmask[Y0:Y1, X0:X1]
    else:
        ink = (meas or {}).get("ink") or (0, 0, 0)
        light = 0.3 * ink[0] + 0.59 * ink[1] + 0.11 * ink[2] > 165
        tm = _ink_mask(img[Y0:Y1, X0:X1], lh, None if zones is None else zones[Y0:Y1, X0:X1], light_ink=light)
        if tm is None:
            return False
    if not tm.any():
        return False
    n, lab, st, _ = cv2.connectedComponentsWithStats(tm, connectivity=8)
    inb = None
    if bmask is not None:
        inb = np.bincount(lab.ravel(), weights=(bmask[Y0:Y1, X0:X1] > 0).ravel(), minlength=n)
    kb = [to_px(b, W, H) for b in keep_boxes]
    def gap(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
        gx = max(bx0 - ax1, ax0 - bx1, 0)
        gy = max(by0 - ay1, ay0 - by1, 0)
        return gx, gy

    cand = []
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area < 4 or bh > 4 * lh:
            continue
        if inb is not None and inb[i] < 0.85 * area:
            continue
        cx, cy = X0 + x + bw / 2, Y0 + y + bh / 2
        if any(a <= cx <= b and c_ <= cy <= d for a, c_, b, d in kb):
            continue
        cand.append((i, X0 + x, Y0 + y, X0 + x + bw, Y0 + y + bh))
    chosen = {}
    for (i, a0, b0, a1, b1) in cand:                      # kutuya yakın parçalar (yaklaşık bir satır)
        gx, gy = gap(a0, b0, a1, b1, x0, y0, x1, y1)
        if max(gx, gy) <= e and not (gx > 0 and gy > 0):
            chosen[i] = (a0, b0, a1, b1)
    for _ in range(3):                                    # seçilenlere yapışık küçük parçalar (kesme işareti, nokta)
        add = {}
        for (i, a0, b0, a1, b1) in cand:
            if i in chosen:
                continue
            for q in chosen.values():
                gx, gy = gap(a0, b0, a1, b1, *q)
                if max(gx, gy) <= 0.5 * lh:
                    add[i] = (a0, b0, a1, b1)
                    break
        if not add:
            break
        chosen.update(add)
    M = np.isin(lab, list(chosen)).astype(np.uint8) if chosen else np.zeros_like(tm)
    if not M.any():
        return False
    roi = img[Y0:Y1, X0:X1]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.int16)
    k = max(3, int(lh * 0.5)) | 1
    near = cv2.dilate(M, np.ones((k, k), np.uint8)).astype(bool) & ~M.astype(bool)
    if bmask is not None:
        near &= bmask[Y0:Y1, X0:X1] > 0
    if near.sum() < 10:
        return False
    bg = float(np.median(gray[near]))
    resid = M.astype(bool) & (np.abs(gray - bg) > 40)
    if resid.sum() < 12:
        return False                                   # zaten silinmiş
    r = max(3, int(lh * 0.25))
    er = cv2.dilate(M, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    # yalnızca hâlâ mürekkep kalan parçaları sil
    n2, lab2, _, _ = cv2.connectedComponentsWithStats(er, connectivity=8)
    hit = np.unique(lab2[resid])
    er = np.isin(lab2, hit[hit > 0]).astype(np.uint8)
    if bmask is not None:
        er &= (bmask[Y0:Y1, X0:X1] > 0).astype(np.uint8)
        fill = np.median(roi[near & ~er.astype(bool)], axis=0) if (near & ~er.astype(bool)).sum() >= 10 \
            else np.median(roi[near], axis=0)
        roi[er.astype(bool)] = fill.astype(np.uint8)
        return True
    std = float(gray[near].std())
    if std < 8:
        roi[er.astype(bool)] = np.median(roi[near], axis=0).astype(np.uint8)
        return True
    if inpaint is not None:
        try:
            if inpaint(img, (X0, Y0), er):
                return True
        except Exception:
            pass
    roi[:] = cv2.inpaint(np.ascontiguousarray(roi), er, 4, cv2.INPAINT_TELEA)
    return True

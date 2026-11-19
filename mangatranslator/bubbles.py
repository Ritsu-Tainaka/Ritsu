"""Konuşma balonu algılama (OpenCV, ek model gerekmez) ve balon içi yazı alanı."""
import cv2
import numpy as np

VERSION = 18


def _shape(comp, x, y, w, h, boxes):
    """Beyaz bileşen maskesinden balon (dış kontur, delikler dolu) çıkarır; uygun değilse None."""
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, [c], -1, 1, -1)
    fa = int(filled.sum())
    hull = cv2.contourArea(cv2.convexHull(c))
    if not fa or not hull or fa / hull < 0.54:
        return None
    # Balon çizgisine değen (kalın/büyük) harfler beyaz alanda çentik açar; o harfler balon dışında
    # sayılıp silinmezdi ("ALWA" kalıntısı). OCR'ın bu balonda bulduğu yazı kutuları, balonun dış
    # bükey zarfı içinde kalan kısmıyla balona eklenir.
    add = None
    for bx0, by0, bx1, by1 in boxes:
        lx0, ly0 = int(max(0, bx0 - x)), int(max(0, by0 - y))
        lx1, ly1 = int(min(w, bx1 - x)), int(min(h, by1 - y))
        if lx1 - lx0 < 2 or ly1 - ly0 < 2:
            continue
        if filled[ly0:ly1, lx0:lx1].mean() < 0.5:      # kutunun çoğu bu balonda değil
            continue
        if add is None:
            add = np.zeros_like(comp)
        add[ly0:ly1, lx0:lx1] = 1
    if add is not None:
        hm = np.zeros_like(comp)
        cv2.drawContours(hm, [cv2.convexHull(c)], -1, 1, -1)
        filled = np.maximum(filled, add & hm)
        cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        filled = np.zeros_like(comp)
        cv2.drawContours(filled, [c], -1, 1, -1)
        fa = int(filled.sum())
    poly = cv2.approxPolyDP(c, 1.2, True).reshape(-1, 2) + [x, y]
    return dict(poly=poly, bbox=(int(x), int(y), int(w), int(h)), area=fa, solidity=fa / hull)


def _split_leaky(comp, k, boxes, x, y):
    """Dikenli/patlama balonunun beyaz iç alanı dikenlerin arasındaki dar boşluklardan dışarıdaki beyaz
    zemine bağlanır; balon sınırı dışarı sızar, dikenler balonun "içinde" kalıp yazıyla birlikte silinir.
    Dar boğazları kesen bir açma işlemi iç alanı dış zeminden ayırır. Yazı satırları arasındaki dar
    boşluklar kesilmesin diye bileşenin içindeki yazı kutuları bu işlem sırasında dolu sayılır."""
    h, w = comp.shape
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    m = comp.copy()
    # Yalnızca kutunun içinde kalan koyu parçalar (harfler) doldurulur. Balon çizgisi gibi kutunun dışına
    # uzanan çizgiler doldurulmaz: balon kenarına taşan bir yazı ("But...") kenar çizgisini köprüleyip
    # balonu dışarıdaki zemine bağlamasın.
    holes = ((filled > 0) & (comp == 0)).astype(np.uint8)
    nh, hl, hs, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    for bx0, by0, bx1, by1 in boxes:
        lx0, ly0 = int(max(0, bx0 - x - 2)), int(max(0, by0 - y - 2))
        lx1, ly1 = int(min(w, bx1 - x + 2)), int(min(h, by1 - y + 2))
        if lx1 > lx0 and ly1 > ly0:
            pad = 0.3 * (ly1 - ly0)
            sub = hl[ly0:ly1, lx0:lx1]
            ok = np.zeros(nh, bool)
            for j in np.unique(sub[sub > 0]):
                hx, hy, hw, hh = hs[j, :4]
                ok[j] = (hx >= lx0 - pad and hy >= ly0 - pad and hx + hw <= lx1 + pad and hy + hh <= ly1 + pad)
            m[ly0:ly1, lx0:lx1] |= ok[sub].astype(np.uint8)
    bx = (m > comp).astype(np.uint8)
    for bx0, by0, bx1, by1 in boxes:
        lx0, ly0 = int(max(0, bx0 - x - 2)), int(max(0, by0 - y - 2))
        lx1, ly1 = int(min(w, bx1 - x + 2)), int(min(h, by1 - y + 2))
        if lx1 > lx0 and ly1 > ly0:
            bx[ly0:ly1, lx0:lx1] = 1
    op = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(op, connectivity=4)
    inbox = np.bincount(lab.ravel(), weights=bx.ravel().astype(np.float64), minlength=n)
    # neredeyse tamamı yazı kutusu olan parça balon değildir (yazı kutusunun kendisi)
    return [(lab == j).astype(np.uint8) for j in range(1, n) if inbox[j] < 0.6 * st[j][4]]


def _find_opencv_fallback(rgb, white_thr=200, boxes=()):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = g.shape
    dark = cv2.dilate((g <= white_thr).astype(np.uint8), np.ones((3, 3), np.uint8))
    white = 1 - dark
    n, lab, st, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    page, out = H * W, []
    k = max(7, int(0.006 * min(H, W))) | 1
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < page * 0.0006 or a > page * 0.20 or w < 15 or h < 15:
            continue
        edge_touches = int(x <= 1) + int(y <= 1) + int(x + w >= W - 1) + int(y + h >= H - 1)
        if edge_touches >= 2 or (edge_touches == 1 and a > page * 0.14):
            continue
        comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        b = _shape(comp, x, y, w, h, boxes)
        # Dar boğazla birbirine bağlı alanlar (sızıntılı dikenli balon + dış zemin) ayrı balon sayılır
        pieces = []
        for pc in _split_leaky(comp, k, boxes, x, y):
            if pc.sum() < page * 0.0006:
                continue
            px, py, pw, ph = cv2.boundingRect(pc)
            sub = pc[py:py + ph, px:px + pw]
            pb = _shape(np.ascontiguousarray(sub), x + px, y + py, pw, ph, boxes)
            if pb is not None:
                pieces.append(pb)
        if len(pieces) >= 2 or (pieces and b is None):
            out.extend(pieces)
        elif b is not None:
            out.append(b)
    return out


def _near_ink(dark, comp, k):
    """Balonun beyaz iç alanına komşu, harf boyutlu koyu lekeler (yazı). Kenar çizgileri (balonun dış
    halkası) dışarıda bırakılır: iç alandan en az bir satır mesafede başlayan ince uzun parçalar alınmaz."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    near = cv2.dilate(comp, np.ones((3 * k, 3 * k), np.uint8)) > 0
    inner = cv2.erode(cv2.dilate(comp, np.ones((k, k), np.uint8)), np.ones((k, k), np.uint8)) > 0
    keep = np.zeros(n, bool)
    hit = np.bincount(lab.ravel(), weights=inner.ravel().astype(np.float64), minlength=n)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        lo, hi = min(w, h), max(w, h)
        if hi > 6 * k or (hi > 3 * lo and a < 0.3 * w * h):
            continue                                    # uzun ince çizgi: kenar darbesi
        keep[i] = hit[i] > 0
    letters = keep[lab]
    if letters.any():
        letters = cv2.dilate(letters.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    return letters & near


def _find_hatched(rgb, boxes, found, white_thr=200):
    """Kenarı düz bir çizgi değil, tek tek fırça darbelerinden (ışınsal çizgilerden) oluşan balonlar:
    çizgilerin arası boş olduğu için iç alan dışarıdaki beyaz zeminle birleşir ve balon bulunamaz.
    Koyu çizgiler kapama işlemiyle birleştirilip kesintisiz bir duvar yapılır; içeride kalan beyaz alan
    balondur. Yalnızca normal yolla balonu bulunamayan yazı kutuları için çalışır."""
    if not boxes:
        return []
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = g.shape
    polys = [b["poly"].astype(np.int32) for b in found]
    todo = []
    for bx0, by0, bx1, by1 in boxes:
        c = (float((bx0 + bx1) / 2), float((by0 + by1) / 2))
        if not any(cv2.pointPolygonTest(p, c, False) >= 0 for p in polys):
            todo.append((bx0, by0, bx1, by1))
    if not todo:
        return []
    dark = (g <= white_thr).astype(np.uint8)
    k = max(9, int(0.012 * min(H, W))) | 1
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    white = (1 - closed).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    page, out, seen = H * W, [], set()
    for bx0, by0, bx1, by1 in todo:
        ids, cnt = np.unique(lab[int(by0):int(by1), int(bx0):int(bx1)], return_counts=True)
        for i, ct in zip(ids, cnt):
            if i == 0 or i in seen or ct < 0.15 * max(1, (bx1 - bx0) * (by1 - by0)):
                continue
            x, y, w, h, a = st[i]
            if a < page * 0.0006 or a > page * 0.12:
                continue
            if int(x <= 1) + int(y <= 1) + int(x + w >= W - 1) + int(y + h >= H - 1) >= 2:
                continue
            comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
            # kapama işlemiyle duvara yapışan yazı lekeleri balona geri katılır (dış bükey zarf içinde kalan
            # koyu bölgeler: duvar değil, yazı)
            hull = np.zeros_like(comp)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(hull, [cv2.convexHull(max(cnts, key=cv2.contourArea))], -1, 1, -1)
            wall = cv2.erode(hull, np.ones((k, k), np.uint8))
            comp = np.maximum(comp, (wall > 0) & (closed[y:y + h, x:x + w] > 0) & (_near_ink(dark[y:y + h, x:x + w], comp, k)))
            b = _shape(comp.astype(np.uint8), x, y, w, h, ())
            if b is None or b["solidity"] < 0.7:
                continue
            # gerçekten çizgili kenar mı: balonun hemen dışında çok sayıda ince uzun fırça darbesi olmalı
            # (düz çizgili, kopuk ya da panel kenarına dayanan sıradan balonlar burada elenir)
            fm = np.zeros((H, W), np.uint8)
            cv2.fillPoly(fm, [b["poly"].astype(np.int32)], 1)
            ring = (cv2.dilate(fm, np.ones((2 * k, 2 * k), np.uint8)) > 0) & (fm == 0)
            nr, _, rs, _ = cv2.connectedComponentsWithStats((dark & ring).astype(np.uint8), connectivity=8)
            fa = int(fm.sum())
            thin = sum(1 for j in range(1, nr) if rs[j][4] >= 8 and rs[j][4] < 0.02 * fa
                       and max(rs[j][2], rs[j][3]) >= 2.5 * min(rs[j][2], rs[j][3]))
            cs, _ = cv2.findContours(fm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            per = max(1.0, max(cv2.arcLength(c_, True) for c_ in cs))
            if thin < 15 or thin / per < 0.015:
                continue
            seen.add(i)
            b["hatched"] = True
            out.append(b)
    return out


def local_bubble(rgb, line_boxes, white_thr=200):
    """Yazının içinde durduğu beyaz alan dışarıdaki beyaz zemine geniş bir açıklıktan bağlıysa
    (balonun kenarına taşan bir kelime, açık kuyruk) sayfa genelinde balon bulunamaz. Burada yalnızca
    bu yazının çevresine bakılır: beyaz alan satır boyuna göre giderek daha kalın bir açmayla bölünür,
    yazıyı tamamen içine alan düzgün (oval/yuvarlak) bir parça çıkarsa balon odur.
    line_boxes: piksel (x0, y0, x1, y1) satır kutuları. Dönüş: piksel poligon ya da None."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = g.shape
    x0 = min(b[0] for b in line_boxes); y0 = min(b[1] for b in line_boxes)
    x1 = max(b[2] for b in line_boxes); y1 = max(b[3] for b in line_boxes)
    lh = float(np.median([b[3] - b[1] for b in line_boxes]))
    bw, bh = x1 - x0, y1 - y0
    rx0, ry0 = int(max(0, x0 - 1.5 * bw)), int(max(0, y0 - 1.5 * bh))
    rx1, ry1 = int(min(W, x1 + 1.5 * bw)), int(min(H, y1 + 1.5 * bh))
    sub = g[ry0:ry1, rx0:rx1]
    white = 1 - cv2.dilate((sub <= white_thr).astype(np.uint8), np.ones((3, 3), np.uint8))
    m = white.copy()
    for bx0, by0, bx1, by1 in line_boxes:                 # satırlar dolu sayılır
        m[int(by0 - ry0):int(by1 - ry0) + 1, int(bx0 - rx0):int(bx1 - rx0) + 1] = 1
    tb = np.zeros_like(m)
    tb[int(y0 - ry0):int(y1 - ry0) + 1, int(x0 - rx0):int(x1 - rx0) + 1] = 1
    tarea = float(tb.sum())
    for f in (0.6, 0.9, 1.2, 1.6):
        k = max(7, int(f * lh)) | 1
        op = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, lab = cv2.connectedComponents(op, connectivity=4)
        if n < 2:
            continue
        ov = np.bincount(lab[tb > 0].ravel(), minlength=n)
        ov[0] = 0
        j = int(ov.argmax())
        pc = (lab == j).astype(np.uint8)
        # açmanın kırptığı kenarlar geri gelsin (yalnızca bu parçaya bağlı beyaz)
        pc = cv2.dilate(pc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) & m
        if pc[0, :].any() or pc[-1, :].any() or pc[:, 0].any() or pc[:, -1].any():
            continue                                          # hâlâ dışarı sızıyor
        cnts, _ = cv2.findContours(pc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        fill = np.zeros_like(pc)
        cv2.drawContours(fill, [c], -1, 1, -1)
        fa = float(fill.sum())
        hull = cv2.contourArea(cv2.convexHull(c))
        if not hull or fa / hull < 0.85 or fa > 8 * tarea:
            continue
        if float((fill & tb).sum()) < 0.9 * tarea:
            continue
        return cv2.approxPolyDP(c, 1.2, True).reshape(-1, 2) + [rx0, ry0]
    return None


def find(rgb, white_thr=200, boxes=()):
    found = _find_opencv_fallback(rgb, white_thr, boxes)
    try:
        found += _find_hatched(rgb, list(boxes), found, white_thr)
    except Exception:
        pass
    return found


def to_norm(poly, W, H):
    return [[int(round(px * 1000 / W)), int(round(py * 1000 / H))] for px, py in poly]


def mask(shape, poly_norm):
    H, W = shape[:2]
    m = np.zeros((H, W), np.uint8)
    pts = np.array([[x * W / 1000, y * H / 1000] for x, y in poly_norm], np.int32)
    cv2.fillPoly(m, [pts], 1)
    return m


def _iou(boxA, boxB):
    ax0, ay0, ax1, ay1 = boxA
    bx0, by0, bx1, by1 = boxB
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    areaA = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    areaB = max(1.0, (bx1 - bx0) * (by1 - by0))
    return inter / (areaA + areaB - inter)


def attach(rgb, regions, lines=None):
    todo = [r for r in regions if "bubble" not in r]
    if not todo:
        return
    H, W = rgb.shape[:2]
    tb = [(r["box"][1] * W / 1000, r["box"][0] * H / 1000, r["box"][3] * W / 1000, r["box"][2] * H / 1000)
          for r in todo if r.get("box")]
    bubs = find(rgb, boxes=tb)

    for r in todo:
        r["bubble"] = None
        if not bubs:
            continue
        y0, x0, y1, x1 = r["box"]
        rbox = (x0 * W / 1000, y0 * H / 1000, x1 * W / 1000, y1 * H / 1000)
        rcx, rcy = (rbox[0] + rbox[2]) / 2, (rbox[1] + rbox[3]) / 2

        best_idx = None
        best_score = -1.0
        for idx, b in enumerate(bubs):
            bx, by, bw, bh = b["bbox"]
            bbox = (bx, by, bx + bw, by + bh)
            inside = cv2.pointPolygonTest(b["poly"].astype(np.int32), (float(rcx), float(rcy)), False) >= 0
            ov = _iou(rbox, bbox)
            if not inside and ov < 0.15:
                continue
            r_area = max(1.0, (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]))
            b_area = max(1.0, bw * bh)
            if b_area > 25 * r_area:
                continue
            score = ov + (0.5 if inside else 0.0) - 0.05 * (b_area / r_area)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            b = bubs[best_idx]
            r["bubble"] = to_norm(b["poly"], W, H)
            bx, by, bw, bh = b["bbox"]
            pad_x, pad_y = max(2, int(bw * 0.04)), max(2, int(bh * 0.04))
            nb = [
                max(0, int((by + pad_y) * 1000 / H)),
                max(0, int((bx + pad_x) * 1000 / W)),
                min(1000, int((by + bh - pad_y) * 1000 / H)),
                min(1000, int((bx + bw - pad_x) * 1000 / W)),
            ]
            if nb[2] > nb[0] + 4 and nb[3] > nb[1] + 4 and "clean_box" not in r:
                r["clean_box"] = nb


def grow_in(allowed, box_px, aspect=1.0, max_steps=600):
    H, W = allowed.shape
    bx0, by0, bx1, by1 = box_px
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    sw, sh = max(2.0, (bx1 - bx0) * 0.10), max(2.0, (by1 - by0) * 0.10)
    x0, y0, x1, y1 = cx - sw, cy - sh, cx + sw, cy + sh

    def ok(a, b, c, d):
        a, b, c, d = int(a), int(b), int(c + 0.999), int(d + 0.999)
        if a < 0 or b < 0 or c > W or d > H or c <= a or d <= b:
            return False
        return allowed[b:d, a:c].mean() >= 0.998

    if not ok(x0, y0, x1, y1):
        return None
    ax = 2.0 * min(1.0, aspect)
    ay = 2.0 * min(1.0, 1.0 / max(aspect, 1e-3))
    for _ in range(max_steps):
        if ok(x0 - ax, y0 - ay, x1 + ax, y1 + ay):
            x0, y0, x1, y1 = x0 - ax, y0 - ay, x1 + ax, y1 + ay
        else:
            break
    for _ in range(max_steps):
        moved = False
        if ok(x0 - 2, y0, x1, y1):
            x0 -= 2
            moved = True
        if ok(x0, y0, x1 + 2, y1):
            x1 += 2
            moved = True
        if ok(x0, y0 - 2, x1, y1):
            y0 -= 2
            moved = True
        if ok(x0, y0, x1, y1 + 2):
            y1 += 2
            moved = True
        if not moved:
            break
    return int(x0), int(y0), int(x1), int(y1)


def text_area(cleaned_rgb, bmask, box_px):
    x, y, w, h = cv2.boundingRect(bmask)
    k = max(2, int(min(w, h) * 0.05))
    g = cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2GRAY)
    ink = cv2.dilate((g < 160).astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    allowed = cv2.erode(bmask, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) & (1 - ink)
    return grow_in(allowed, box_px, aspect=w / max(1, h))
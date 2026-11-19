"""Bölüm numarası algılama, doğal sıralama ve tutarlılık raporu."""
import re, unicodedata

_KW = r"(?:chapter|chap|ch|bölüm|bolum|böl|episode|ep|c|#)"


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def _num(t):
    v = float(t)
    return int(v) if v.is_integer() else v


def _parse(name):
    """-> (sayı|None, güven, cilt_mi)"""
    s = unicodedata.normalize("NFKC", str(name))
    s = re.sub(r"\.(pdf|png|jpe?g|webp|cbz|cbr|zip)$", "", s, flags=re.I)
    s = re.sub(r"[\[\(\{][^\]\)\}]*[\]\)\}]", " ", s).lower()
    vm = re.search(r"(?<![a-zçğıöşü])(?:vol(?:ume)?|cilt|v)\.?\s*(\d+(?:\.\d+)?)", s)
    s = re.sub(r"(?<![a-zçğıöşü])(?:vol(?:ume)?|cilt|v)\.?\s*\d+(?:\.\d+)?", " ", s)
    s = re.sub(r"\b\d{3,4}p\b|\bx26[45]\b", " ", s)
    m = re.search(rf"(?<![a-zçğıöşü]){_KW}\.?\s*[-_:]?\s*(\d+(?:\.\d+)?)", s)
    if m:
        return _num(m.group(1)), "yüksek", False
    nums = re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d])", s)
    if not nums:
        if vm:                      # bölüm numarası yok ama cilt numarası var (ör. "Seri v01")
            return _num(vm.group(1)), "orta", True
        return None, "yok", False
    if len(nums) == 1:
        return _num(nums[0]), "yüksek" if re.search(r"\d\s*$", s) or s.strip() == nums[0] else "orta", False
    return _num(nums[-1]), "düşük", False  # birden çok sayı: sondakini al ama uyar


def extract_number(name):
    """-> (sayı|None, güven 'yüksek'|'orta'|'düşük'|'yok')"""
    return _parse(name)[:2]


def is_volume(name):
    return _parse(name)[2]


def label(ch):
    """'Cilt' (yalnızca cilt numarası varsa ve elle düzeltilmediyse) ya da 'Bölüm'."""
    return "Cilt" if ch.get("manual_num") is None and is_volume(ch["name"]) else "Bölüm"


def effective(ch):
    return ch["manual_num"] if ch["manual_num"] is not None else ch["auto_num"]


def sort_chapters(chs):
    return sorted(chs, key=lambda c: (effective(c) is None,
                                     effective(c) if effective(c) is not None else 0,
                                     natural_key(c["name"])))


def fmt_num(n):
    if n is None:
        return "?"
    return str(int(n)) if float(n).is_integer() else str(n)


def chapter_warnings(chs):
    """Bölüm id'si -> uyarı türü (liste görünümündeki 'Uyarı' sütunu için; isim eşleştirmesi yapmaz)."""
    out, by_num = {}, {}
    for c in chs:
        n = effective(c)
        if n is None:
            out[c["id"]] = "algılanamadı"
            continue
        by_num.setdefault(n, []).append(c["id"])
        if c["manual_num"] is None and c["conf"] == "düşük":
            out[c["id"]] = "belirsiz"
    for ids in by_num.values():
        if len(ids) > 1:
            for i in ids:
                out.setdefault(i, "tekrar")
    return out


def check_order(chs):
    """Eksik / tekrarlanan / belirsiz / algılanamayan bölümleri raporlar."""
    warns = []
    nums = {}
    for c in chs:
        n = effective(c)
        if n is None:
            warns.append(("algılanamadı", f"'{c['name']}' için numara bulunamadı"))
            continue
        nums.setdefault(n, []).append(c["name"])
        if c["manual_num"] is None and c["conf"] == "düşük":
            warns.append(("belirsiz", f"'{c['name']}' → {fmt_num(n)} (birden çok sayı var, kontrol edin)"))
    for n, names in nums.items():
        if len(names) > 1:
            warns.append(("tekrar", f"Numara {fmt_num(n)} {len(names)} kez: " + " | ".join(names)))
    ints = sorted(int(n) for n in nums if float(n).is_integer())
    if ints:
        have = set(ints)
        missing = [i for i in range(ints[0], ints[-1] + 1) if i not in have]
        if missing:
            txt = ", ".join(map(str, missing[:40])) + (" ..." if len(missing) > 40 else "")
            warns.append(("eksik", f"Eksik bölüm numaraları: {txt}"))
    return warns

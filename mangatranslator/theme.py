"""Hızlı ve modern koyu arayüz teması."""
import sys
import tkinter as tk
from tkinter import ttk

# Palet: "mürekkep ve fosforlu kalem". Zemin nötr mürekkep grisi (sv-ttk ile aynı ton),
# tek vurgu rengi çevirmenin balonları işaretlediği fosforlu sarı.
BG = "#1c1c1c"        # mürekkep (sv-ttk zemini)
SIDE = "#151515"      # sol menü
SURF = "#242424"      # kart yüzeyi
SURF2 = "#292929"     # alternatif satır / alan
HOVER = "#313131"
LINE = "#2f2f2f"
FG = "#ede9e0"        # hafif sıcak kağıt beyazı
MUTED = "#9b978f"
ACCENT = "#f2c14e"    # fosforlu kalem
ACCENT_BG = "#3a311c" # vurgu zemini (seçili satır, aktif menü)
ON_ACCENT = "#1a1606"
OK = "#7fcf8b"
WARN = "#f0904a"
ERR = "#ef6b6b"
FONT = ("Segoe UI", 10)
MONO = ("Consolas", 9)

STATUS_TAGS = {
    "çevrildi": OK,
    "kısmi": WARN,
    "hata": ERR,
    "çevriliyor": ACCENT,
    "durduruldu": MUTED,
    "bekliyor": FG,
}

_sv = False
BRAND_FONT = ("Segoe UI Semibold", 15)

_SV_BLUE_HUE = (0.47, 0.70)   # sv-ttk vurgu mavisinin ton aralığı (0-1)


def dark_titlebar(win):
    """Windows 10/11 pencere başlık çubuğunu DWM katmanı eklemeden koyu yapar."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        win.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
        if not hwnd:
            return
        val = ctypes.c_int(1)
        for attr in (20, 19):
            res = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attr, ctypes.byref(val), ctypes.sizeof(val)
            )
            if res == 0:
                break
        # SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED (0x0027)
        ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, 0x0027)
    except Exception:
        pass


def _fix_fonts(root):
    """Windows 10'da Segoe UI Variable yoksa Segoe UI kullan."""
    try:
        import tkinter.font as tkfont
        fams = set(tkfont.families(root))
        if any(f.startswith("Segoe UI Variable") for f in fams):
            return
        base = "Segoe UI" if "Segoe UI" in fams else None
        if not base:
            return
        semi = "Segoe UI Semibold" if "Segoe UI Semibold" in fams else base
        fonts_cfg = [
            ("SunValleyCaptionFont", base, -12),
            ("SunValleyBodyFont", base, -14),
            ("SunValleyBodyStrongFont", semi, -14),
            ("SunValleyBodyLargeFont", base, -18),
            ("SunValleySubtitleFont", semi, -20),
            ("SunValleyTitleFont", semi, -28),
            ("SunValleyTitleLargeFont", semi, -40),
            ("SunValleyDisplayFont", semi, -68),
        ]
        for name, fam, size in fonts_cfg:
            try:
                root.tk.call("font", "configure", name, "-family", fam, "-size", size)
            except Exception:
                pass
    except Exception:
        pass


def _load_brand_font(root):
    """Paketle gelen Bangers fontunu yalnızca bu işlem için yükler (logo yazısı)."""
    global BRAND_FONT
    try:
        import tkinter.font as tkfont
        from . import settings as S
        path = S.BUILTIN_FONT_DIR / "Bangers-Regular.ttf"
        if sys.platform == "win32" and path.exists():
            import ctypes
            ctypes.windll.gdi32.AddFontResourceExW(str(path), 0x10, 0)  # FR_PRIVATE
        if "Bangers" in set(tkfont.families(root)):
            BRAND_FONT = ("Bangers", 28)
    except Exception:
        pass


def _recolor_sv_accent(root):
    """sv-ttk'nin mavi vurgu görsellerini (düğme, anahtar, ilerleme çubuğu) fosforlu sarıya boyar."""
    try:
        import base64, colorsys, io
        from PIL import Image
        import sv_ttk
        from pathlib import Path
        sheet = Path(sv_ttk.__file__).parent / "theme" / "spritesheet_dark.png"
        im = Image.open(sheet).convert("RGBA")
        th, ts, tv = colorsys.rgb_to_hsv(*(int(ACCENT[i:i + 2], 16) / 255 for i in (1, 3, 5)))
        px = im.load()
        lo, hi = _SV_BLUE_HUE
        cache = {}
        for y in range(im.height):
            for x in range(im.width):
                r, g, b, a = px[x, y]
                if not a:
                    continue
                key = (r, g, b)
                if key not in cache:
                    h, s_, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
                    if s_ > 0.25 and lo <= h <= hi:
                        nr, ng, nb = colorsys.hsv_to_rgb(th, min(1.0, s_ * ts / 0.66), min(1.0, v * tv / 1.0))
                        cache[key] = (int(nr * 255), int(ng * 255), int(nb * 255))
                    else:
                        cache[key] = None
                c = cache[key]
                if c:
                    px[x, y] = (*c, a)
        buf = io.BytesIO()
        im.save(buf, "PNG")
        tcl = root.tk
        tcl.call("image", "create", "photo", "mt_sprites", "-data", base64.b64encode(buf.getvalue()).decode("ascii"))
        tcl.eval("""
            foreach {name x y width height} $::spriteinfo {
                if {[info exists ::ttk::theme::sv_dark::I($name)]} {
                    $::ttk::theme::sv_dark::I($name) copy mt_sprites -from $x $y [expr {$x+$width}] [expr {$y+$height}]
                }
            }
            image delete mt_sprites
        """)
        # Erişilebilirlik: sarı düğme üzerinde koyu yazı
        st = ttk.Style(root)
        st.configure("Accent.TButton", foreground=ON_ACCENT)
        st.map("Accent.TButton", foreground=[("disabled", "#6b6450"), ("!disabled", ON_ACCENT)])
        tcl.eval(f"set ::ttk::theme::sv_dark::colors(-accent) {ACCENT}")
        tcl.eval(f"set ::ttk::theme::sv_dark::colors(-selbg) {ACCENT_BG}")
    except Exception:
        pass


_CARD_SKIP = ("Treeview", "TScrollbar", "TProgressbar", "TNotebook", "TSeparator", "TPanedwindow")


def _in_card(w):
    p = w.master
    while p is not None:
        try:
            if str(p.cget("style")).startswith("Card."):
                return True
        except Exception:
            pass
        p = getattr(p, "master", None)
    return False


def _fit_card(w):
    """Kart içindeki ttk öğelerinin zeminini kart rengine eşitler (açık kalan şeritleri önler)."""
    try:
        cls = w.winfo_class()
        if cls in _CARD_SKIP or not _in_card(w):
            return
        cur = str(w.cget("style"))
        if cur.startswith("Card."):
            return
        if cls == "TFrame" and not cur:
            w.configure(style="Card.TFrame")
            return
        name = "Card." + (cur or cls)
        st = ttk.Style(w)
        if name not in _fitted:
            st.configure(name, background=SURF)
            _fitted.add(name)
        w.configure(style=name)
    except Exception:
        pass


_fitted = set()


def apply(root):
    """Temayı uygular."""
    global _sv
    try:
        import sv_ttk
        sv_ttk.set_theme("dark")
        root.tk.call("configure_colors")
        _fix_fonts(root)
        _sv = True
        _recolor_sv_accent(root)
    except Exception:
        _sv = False

    _load_brand_font(root)
    st = ttk.Style(root)
    if _sv:
        st.configure(".", foreground=FG)
        st.configure("TLabel", foreground=FG)
    if not _sv:
        st.theme_use("clam")
        st.configure(
            ".", background=BG, foreground=FG, fieldbackground=SURF2,
            bordercolor=LINE, darkcolor=SURF, lightcolor=SURF, troughcolor=BG,
            selectbackground=ACCENT, selectforeground="#000000",
            insertcolor=FG, font=FONT,
        )
        st.configure(
            "TButton", background=SURF2, foreground=FG, padding=(12, 6),
            relief="flat", borderwidth=1, bordercolor=LINE,
        )
        st.map(
            "TButton",
            background=[("active", HOVER), ("disabled", SURF)],
            foreground=[("disabled", "#565b66")],
        )
        st.configure(
            "Accent.TButton", background=ACCENT, foreground=ON_ACCENT,
            font=("Segoe UI Semibold", 10), padding=(14, 6),
            relief="flat", borderwidth=0,
        )
        st.map(
            "Accent.TButton",
            background=[("active", "#f6d17a"), ("disabled", "#4a4130")],
            foreground=[("disabled", "#8a8170")],
        )
        st.configure(
            "TEntry", fieldbackground=SURF2, foreground=FG, padding=(8, 5),
            bordercolor=LINE, lightcolor=ACCENT, darkcolor=LINE,
        )
        st.map("TEntry", bordercolor=[("focus", ACCENT)])

        st.configure(
            "TCombobox", fieldbackground=SURF2, foreground=FG,
            arrowcolor=MUTED, padding=(8, 5), bordercolor=LINE,
        )
        st.map(
            "TCombobox",
            fieldbackground=[("readonly", SURF2)],
            bordercolor=[("focus", ACCENT)],
            arrowcolor=[("active", FG)],
        )
        st.configure(
            "TSpinbox", fieldbackground=SURF2, foreground=FG,
            arrowcolor=MUTED, padding=(6, 5), bordercolor=LINE,
        )
        st.map("TSpinbox", bordercolor=[("focus", ACCENT)])

        st.layout("Vertical.TScrollbar", [
            ("Vertical.Scrollbar.trough", {"sticky": "ns", "children": [
                ("Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})
            ]})
        ])
        st.layout("Horizontal.TScrollbar", [
            ("Horizontal.Scrollbar.trough", {"sticky": "we", "children": [
                ("Horizontal.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})
            ]})
        ])
        st.configure(
            "TScrollbar", background="#383e4b", troughcolor=BG,
            bordercolor=BG, lightcolor="#383e4b", darkcolor="#383e4b",
            width=8, arrowsize=0,
        )
        st.map("TScrollbar", background=[("active", "#555d70")])

        st.configure(
            "Horizontal.TProgressbar", background=ACCENT,
            troughcolor=SURF2, bordercolor=LINE, thickness=6,
        )
        st.configure("TCheckbutton", background=BG, foreground=FG)
        st.configure("Switch.TCheckbutton", background=BG, foreground=FG)
        st.configure("TSeparator", background=LINE)
        st.configure("TScale", background=SURF, troughcolor=SURF2)

        root.option_add("*TCombobox*Listbox.background", SURF2)
        root.option_add("*TCombobox*Listbox.foreground", FG)
        root.option_add("*TCombobox*Listbox.selectBackground", ACCENT_BG)
        root.option_add("*TCombobox*Listbox.selectForeground", FG)

    # Kart: sv-ttk'nin görselli kartı yerine düz yüzey; iç içe kullanılabilir, çerçeve çizmez
    st.layout("Card.TFrame", [("Frame.border", {"sticky": "nswe", "children": [
        ("Frame.padding", {"sticky": "nswe"})]})])
    st.configure("Card.TFrame", background=SURF, borderwidth=0, relief="flat")
    for cls in ("TLabel", "TFrame", "TButton", "TCheckbutton", "TRadiobutton", "TEntry", "TCombobox", "TSpinbox", "TScale"):
        root.bind_class(cls, "<Map>", lambda e: _fit_card(e.widget), add="+")

    st.configure(
        "Treeview", background=SURF, fieldbackground=SURF,
        foreground=FG, rowheight=34, font=FONT, borderwidth=0,
    )
    st.configure(
        "Treeview.Heading", background=SURF, foreground=MUTED,
        font=("Segoe UI Semibold", 9), padding=(10, 8), relief="flat",
    )
    st.map("Treeview.Heading", background=[("active", SURF2)], foreground=[("active", FG)])
    st.map("Treeview", background=[("selected", ACCENT_BG)], foreground=[("selected", FG)])

    st.configure("Title.TLabel", font=("Segoe UI Semibold", 20), foreground=FG)
    st.configure("H2.TLabel", font=("Segoe UI Semibold", 13), foreground=FG)
    st.configure("Section.TLabel", font=("Segoe UI Semibold", 11), foreground=FG)
    st.configure("Group.TLabel", font=("Segoe UI Semibold", 9), foreground=MUTED, background=SURF)
    st.configure("Muted.TLabel", foreground=MUTED)
    st.configure("Small.TLabel", foreground=MUTED, font=("Segoe UI", 9))

    st.configure("Side.TFrame", background=SIDE)
    st.configure("Side.TLabel", background=SIDE, foreground=FG)
    st.configure("SideMuted.TLabel", background=SIDE, foreground=MUTED, font=("Segoe UI", 9))
    st.configure("Brand.TLabel", background=SIDE, foreground=ACCENT, font=BRAND_FONT, padding=(6, 2, 4, 8))

    # Düz (görselsiz) düğme düzeni: sol menü ve tablo başlıkları için
    flat = [("Button.border", {"sticky": "nswe", "border": "1", "children": [
        ("Button.padding", {"sticky": "nswe", "children": [("Button.label", {"sticky": "nswe"})]})]})]
    st.layout("Side.TButton", flat)
    st.layout("SideActive.TButton", flat)
    if _sv:
        st.layout("Treeview.Heading", [("Treeheading.border", {"sticky": "nswe", "children": [
            ("Treeheading.padding", {"sticky": "nswe", "children": [("Treeheading.text", {"sticky": "we"})]})]})])
        st.configure("Treeview.Heading", borderwidth=0, relief="flat")
    st.configure(
        "Side.TButton", background=SIDE, foreground=MUTED, anchor="w",
        padding=(14, 9), relief="flat", borderwidth=0, font=FONT,
    )
    st.map("Side.TButton", background=[("active", SURF)], foreground=[("active", FG)])

    st.configure(
        "SideActive.TButton", background=ACCENT_BG, foreground=ACCENT, anchor="w",
        padding=(14, 9), relief="flat", borderwidth=0, font=("Segoe UI Semibold", 10),
    )
    st.map("SideActive.TButton", background=[("active", "#463b20")], foreground=[("active", ACCENT)])

    bg = BG if not _sv else (st.lookup("TFrame", "background") or BG)
    root.configure(bg=bg)

    opts = {
        "*Text.background": SURF2,
        "*Text.foreground": FG,
        "*Text.insertBackground": FG,
        "*Text.selectBackground": ACCENT_BG,
        "*Text.relief": "flat",
        "*Text.borderWidth": 6,
        "*Text.highlightThickness": 1,
        "*Text.highlightBackground": LINE,
        "*Text.highlightColor": ACCENT,
        "*Listbox.background": SURF2,
        "*Listbox.foreground": FG,
        "*Listbox.selectBackground": ACCENT_BG,
        "*Listbox.selectForeground": FG,
        "*Listbox.relief": "flat",
        "*Listbox.borderWidth": 4,
        "*Listbox.highlightThickness": 1,
        "*Listbox.highlightBackground": LINE,
        "*Canvas.background": "#111111",
        "*Canvas.highlightThickness": 0,
        "*Menu.background": SURF2,
        "*Menu.foreground": FG,
        "*Menu.activeBackground": ACCENT_BG,
        "*Menu.activeForeground": FG,
        "*Menu.relief": "flat",
        "*Menu.borderWidth": 1,
        "*Toplevel.background": bg,
    }
    for k, v in opts.items():
        root.option_add(k, v)

    # Fare tekerleği açılır liste, sayı kutusu ve kaydırıcının değerini değiştirmesin;
    # olay pencereye geçer ve sayfa kayar (ör. Ayarlar penceresi).
    for cls in ("TCombobox", "TSpinbox", "TScale"):
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>", "<Shift-MouseWheel>"):
            root.bind_class(cls, ev, lambda e: None)

    root.after(20, lambda: dark_titlebar(root))
    root.bind_class("Toplevel", "\x3cMap\x3e", lambda e: _on_map(e.widget), add="+")
    return _sv


def _on_map(w):
    if isinstance(w, tk.Toplevel) and not getattr(w, "_dark_tb", False):
        w._dark_tb = True
        try:
            bg = BG if not _sv else (ttk.Style(w).lookup("TFrame", "background") or BG)
            w.configure(bg=bg)
        except Exception:
            pass
        w.after(10, lambda: dark_titlebar(w))


def switchify(container):
    """Onay kutularını modern anahtar (switch) görünümüne çevirir."""
    for w in container.winfo_children():
        if isinstance(w, ttk.Checkbutton):
            try:
                w.configure(style="Switch.TCheckbutton")
            except Exception:
                pass
        switchify(w)


def status_color(status):
    s = (status or "").lower()
    for k, c in STATUS_TAGS.items():
        if s.startswith(k):
            return k
    return "bekliyor"
"""Tkinter arayüzü: proje listesi, bölüm kuyruğu, sözlük, ayarlar, sayfa düzeltme ekranı."""
import json, os, queue, shutil, sys, threading, time, traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
from . import settings as S, store, source, pipeline, export, fonts, theme, ocr
from .chapters import effective, fmt_num, sort_chapters, check_order, chapter_warnings, label as ch_label
from .gemini import Gemini, GeminiError
from .render import render_page

_LINE = getattr(theme, "LINE", "#303542")
_SURF = getattr(theme, "SURF", "#20232a")
_SURF2 = getattr(theme, "SURF2", "#282c35")


def _lower_thread_priority():
    """Arka plan iş parçacığının CPU'yu kilitleyip Windows arayüzünü dondurmasını önler."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadPriority(ctypes.windll.kernel32.GetCurrentThread(), -1)
        except Exception:
            pass


def _load_opencv_cfg():
    """Ayarları yükler. Silme motoru yalnızca geçerli seçeneklerden biri olabilir (geçersizse OpenCV)."""
    cfg = S.load()
    try:
        from . import models
        if cfg.get("inpaint") != "opencv" and not models.inpaint_spec(cfg.get("inpaint")):
            cfg["inpaint"] = "opencv"                       # bilinmeyen / silinmiş model
    except Exception:
        cfg["inpaint"] = "opencv"
    return cfg


def _stripe_tree(tv):
    tv.tag_configure("odd", background=_SURF)
    tv.tag_configure("even", background=_SURF2)


def _start_freeze_watchdog(app):
    """Arayüz 4 sn'den uzun donarsa o anda hangi kodun çalıştığını donma.log'a yazar (teşhis için)."""
    import io
    app._hb = time.time()

    def beat():
        app._hb = time.time()
        try:
            app.after(250, beat)
        except Exception:
            pass
    app.after(250, beat)

    targets = [S.APP_DIR / "donma.log"]
    if not getattr(sys, "frozen", False):
        targets.append(S.BASE_DIR / "donma.log")

    def watch():
        reported = False
        while True:
            time.sleep(1.0)
            stale = time.time() - app._hb
            if stale > 4 and not reported:
                reported = True
                frames = sys._current_frames()
                buf = io.StringIO()
                buf.write(time.strftime("%Y-%m-%d %H:%M:%S") + f"  arayüz {stale:.0f} sn yanıt vermiyor\n")
                for tid, fr in frames.items():
                    name = next((t.name for t in threading.enumerate() if t.ident == tid), str(tid))
                    buf.write(f"--- iş parçacığı: {name}\n")
                    buf.write("".join(traceback.format_stack(fr)))
                for t in targets:
                    try:
                        t.parent.mkdir(parents=True, exist_ok=True)
                        with open(t, "a", encoding="utf-8") as fh:
                            fh.write(buf.getvalue() + "\n")
                    except Exception:
                        pass
            elif stale < 1:
                reported = False
    threading.Thread(target=watch, daemon=True, name="donma-izleyici").start()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ritsu")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1340, sw - 40)}x{min(820, sh - 80)}")
        self.minsize(960, 580)
        self.cfg = _load_opencv_cfg()
        self.frame = None
        theme.apply(self)

        # Sol menü
        side = ttk.Frame(self, style="Side.TFrame", width=224)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)

        brand = ttk.Frame(side, style="Side.TFrame")
        brand.pack(fill="x", padx=18, pady=(22, 18))
        ttk.Label(brand, text="Ritsu", style="Brand.TLabel").pack(anchor="w")
        ttk.Label(brand, text="Manga çeviri ve dizgi", style="SideMuted.TLabel").pack(anchor="w", pady=(3, 0))

        tk.Frame(side, bg=_LINE, height=1).pack(fill="x", padx=14, pady=(0, 14))

        self.nav = {}
        for key, txt, cmd in [
            ("projects", "▤   Projeler", self.show_projects),
            ("settings", "⚙   Ayarlar", lambda: SettingsDialog(self)),
        ]:
            b = ttk.Button(side, text=txt, style="Side.TButton", command=cmd)
            b.pack(fill="x", padx=12, pady=3)
            self.nav[key] = b

        info_card = tk.Frame(side, bg=_SURF, highlightthickness=0, padx=14, pady=12)
        info_card.pack(side="bottom", fill="x", padx=14, pady=16)
        tk.Label(info_card, text="Etkin ayarlar", bg=_SURF, fg=getattr(theme, "FG", "#ede9e0"),
                 font=("Segoe UI Semibold", 9)).pack(anchor="w", pady=(0, 4))
        self.side_info = tk.Label(info_card, text="", bg=_SURF, fg=getattr(theme, "MUTED", "#9ca3af"),
                                  font=("Segoe UI", 9), wraplength=164, justify="left")
        self.side_info.pack(anchor="w")
        self.update_side_info()

        tk.Frame(self, bg=_LINE, width=1).pack(side="left", fill="y")

        self.content = ttk.Frame(self)
        self.content.pack(side="left", fill="both", expand=True)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.show_projects()
        _start_freeze_watchdog(self)

    def _set_nav(self, active_key):
        for k, btn in self.nav.items():
            btn.configure(style="SideActive.TButton" if k == active_key else "Side.TButton")

    def update_side_info(self):
        c = _load_opencv_cfg()
        ocr_map = {"auto": "Otomatik", "rapid": "RapidOCR", "tesseract": "Tesseract"}
        mode = "Yerel OCR (" + ocr_map.get(c.get("ocr_engine", "auto"), "Otomatik") + ")"
        tl = c.get("thinking_level", "medium")
        tl_map = {"auto": "Oto", "low": "Düşük", "medium": "Orta", "high": "Yüksek", "xhigh": "Çok Yüksek"}
        try:
            from . import models as _m
            inp_label = _m.inpaint_label(c.get("inpaint", "opencv")).split(" (")[0].split(" — ")[0]
        except Exception:
            inp_label = "OpenCV"
        self.side_info.config(
            text=f"• OCR: {mode}\n• Model: {c.get('model', '')}\n• Düşünme: {tl_map.get(tl, tl)}\n• Silme: {inp_label}"
        )

    def report_callback_exception(self, exc, val, tb):
        txt = "".join(traceback.format_exception(exc, val, tb))
        log = S.APP_DIR / "hata.log"
        try:
            S.ensure_dirs()
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + txt + "\n")
        except Exception:
            pass
        messagebox.showerror("Beklenmeyen hata", f"{val}\n\nAyrıntılar: {log}", parent=self)

    def busy(self):
        return isinstance(self.frame, ProjectView) and self.frame.busy()

    def on_close(self):
        if self.busy():
            if not messagebox.askyesno("Çıkış", "Çeviri devam ediyor. Durdurup çıkılsın mı?\n(Sonra kaldığı yerden devam edilebilir.)"):
                return
            self.frame.stop.set()
            self.frame.worker.join(timeout=5)
        self.destroy()

    def show_projects(self):
        if self.busy():
            return messagebox.showinfo("Bilgi", "Çeviri çalışıyor. Projeler ekranına dönmek için önce ■ Durdur'a basın.")
        self._set_nav("projects")
        if self.frame:
            self.frame.destroy()
        self.frame = ProjectList(self)
        self.frame.pack(fill="both", expand=True)

    def open_project(self, p):
        self._set_nav("projects")
        self.frame.destroy()
        self.frame = ProjectView(self, p)
        self.frame.pack(fill="both", expand=True)


class ProjectList(ttk.Frame):
    def __init__(self, app):
        super().__init__(app.content, padding=(32, 26))
        self.app = app

        head = ttk.Frame(self)
        head.pack(fill="x")
        left = ttk.Frame(head)
        left.pack(side="left")
        ttk.Label(left, text="Projeler", style="Title.TLabel").pack(anchor="w")
        ttk.Label(left, text="Her manga serisi ayrı bir proje olarak saklanır; bölümleri ve sözlüğü birbirine karışmaz.",
                  style="Muted.TLabel").pack(anchor="w", pady=(3, 0))

        btns = ttk.Frame(head)
        btns.pack(side="right", pady=4)
        ttk.Button(btns, text="Seçiliyi Aç", command=self.open).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="＋  Yeni Proje", style="Accent.TButton", command=self.new).pack(side="left")

        self.projects = store.list_projects()
        if not self.projects:
            empty = ttk.Frame(self, style="Card.TFrame", padding=(40, 60))
            empty.pack(fill="both", expand=True, pady=(22, 0))
            inner = ttk.Frame(empty, style="Card.TFrame")
            inner.place(relx=0.5, rely=0.46, anchor="center")
            ttk.Label(inner, text="📚", font=("Segoe UI", 32), background=_SURF).pack()
            ttk.Label(inner, text="Henüz proje oluşturulmadı", style="H2.TLabel", background=_SURF).pack(pady=(8, 4))
            ttk.Label(inner, text="Çevirmek istediğiniz seri için yeni bir proje oluşturup bölüm klasörünü taratın.",
                      style="Muted.TLabel", background=_SURF).pack(pady=(0, 16))
            ttk.Button(inner, text="＋  İlk Projeni Oluştur", style="Accent.TButton", command=self.new).pack()
            return

        card = ttk.Frame(self, style="Card.TFrame", padding=1)
        card.pack(fill="both", expand=True, pady=(22, 0))
        self.tv = ttk.Treeview(card, columns=("ad", "bolum", "cevrilen", "sayfa"), show="headings", selectmode="browse")
        _stripe_tree(self.tv)
        for c, t, w, a in [
            ("ad", "Seri Adı", 440, "w"),
            ("bolum", "Toplam Bölüm", 130, "center"),
            ("cevrilen", "Tamamlanan", 140, "center"),
            ("sayfa", "Toplam Sayfa", 130, "center"),
        ]:
            self.tv.heading(c, text=t)
            self.tv.column(c, width=w, anchor=a)

        sb = ttk.Scrollbar(card, orient="vertical", command=self.tv.yview)
        self.tv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tv.pack(fill="both", expand=True)
        self.tv.bind("<Double-1>", lambda e: self.open())
        self.tv.bind("<Return>", lambda e: self.open())

        for k, p in enumerate(self.projects):
            try:
                chs = p.chapters()
            except Exception:
                chs = []
            done = sum(1 for c in chs if str(c["status"]).startswith("çevrildi"))
            pages = sum(c["n_pages"] or 0 for c in chs)
            tag = "even" if k % 2 == 0 else "odd"
            self.tv.insert("", "end", iid=str(k), tags=(tag,),
                           values=(f"  {p.name}", f"{len(chs)} bölüm", f"{done} / {len(chs)}", f"{pages} sayfa"))

        self.tv.selection_set("0")
        self.tv.focus("0")

    def new(self):
        n = simpledialog.askstring("Yeni proje", "Manga adı:", parent=self)
        if n and n.strip():
            self.app.open_project(store.create_project(n.strip()))

    def open(self):
        if not hasattr(self, "tv"):
            return
        s = self.tv.selection()
        if s:
            self.app.open_project(self.projects[int(s[0])])


class ProjectView(ttk.Frame):
    COLS = ("sira", "no", "ad", "sayfa", "durum", "uyari")

    def __init__(self, app, project):
        super().__init__(app.content, padding=(26, 20))
        self.app, self.p = app, project
        self.q = queue.Queue()
        self.stop = threading.Event()
        self.worker = None
        self._row_cache = {}

        head = ttk.Frame(self)
        head.pack(fill="x")
        ttk.Button(head, text="← Projeler", command=self.app.show_projects).pack(side="left", padx=(0, 14))
        tbox = ttk.Frame(head)
        tbox.pack(side="left")
        ttk.Label(tbox, text=project.name, style="Title.TLabel").pack(anchor="w")
        self.stats = ttk.Label(tbox, text="", style="Muted.TLabel")
        self.stats.pack(anchor="w", pady=(2, 0))

        self.btn_stop = ttk.Button(head, text="■  Durdur", command=self.request_stop, state="disabled")
        self.btn_stop.pack(side="right", pady=4)
        self.btn_start = ttk.Button(head, text="▶  Çeviriyi Başlat", style="Accent.TButton", command=self.start)
        self.btn_start.pack(side="right", padx=(0, 8), pady=4)

        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", pady=(14, 12))
        row1 = ttk.Frame(toolbar)
        row1.pack(fill="x", pady=(0, 8))
        row2 = ttk.Frame(toolbar)
        row2.pack(fill="x")

        def add_group(parent, title, items, last=False):
            card = ttk.Frame(parent, style="Card.TFrame", padding=(12, 8, 12, 10))
            card.pack(side="left", fill="both", expand=True, padx=(0, 0 if last else 10))
            ttk.Label(card, text=title, style="Group.TLabel").pack(anchor="w", padx=2, pady=(0, 6))
            brow = ttk.Frame(card, style="Card.TFrame")
            brow.pack(anchor="w")
            for idx, (txt, cmd) in enumerate(items):
                ttk.Button(brow, text=txt, command=cmd).pack(side="left", padx=(0 if idx == 0 else 4, 0))

        add_group(row1, "Kaynak", [
            ("📂 Klasör tara", self.scan),
            ("📄 Dosya ekle", self.add_files),
        ])
        add_group(row1, "Bölümler", [
            ("✎ Numara düzelt", self.fix_num),
            ("⇅ Sıralama", self.report),
            ("📖 Sözlük", lambda: GlossaryDialog(self, self.p)),
        ])
        add_group(row1, "Düzenle", [
            ("🖼 Sayfa düzenle", self.edit_pages),
            ("💬 Metinler", self.proofread),
        ], last=True)

        add_group(row2, "Çıktı", [
            ("📑 PDF yap", self.export_selected),
            ("📦 Hepsini ZIP", self.zip_all),
            ("📁 Klasörü aç", self.open_out),
        ])
        add_group(row2, "Bakım", [
            ("↻ Yeniden çiz", self.redraw_selected),
            ("↺ Sıfırla", self.reset_selected),
        ], last=True)

        # Önce alt durum ve günlük panelini en alta sabitle (kesilme sorununu önler)
        foot_card = ttk.Frame(self, style="Card.TFrame", padding=(14, 8))
        foot_card.pack(side="bottom", fill="x", pady=(10, 0))

        foot = ttk.Frame(foot_card, style="Card.TFrame")
        foot.pack(fill="x")
        self.status = ttk.Label(foot, text="Hazır.", style="Muted.TLabel", background=_SURF)
        self.status.pack(side="left")
        self.log_btn = ttk.Button(foot, text="Günlüğü gizle", command=self.toggle_log)
        self.log_btn.pack(side="right")

        self.pb = ttk.Progressbar(foot_card, mode="determinate")
        self.pb.pack(fill="x", pady=(6, 0))

        self.log_wrap = ttk.Frame(foot_card, style="Card.TFrame")
        self.log_wrap.pack(fill="x", pady=(6, 0))
        self.logbox = tk.Text(self.log_wrap, height=4, state="disabled", font=theme.MONO, wrap="word", padx=10, pady=6)
        self.logbox.pack(fill="x")

        # Ardından tabloyu kalan alana yerleştir
        card = ttk.Frame(self, style="Card.TFrame", padding=1)
        card.pack(side="top", fill="both", expand=True)
        self.tree = ttk.Treeview(card, columns=self.COLS, show="headings", selectmode="extended")
        _stripe_tree(self.tree)
        for c, t, w, a in [
            ("sira", "#", 48, "center"),
            ("no", "Bölüm", 86, "center"),
            ("ad", "Kaynak Adı", 380, "w"),
            ("sayfa", "Sayfa", 86, "center"),
            ("durum", "Durum", 170, "w"),
            ("uyari", "Uyarı", 220, "w"),
        ]:
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor=a)
        for tag, col in theme.STATUS_TAGS.items():
            self.tree.tag_configure(tag, foreground=col)

        sb = ttk.Scrollbar(card, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", lambda e: self.edit_pages())
        self.tree.bind("<Button-3>", self.context_menu)

        self.menu = tk.Menu(self, tearoff=0)
        for txt, cmd in [
            ("🖼  Sayfa düzenle", self.edit_pages),
            ("💬  Metinleri gözden geçir", self.proofread),
            (None, None),
            ("✎  Numarayı düzelt", self.fix_num),
            ("📑  PDF yap", self.export_selected),
            (None, None),
            ("↻  Yeniden çiz (kota harcamaz)", self.redraw_selected),
            ("↺  Yeniden çevir (sıfırla)", self.reset_selected),
        ]:
            if txt is None:
                self.menu.add_separator()
            else:
                self.menu.add_command(label=txt, command=cmd)

        self.refresh(full=True)
        self._after = self.after(250, self.poll)

    def request_stop(self):
        self.stop.set()
        self.btn_stop.config(state="disabled")
        self.status.config(text="Durduruluyor… (yarım kalan sayfa kaydedilmez, sonra kaldığı yerden devam eder)")
        self.log("Durdurma istendi...")

    def toggle_log(self):
        if self.log_wrap.winfo_ismapped():
            self.log_wrap.pack_forget()
            self.log_btn.config(text="Günlüğü göster")
        else:
            self.log_wrap.pack(fill="x", pady=(6, 0))
            self.log_btn.config(text="Günlüğü gizle")

    def context_menu(self, e):
        row = self.tree.identify_row(e.y)
        if row and row not in self.tree.selection():
            self.tree.selection_set(row)
        if self.tree.selection():
            try:
                self.menu.tk_popup(e.x_root, e.y_root)
            finally:
                self.menu.grab_release()

    def destroy(self):
        try:
            self.after_cancel(self._after)
        except Exception:
            pass
        super().destroy()

    def busy(self):
        return bool(self.worker and self.worker.is_alive())

    def guard(self, what):
        if self.busy():
            messagebox.showinfo("Bilgi", f"Çeviri çalışırken {what}; önce ■ Durdur'a basın.")
            return True
        return False

    def log(self, m):
        self.q.put(("log", m))

    def poll(self):
        """Arka plan olaylarını arayüzü yormadan toplu işler."""
        want_refresh, logs, n = False, [], 0
        try:
            while n < 400:
                n += 1
                k, *a = self.q.get_nowait()
                if k == "log":
                    logs.append(a[0])
                elif k == "progress":
                    self.pb.config(maximum=max(1, a[1]), value=a[0])
                    self.status.config(text=a[2])
                    want_refresh = True
                elif k == "done":
                    self.btn_start.config(state="normal")
                    self.btn_stop.config(state="disabled")
                    self.status.config(text=a[0])
                    self.refresh(full=True)
                    self._last_refresh = time.time()
                elif k == "bgdone":
                    done, res, err = a
                    self.pb.stop()
                    self.pb.config(mode="determinate", value=0)
                    self.btn_start.config(state="normal")
                    if err is not None:
                        self.status.config(text="Hata.")
                        messagebox.showerror("Hata", str(err), parent=self)
                    elif done:
                        done(res)
        except queue.Empty:
            pass
        if logs:
            self.logbox.config(state="normal")
            self.logbox.insert("end", "\n".join(logs[-150:]) + "\n")
            if int(self.logbox.index("end-1c").split(".")[0]) > 2000:
                self.logbox.delete("1.0", "800.0")
            self.logbox.see("end")
            self.logbox.config(state="disabled")
        if want_refresh and time.time() - getattr(self, "_last_refresh", 0) > 3.5:
            self.refresh(full=False)
            self._last_refresh = time.time()
        self._after = self.after(250, self.poll)

    def run_bg(self, label, job, done=None):
        if self.busy():
            return messagebox.showinfo("Bilgi", "Başka bir işlem sürüyor; bitmesini bekleyin.")
        self.status.config(text=label)
        self.pb.config(mode="indeterminate")
        self.pb.start(12)
        self.btn_start.config(state="disabled")

        def work():
            _lower_thread_priority()
            try:
                self.q.put(("bgdone", done, job(), None))
            except Exception as e:
                self.q.put(("bgdone", None, None, e))
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def refresh(self, full=True):
        """Tabloyu silip baştan kurmak yerine yalnızca değişen satırları yerinde günceller."""
        try:
            chs = sort_chapters(self.p.chapters())
        except Exception:
            return
        warn = chapter_warnings(chs)
        new_ids = [str(c["id"]) for c in chs]
        cur_ids = list(self.tree.get_children())

        if full or new_ids != cur_ids:
            sel = self.tree.selection()
            self.tree.delete(*cur_ids)
            self._row_cache.clear()
            for i, c in enumerate(chs, 1):
                iid = str(c["id"])
                parity = "even" if i % 2 == 0 else "odd"
                tags = (theme.status_color(c["status"]), parity)
                vals = (
                    i,
                    fmt_num(effective(c)) + (" ✎" if c["manual_num"] is not None else ""),
                    c["name"],
                    f"{c['done']}/{c['n_pages']}",
                    "●  " + str(c["status"]),
                    ("⚠ " + warn[c["id"]]) if c["id"] in warn else "",
                )
                self.tree.insert("", "end", iid=iid, tags=tags, values=vals)
                self._row_cache[iid] = (tags, vals)
            for s in sel:
                if self.tree.exists(s):
                    self.tree.selection_add(s)
        else:
            for i, c in enumerate(chs, 1):
                iid = str(c["id"])
                parity = "even" if i % 2 == 0 else "odd"
                tags = (theme.status_color(c["status"]), parity)
                vals = (
                    i,
                    fmt_num(effective(c)) + (" ✎" if c["manual_num"] is not None else ""),
                    c["name"],
                    f"{c['done']}/{c['n_pages']}",
                    "●  " + str(c["status"]),
                    ("⚠ " + warn[c["id"]]) if c["id"] in warn else "",
                )
                if self._row_cache.get(iid) != (tags, vals):
                    self.tree.item(iid, tags=tags, values=vals)
                    self._row_cache[iid] = (tags, vals)

        done = sum(1 for c in chs if str(c["status"]).startswith("çevrildi"))
        pages = sum(c["n_pages"] or 0 for c in chs)
        self.stats.config(text=f"{len(chs)} bölüm   •   {done} çevrildi   •   {pages} sayfa")

    def selected_ids(self):
        return [int(i) for i in self.tree.selection()]

    def scan(self):
        if self.guard("tarama yapılamaz"):
            return
        root = filedialog.askdirectory(title="Manga ana klasörü (içinde bölüm klasörleri / PDF / CBZ)")
        if not root:
            return
        def job():
            res = source.scan_root(root)
            for r in res:
                if "error" not in r:
                    self.p.upsert_chapter(**r)
            return res

        def done(res):
            errs = [r["error"] for r in res if "error" in r]
            self.refresh(full=True)
            self.status.config(text=f"{len(res) - len(errs)} bölüm bulundu.")
            if errs:
                messagebox.showwarning("Okunamayan kaynaklar", "\n".join(errs[:15]))
            self.report(show_only_if_warn=True)
        self.run_bg("Klasör taranıyor…", job, done)

    def add_files(self):
        if self.guard("dosya eklenemez"):
            return
        paths = filedialog.askopenfilenames(title="Bölüm dosyaları", filetypes=[("Manga", "*.cbz *.zip *.cbr *.pdf")])
        if not paths:
            return
        def job():
            errs = []
            for pth in paths:
                r = source.describe_file(pth)
                if "error" in r:
                    errs.append(r["error"])
                else:
                    self.p.upsert_chapter(**r)
            return errs

        def done(errs):
            self.refresh(full=True)
            self.status.config(text=f"{len(paths) - len(errs)} dosya eklendi.")
            if errs:
                messagebox.showwarning("Eklenemeyen dosyalar", "\n".join(errs[:15]))
            self.report(show_only_if_warn=True)
        self.run_bg("Dosyalar okunuyor…", job, done)

    def fix_num(self):
        ids = self.selected_ids()
        if not ids:
            return messagebox.showinfo("Bilgi", "Önce listeden bir bölüm seçin.")
        c = self.p.chapter(ids[0])
        v = simpledialog.askstring("Bölüm numarası", f"'{c['name']}' için doğru bölüm numarası\n(boş bırakırsanız otomatik algılamaya döner):",
                                   initialvalue=fmt_num(effective(c)), parent=self)
        if v is None:
            return
        v = v.strip().replace(",", ".")
        try:
            self.p.update_chapter(c["id"], manual_num=float(v) if v else None)
        except ValueError:
            return messagebox.showerror("Hata", "Geçerli bir sayı girin (ör. 12 veya 12.5).")
        self.refresh(full=True)

    def report(self, show_only_if_warn=False):
        chs = sort_chapters(self.p.chapters())
        w = check_order(chs)
        if show_only_if_warn and not w:
            return
        t = tk.Toplevel(self)
        t.title("Sıralama raporu")
        t.geometry("640x500")
        wrap = ttk.Frame(t, padding=16)
        wrap.pack(fill="both", expand=True)
        box = tk.Text(wrap, wrap="word", font=theme.MONO, padx=10, pady=10)
        box.pack(fill="both", expand=True)
        box.insert("end", "ÇEVİRİ SIRASI\n" + "\n".join(f"{i:>3}. {ch_label(c)} {fmt_num(effective(c)):<7} {c['name']}  ({c['n_pages']} sayfa)" for i, c in enumerate(chs, 1)))
        box.insert("end", "\n\nUYARILAR\n" + ("\n".join(f"• [{k}] {m}" for k, m in w) if w else "Sorun bulunamadı."))
        box.config(state="disabled")

    def start(self):
        if self.busy():
            return
        if any(isinstance(w, (PageEditor, ProofreadDialog)) and w.winfo_exists() for w in self.winfo_children()):
            return messagebox.showinfo("Bilgi", "Önce sayfa düzenleme / metin penceresini kapatın (aynı dosyalara yazarlar).")
        ids = self.selected_ids() or [c["id"] for c in self.p.chapters() if not c["status"].startswith("çevrildi")]
        if not ids:
            return messagebox.showinfo("Bilgi", "Çevrilecek bölüm yok.")
        if not S.get_api_key():
            messagebox.showwarning("API anahtarı", "Önce Menü > Ayarlar'dan Gemini API anahtarını girin.")
            return SettingsDialog(self.app)
        chs = sort_chapters([c for c in self.p.chapters() if c["id"] in ids])
        w = check_order(sort_chapters(self.p.chapters()))
        txt = "Çeviri sırası:\n" + ", ".join(fmt_num(effective(c)) for c in chs[:40]) + (" ..." if len(chs) > 40 else "")
        if w:
            txt += "\n\nUyarılar:\n" + "\n".join(f"• {m}" for _, m in w[:8])
        if not messagebox.askyesno("Sıralamayı onayla", txt + "\n\nBaşlansın mı?"):
            return
        self.stop.clear()
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        cfg = _load_opencv_cfg()
        pl = pipeline.Pipeline(self.p, cfg, S.get_api_key(), self.stop, self.log,
                               lambda d, t, m: self.q.put(("progress", d, t, m)))

        def work():
            _lower_thread_priority()
            try:
                ok = pl.run(ids)
                self.q.put(("done", "Tamamlandı." if ok else "Durduruldu (kaldığı yerden devam edilebilir)."))
            except Exception as e:
                self.log(f"HATA: {e}")
                if not isinstance(e, GeminiError):
                    self.log(traceback.format_exc())
                self.q.put(("done", "Hata nedeniyle durdu."))
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def edit_pages(self):
        if self.guard("sayfa düzenlenemez"):
            return
        ids = self.selected_ids()
        if not ids:
            return messagebox.showinfo("Bilgi", "Önce bir bölüm seçin.")
        c = self.p.chapter(ids[0])
        if not any((self.p.work_dir(c["id"]) / f"p{i:04d}.json").exists() for i in range(c["n_pages"])):
            return messagebox.showinfo("Bilgi", "Bu bölümde henüz çevrilmiş sayfa yok.")
        PageEditor(self, self.p, c)

    def proofread(self):
        if self.guard("metinler düzenlenemez"):
            return
        ids = self.selected_ids()
        if not ids:
            return messagebox.showinfo("Bilgi", "Önce bir bölüm seçin.")
        c = self.p.chapter(ids[0])
        if not any((self.p.work_dir(c["id"]) / f"p{i:04d}.json").exists() for i in range(c["n_pages"])):
            return messagebox.showinfo("Bilgi", "Bu bölümde henüz çevrilmiş sayfa yok.")
        ProofreadDialog(self, self.p, c)

    def redraw_selected(self, ids=None):
        if self.busy():
            return
        ids = ids or self.selected_ids()
        if not ids:
            return messagebox.showinfo("Bilgi", "Önce listeden bir veya birkaç bölüm seçin.")
        if any(isinstance(w, (PageEditor, ProofreadDialog)) and w.winfo_exists() for w in self.winfo_children()):
            return messagebox.showinfo("Bilgi", "Önce sayfa düzenleme / metin penceresini kapatın.")
        self.stop.clear()
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        pl = pipeline.Pipeline(self.p, _load_opencv_cfg(), "", self.stop, self.log,
                               lambda d, t, m: self.q.put(("progress", d, t, m)), client=object())

        def work():
            _lower_thread_priority()
            try:
                ok = pl.redraw(ids)
                self.q.put(("done", "Yeniden çizim tamamlandı." if ok else "Durduruldu."))
            except Exception as e:
                self.log(f"HATA: {e}")
                self.log(traceback.format_exc())
                self.q.put(("done", "Hata nedeniyle durdu."))
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def reset_selected(self):
        ids = self.selected_ids()
        if not ids:
            return messagebox.showinfo("Bilgi", "Önce listeden bir veya birkaç bölüm seçin.")
        if self.worker and self.worker.is_alive():
            return messagebox.showinfo("Bilgi", "Çeviri çalışırken sıfırlanamaz; önce durdurun.")
        if not messagebox.askyesno("Sıfırla", "Seçili bölümlerin çevrilmiş sayfaları ve elle yaptığınız düzeltmeler SİLİNİR; "
                                   "bölüm baştan çevrilir (API kotası harcar). Devam?"):
            return
        for cid in ids:
            shutil.rmtree(self.p.work_dir(cid), ignore_errors=True)
            self.p.update_chapter(cid, status="bekliyor", done=0, failed=0)
        self.refresh(full=True)
        self.status.config(text=f"{len(ids)} bölüm sıfırlandı. ▶ Çeviriyi Başlat ile yeniden çevirebilirsiniz.")

    def export_selected(self):
        ids = self.selected_ids()
        if not ids:
            return messagebox.showinfo("Bilgi", "Önce listeden bir veya birkaç bölüm seçin.")
        if self.worker and self.worker.is_alive():
            return messagebox.showinfo("Bilgi", "Çeviri çalışırken dışa aktarılamaz; önce durdurun.")
        ans = messagebox.askyesnocancel("Yarım bölüm", "Henüz çevrilmemiş sayfalar da orijinal halleriyle eklensin mi?\n\n"
                                        "Evet: tüm sayfalar (çevrilmeyenler orijinal)\nHayır: yalnızca çevrilmiş sayfalar")
        if ans is None:
            return
        cfg = _load_opencv_cfg()

        def job():
            done, errs = [], []
            for cid in ids:
                c = self.p.chapter(cid)
                try:
                    done.append(export.export_all(self.p, c, cfg, only_translated=not ans).name)
                except Exception as e:
                    errs.append(f"{c['name']}: {e}")
            return done, errs

        def finish(res):
            done, errs = res
            self.refresh(full=False)
            self.status.config(text=f"{len(done)} dosya hazır.")
            msg = f"{len(done)} dosya hazır (Klasörü aç düğmesiyle açabilirsiniz)."
            if errs:
                msg += "\n\nYapılamayanlar:\n" + "\n".join(errs[:10])
            messagebox.showinfo("Dışa aktarma", msg, parent=self)
        self.run_bg("PDF hazırlanıyor…", job, finish)

    def open_out(self):
        import subprocess
        d = str(self.p.path / "out")
        if sys.platform.startswith("win"):
            os.startfile(d)
        else:
            subprocess.Popen(["xdg-open", d])

    def zip_all(self):
        path = filedialog.asksaveasfilename(defaultextension=".zip", initialfile=f"{self.p.name}.zip", filetypes=[("ZIP", "*.zip")])
        if path:
            def done(n):
                self.status.config(text=f"ZIP hazır: {n} bölüm.")
                messagebox.showinfo("ZIP hazır", f"{n} bölüm PDF'i eklendi.", parent=self)
            self.run_bg("ZIP hazırlanıyor…", lambda: export.export_zip(self.p, path), done)


class GlossaryDialog(tk.Toplevel):
    def __init__(self, master, project):
        super().__init__(master)
        self.title("Terim Sözlüğü")
        self.geometry("620x520")
        self.p = project

        wrap = ttk.Frame(self, padding=(20, 16))
        wrap.pack(fill="both", expand=True)

        ttk.Label(wrap, text="Terim Sözlüğü", style="H2.TLabel").pack(anchor="w")
        ttk.Label(wrap, text="‘Otomatik’ terimleri Gemini sayfalardan önerir; düzenleyip kaydettiğinizde kalıcı olur.",
                  style="Muted.TLabel").pack(anchor="w", pady=(2, 12))

        card = ttk.Frame(wrap, style="Card.TFrame", padding=1)
        card.pack(fill="both", expand=True)
        self.tv = ttk.Treeview(card, columns=("en", "tr", "k"), show="headings", selectmode="browse")
        _stripe_tree(self.tv)
        for c, t, w in [("en", "İngilizce", 220), ("tr", "Türkçe", 220), ("k", "Kaynak", 100)]:
            self.tv.heading(c, text=t)
            self.tv.column(c, width=w)
        sb = ttk.Scrollbar(card, orient="vertical", command=self.tv.yview)
        self.tv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tv.pack(fill="both", expand=True)
        self.tv.bind("<<TreeviewSelect>>", self.pick)

        edit_card = ttk.Frame(wrap, style="Card.TFrame", padding=(12, 10))
        edit_card.pack(fill="x", pady=(12, 0))
        self.en, self.tr = tk.StringVar(), tk.StringVar()
        ttk.Entry(edit_card, textvariable=self.en, width=22).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Entry(edit_card, textvariable=self.tr, width=22).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(edit_card, text="Ekle / Güncelle", style="Accent.TButton", command=self.add).pack(side="left", padx=(0, 4))
        ttk.Button(edit_card, text="Sil", command=self.delete).pack(side="left")
        self.load()

    def load(self):
        self.tv.delete(*self.tv.get_children())
        for idx, (en, tr, auto) in enumerate(self.p.glossary()):
            tag = "even" if idx % 2 == 0 else "odd"
            self.tv.insert("", "end", iid=en, tags=(tag,), values=(en, tr, "otomatik" if auto else "elle"))

    def pick(self, _):
        s = self.tv.selection()
        if s:
            v = self.tv.item(s[0])["values"]
            self.en.set(v[0])
            self.tr.set(v[1])

    def add(self):
        if self.en.get().strip() and self.tr.get().strip():
            self.p.set_term(self.en.get(), self.tr.get(), 0)
            self.load()

    def delete(self):
        for s in self.tv.selection():
            self.p.del_term(s)
        self.load()


class SettingsDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        self.title("Ayarlar")
        self.app = app
        self.geometry(f"680x{max(460, min(860, self.winfo_screenheight() - 80))}")
        self.cfg = _load_opencv_cfg()

        tk.Frame(self, bg=_LINE, height=1).pack(side="bottom", fill="x")
        bottom = ttk.Frame(self, padding=(20, 12))
        bottom.pack(side="bottom", fill="x")
        ttk.Button(bottom, text="Kaydet", style="Accent.TButton", command=self.save).pack(side="right")
        ttk.Button(bottom, text="İptal", command=self.destroy).pack(side="right", padx=(0, 8))

        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)
        cv = tk.Canvas(outer, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)

        f = ttk.Frame(cv, padding=(22, 18))
        win = cv.create_window(0, 0, window=f, anchor="nw")
        cv.configure(background=ttk.Style(self).lookup("TFrame", "background") or theme.BG)
        f.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfigure(win, width=e.width))
        self.bind("<MouseWheel>", lambda e: cv.yview_scroll(int(-e.delta / 120) or (-1 if e.delta > 0 else 1), "units"))

        ttk.Label(f, text="Ayarlar", style="Title.TLabel").pack(anchor="w")
        ttk.Label(f, text="API bağlantısı, düşünme seviyesi, detaylı çeviri üslubu ve OCR motorları.",
                  style="Muted.TLabel").pack(anchor="w", pady=(2, 14))

        # --- KART 1: GEMINI ---
        c1 = ttk.Frame(f, style="Card.TFrame", padding=(16, 14))
        c1.pack(fill="x", pady=(0, 12))
        ttk.Label(c1, text="Gemini bağlantısı", style="Section.TLabel", background=_SURF).pack(anchor="w", pady=(0, 8))

        ttk.Label(c1, text="Gemini API anahtarı (aistudio.google.com/apikey):", background=_SURF).pack(anchor="w")
        k0, src0 = S.get_api_key_info()
        self.stored_key = bool(k0) and not src0.startswith("ortam")
        self.key = tk.StringVar(value=k0.strip())
        ttk.Entry(c1, textvariable=self.key, show="•").pack(fill="x", pady=(3, 2))
        ttk.Label(c1, text=f"Kullanılan kaynak: {src0}" + (f" (son 4 hane: {k0.strip()[-4:]})" if k0 else ""),
                  style="Small.TLabel", background=_SURF).pack(anchor="w")

        ttk.Label(c1, text="Model (görsel ve metin destekli):", background=_SURF).pack(anchor="w", pady=(10, 3))
        mrow = ttk.Frame(c1, style="Card.TFrame")
        mrow.pack(fill="x")
        self.model = ttk.Combobox(mrow, values=[self.cfg["model"]])
        self.model.set(self.cfg["model"])
        self.model.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(mrow, text="Modelleri Listele", command=self.list_models).pack(side="left")

        ttk.Label(c1, text="Düşünme seviyesi (Gemini 3 ve 4 serisi):", background=_SURF).pack(anchor="w", pady=(10, 3))
        self.THINKING_LEVELS = {
            "Orta (medium — hız, kalite ve token tasarrufu için önerilen)": "medium",
            "Düşük (low — en hızlı yanıt ve en az düşünme tokenı)": "low",
            "Yüksek (high — karmaşık diyaloglarda derin akıl yürütme)": "high",
            "Çok Yüksek (xhigh — yalnızca Hydrogen, en ağır)": "xhigh",
            "Otomatik (modelin kendi varsayılanı)": "auto",
        }
        self.thinking = ttk.Combobox(c1, values=list(self.THINKING_LEVELS), state="readonly")
        cur_tl = self.cfg.get("thinking_level", "medium")
        self.thinking.set(next((k for k, v in self.THINKING_LEVELS.items() if v == cur_tl), next(iter(self.THINKING_LEVELS))))
        self.thinking.pack(fill="x", pady=(0, 2))
        ttk.Label(c1, text="Gemini 4 Hydrogen varsayılan olarak 'high' kullanır; 'medium' seçmek token tüketimini ~%55 azaltır.",
                  style="Small.TLabel", background=_SURF, wraplength=600, justify="left").pack(anchor="w", pady=(0, 4))

        row = ttk.Frame(c1, style="Card.TFrame")
        row.pack(fill="x", pady=(8, 6))
        self.rpm = tk.IntVar(value=self.cfg["rpm"])
        self.side = tk.IntVar(value=self.cfg["render_long_side"])
        ttk.Label(row, text="Dakikada istek (RPM):", background=_SURF).pack(side="left")
        ttk.Spinbox(row, from_=1, to=300, textvariable=self.rpm, width=6).pack(side="left", padx=(6, 16))
        ttk.Label(row, text="PDF render (px):", background=_SURF).pack(side="left")
        ttk.Spinbox(row, from_=1200, to=4000, increment=200, textvariable=self.side, width=7).pack(side="left", padx=6)

        self.sfx = tk.BooleanVar(value=self.cfg["translate_sfx"])
        self.auto = tk.BooleanVar(value=self.cfg["auto_export"])
        self.cbz = tk.BooleanVar(value=self.cfg.get("export_cbz", False))
        ttk.Checkbutton(c1, text="Efekt yazılarını (SFX) da çevir", variable=self.sfx).pack(anchor="w", pady=2)
        ttk.Checkbutton(c1, text="Bölüm bitince PDF'i otomatik üret", variable=self.auto).pack(anchor="w", pady=2)
        ttk.Checkbutton(c1, text="PDF'e ek olarak .cbz arşivi de üret", variable=self.cbz).pack(anchor="w", pady=2)

        # --- KART 2: DETAYLI ÇEVİRİ ÜSLUBU VE TALİMATLAR ---
        c_style = ttk.Frame(f, style="Card.TFrame", padding=(16, 14))
        c_style.pack(fill="x", pady=(0, 12))
        ttk.Label(c_style, text="Çeviri üslubu ve talimatlar", style="Section.TLabel", background=_SURF).pack(anchor="w", pady=(0, 8))

        ttk.Label(c_style, text="Çeviri üslubu:", background=_SURF).pack(anchor="w")
        self.STYLES = {
            "Doğal ve Akıcı Diyalog (günlük konuşma, deyim ve karakter tonu)": "natural",
            "Detaylı ve Sadık Çeviri (anlam inceliklerini ve hitapları eksiksiz korur)": "detailed",
            "Edebi ve Dramatik Ton (aksiyon, fantastik ve tarihi seriler)": "dramatic",
        }
        self.style_cb = ttk.Combobox(c_style, values=list(self.STYLES), state="readonly")
        cur_st = self.cfg.get("translation_style", "natural")
        self.style_cb.set(next((k for k, v in self.STYLES.items() if v == cur_st), next(iter(self.STYLES))))
        self.style_cb.pack(fill="x", pady=(3, 8))

        ttk.Label(c_style, text="Ek çeviri talimatları (her çeviri isteğine otomatik eklenir):", background=_SURF).pack(anchor="w")
        self.custom_prompt = tk.Text(c_style, height=3, wrap="word", font=("Segoe UI", 9), padx=8, pady=6)
        self.custom_prompt.pack(fill="x", pady=(3, 3))
        if self.cfg.get("custom_prompt"):
            self.custom_prompt.insert("1.0", self.cfg["custom_prompt"])
        ttk.Label(c_style, text="Örn: Karakterler birbirine 'sen' diye hitap etsin, Japonca ekleri (-kun, -san, -sama) koru, argo ifadeleri doğal çevir.",
                  style="Small.TLabel", background=_SURF, wraplength=600, justify="left").pack(anchor="w")

        # --- KART 3: ÇEVİRİ MODU VE OCR ---
        c2 = ttk.Frame(f, style="Card.TFrame", padding=(16, 14))
        c2.pack(fill="x", pady=(0, 12))
        ttk.Label(c2, text="Yazı okuma (OCR)", style="Section.TLabel", background=_SURF).pack(anchor="w", pady=(0, 8))

        ttk.Label(c2, text="OCR motoru:", background=_SURF).pack(anchor="w")
        self.OCRS = {
            "Otomatik (RapidOCR kuruluysa o, değilse Tesseract)": "auto",
            "RapidOCR (PaddleOCR modelleri, küçük/eğik yazılarda daha doğru)": "rapid",
            "Tesseract": "tesseract",
        }
        self.ocr = ttk.Combobox(c2, values=list(self.OCRS), state="readonly")
        cur_ocr = self.cfg.get("ocr_engine", "auto")
        self.ocr.set(next((k for k, v in self.OCRS.items() if v == cur_ocr), next(iter(self.OCRS))))
        self.ocr.pack(fill="x", pady=(3, 2))
        self.ocr_st = ttk.Label(c2, text="RapidOCR: kontrol ediliyor...", style="Small.TLabel", background=_SURF)
        self.ocr_st.pack(anchor="w", pady=(0, 6))

        self.snap = tk.BooleanVar(value=self.cfg.get("text_snap", True))
        ttk.Checkbutton(c2, text="Kutuları hizala: kutuyu ve silme alanını sayfadaki gerçek yazıya oturt",
                        variable=self.snap).pack(anchor="w", pady=(0, 2))
        self.near = tk.BooleanVar(value=self.cfg.get("erase_nearby", True))
        ttk.Checkbutton(c2, text="OCR'ın okuyamadığı yakın harfleri de sil (\"HE'S\", \"T…Y\" gibi; hızlı, modelsiz)",
                        variable=self.near).pack(anchor="w", pady=(0, 2))
        self.tdet = tk.BooleanVar(value=self.cfg.get("text_detector_model", False))
        ttk.Checkbutton(c2, text="Bunun için yazı bulucu model kullan (daha isabetli; ~95 MB, yavaş işlemcide sayfa başı dakikalar)",
                        variable=self.tdet).pack(anchor="w", pady=(0, 2))
        from . import models as _models
        ttk.Label(c2, text="Yazı bulucu: " + _models.text_det_status(), style="Small.TLabel",
                  background=_SURF).pack(anchor="w", pady=(0, 8))

        threading.Thread(target=self._check_models, daemon=True).start()

        r2 = ttk.Frame(c2, style="Card.TFrame")
        r2.pack(fill="x", pady=(0, 8))
        ttk.Label(r2, text="Tesseract yolu:", background=_SURF).pack(side="left")
        self.tess = tk.StringVar(value=self.cfg.get("tesseract_path", ""))
        ttk.Entry(r2, textvariable=self.tess).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r2, text="Gözat", command=lambda: self.tess.set(filedialog.askopenfilename(parent=self) or self.tess.get())).pack(side="left")

        r3 = ttk.Frame(c2, style="Card.TFrame")
        r3.pack(fill="x", pady=(0, 4))
        self.batch = tk.IntVar(value=self.cfg.get("eco_batch", 200))
        self.rtl = tk.BooleanVar(value=self.cfg.get("rtl", True))
        ttk.Label(r3, text="İstek başına en fazla kutu:", background=_SURF).pack(side="left")
        ttk.Spinbox(r3, from_=10, to=500, textvariable=self.batch, width=6).pack(side="left", padx=(6, 14))
        ttk.Checkbutton(r3, text="Sağdan sola okuma (Manga)", variable=self.rtl).pack(side="left")
        ttk.Label(c2, text="İpucu: 200-300 kutu ayarı 1-2 bölümü tek istekte göndererek günlük RPD kotasını korur.",
                  style="Small.TLabel", background=_SURF, wraplength=600, justify="left").pack(anchor="w")

        # --- KART: YAZI SİLME MOTORU ---
        c_inp = ttk.Frame(f, style="Card.TFrame", padding=(16, 14))
        c_inp.pack(fill="x", pady=(0, 12))
        ttk.Label(c_inp, text="Yazı silme", style="Section.TLabel", background=_SURF).pack(anchor="w", pady=(0, 8))
        ttk.Label(c_inp, text="Orijinal yazı silinirken arkadaki çizimin nasıl doldurulacağı:", background=_SURF).pack(anchor="w")
        from . import models as _models
        self.INPAINTS = {label: kind for kind, label in _models.inpaint_choices()}
        self.inp = ttk.Combobox(c_inp, values=list(self.INPAINTS), state="readonly")
        cur_inp = self.cfg.get("inpaint", "opencv")
        self.inp.set(next((k for k, v in self.INPAINTS.items() if v == cur_inp), next(iter(self.INPAINTS))))
        self.inp.pack(fill="x", pady=(3, 6))
        self.inp.bind("<<ComboboxSelected>>", lambda _e: threading.Thread(target=self._check_lama, daemon=True).start())

        r_l = ttk.Frame(c_inp, style="Card.TFrame")
        r_l.pack(fill="x", pady=(0, 2))
        self.lama_btn = ttk.Button(r_l, text="Seçili modeli indir", command=self.download_lama)
        self.lama_btn.pack(side="left")
        ttk.Button(r_l, text="Test et", command=self.test_inpaint).pack(side="left", padx=(6, 0))
        self.lama_pb = ttk.Progressbar(r_l, length=140, mode="determinate")
        self.lama_pb.pack(side="left", padx=8)
        self.lama_st = ttk.Label(c_inp, text="Durum: kontrol ediliyor...", style="Small.TLabel", background=_SURF,
                                 wraplength=620, justify="left")
        self.lama_st.pack(anchor="w", pady=(2, 0))
        ttk.Label(c_inp, text="Manga-LaMa desenli zemindeki yazıyı iz bırakmadan siler (CPU'da kutu başına ~1-3 sn). "
                              "Seçip 'Seçili modeli indir'e, ardından 'Test et'e basın. Model yoksa ya da hata verirse "
                              "çeviri OpenCV ile devam eder.",
                  style="Small.TLabel", background=_SURF, wraplength=620, justify="left").pack(anchor="w", pady=(2, 0))
        threading.Thread(target=self._check_lama, daemon=True).start()

        # --- KART 4: FONTLAR ---
        c3 = ttk.Frame(f, style="Card.TFrame", padding=(16, 14))
        c3.pack(fill="x")
        ttk.Label(c3, text="Yazı tipleri", style="Section.TLabel", background=_SURF).pack(anchor="w", pady=(0, 4))
        ttk.Label(c3, text="Orijinal metnin türüne göre kullanılacak Türkçe destekli fontları seçin:",
                  style="Muted.TLabel", background=_SURF).pack(anchor="w", pady=(0, 8))

        self.fm = {}
        opts = ["(varsayılan)"] + fonts.choices()
        for st, lab in fonts.STYLE_LABELS.items():
            r = ttk.Frame(c3, style="Card.TFrame")
            r.pack(fill="x", pady=3)
            ttk.Label(r, text=lab, width=22, background=_SURF).pack(side="left")
            cb = ttk.Combobox(r, values=opts, state="readonly")
            cb.set(self.cfg["font_map"].get(st, "(varsayılan)"))
            cb.pack(side="left", fill="x", expand=True)
            self.fm[st] = cb

        fbot = ttk.Frame(c3, style="Card.TFrame")
        fbot.pack(fill="x", pady=(10, 0))
        ttk.Button(fbot, text="＋ Font Dosyası Ekle (.ttf/.otf)", command=self.add_font).pack(side="left")
        ttk.Label(fbot, text="Türkçe karakter (ğ ş ı İ ö ç ü) içermeyen fontlar reddedilir.",
                  style="Small.TLabel", background=_SURF).pack(side="left", padx=10)

        theme.switchify(f)

    def _ui(self, fn):
        try:
            self.after(0, fn)
        except Exception:
            pass

    def _check_models(self):
        from . import models
        st = models.rapid_status()
        def show():
            if self.winfo_exists():
                self.ocr_st.config(text="RapidOCR durumu: " + st)
        self._ui(show)

    def _selected_inpaint(self):
        return self.INPAINTS.get(self.inp.get(), "opencv")

    def _check_lama(self):
        from . import models
        kind = self._selected_inpaint()
        st = models.inpaint_status(kind)
        ready = (kind == "opencv") or models.inpaint_ready(kind) or kind not in models.INPAINT_MODELS

        def show():
            if self.winfo_exists():
                self.lama_st.config(text="Silme modeli durumu: " + st)
                self.lama_btn.config(state="disabled" if ready else "normal")
        self._ui(show)

    def test_inpaint(self):
        from . import models
        kind = self._selected_inpaint()
        self.lama_st.config(text="Test ediliyor... Model ilk kez yüklenirken pencere birkaç saniye yanıt vermeyebilir, bekleyin.")
        self.update_idletasks()

        def work():
            ok, msg = models.selftest_inpaint(kind)
            self._ui(lambda: self.winfo_exists() and self.lama_st.config(text=("✔ " if ok else "✘ ") + msg))
        threading.Thread(target=work, daemon=True).start()

    def download_lama(self):
        from . import models
        kind = self._selected_inpaint()
        if kind == "opencv" or kind not in models.INPAINT_MODELS:
            messagebox.showinfo("Silme Modeli", "OpenCV yerleşiktir, indirme gerekmez.", parent=self)
            return
        self.lama_btn.config(state="disabled")
        self.lama_stop = threading.Event()

        def progress(got, total):
            def upd():
                if self.winfo_exists():
                    if total:
                        self.lama_pb.config(maximum=total, value=got)
                    self.lama_st.config(text=f"İndiriliyor: {got / 1e6:.0f} / {total / 1e6:.0f} MB" if total
                                        else f"İndiriliyor: {got / 1e6:.0f} MB")
            self._ui(upd)

        def work():
            sha = None
            try:
                _, sha = models.download_inpaint(kind, progress, self.lama_stop)
                err = None
            except Exception as e:
                err = e

            def done():
                if not self.winfo_exists():
                    return
                if err is not None:
                    self.lama_btn.config(state="normal")
                    self.lama_st.config(text="İndirme başarısız: " + str(err))
                    messagebox.showerror("Silme Modeli", str(err), parent=self)
                else:
                    self.lama_pb.config(value=0)
                    self.lama_st.config(text=f"İndirildi. SHA-256: {sha}  — şimdi 'Test et' ile doğrulayın.")
            self._ui(done)
        threading.Thread(target=work, daemon=True).start()

    def list_models(self):
        key, model = self.key.get().strip(), self.model.get()
        self.config(cursor="watch")

        def work():
            try:
                ms, err = Gemini(key, model).list_models(), None
            except Exception as e:
                ms, err = None, e

            def show():
                if not self.winfo_exists():
                    return
                self.config(cursor="")
                if err is not None:
                    messagebox.showerror("Hata", str(err), parent=self)
                else:
                    self.model.config(values=ms)
                    messagebox.showinfo("Modeller", f"{len(ms)} model bulundu; listeden seçebilirsiniz.", parent=self)
            self._ui(show)
        threading.Thread(target=work, daemon=True).start()

    def add_font(self):
        p = filedialog.askopenfilename(filetypes=[("Font", "*.ttf *.otf")], parent=self)
        if p:
            S.ensure_dirs()
            shutil.copy(p, S.USER_FONT_DIR)
            if not fonts.has_turkish(str(S.USER_FONT_DIR / p.split("/")[-1].split("\\")[-1])):
                messagebox.showwarning("Uyarı", "Bu fontta bazı Türkçe harfler yok; seçilse bile kullanılmaz.", parent=self)
            messagebox.showinfo("Eklendi", "Kaydedip ayarları yeniden açınca listede görünür.", parent=self)

    def save(self):
        try:
            rpm, side, batch = (int(v.get()) for v in (self.rpm, self.side, self.batch))
        except (tk.TclError, ValueError):
            return messagebox.showerror("Hata", "Sayısal alanlara geçerli bir tam sayı girin.", parent=self)
        self.cfg.update(
            model=self.model.get().strip() or S.DEFAULTS["model"],
            thinking_level=self.THINKING_LEVELS.get(self.thinking.get(), "medium"),
            translation_style=self.STYLES.get(self.style_cb.get(), "natural"),
            custom_prompt=self.custom_prompt.get("1.0", "end").strip(),
            rpm=max(1, rpm),
            render_long_side=max(800, min(6000, side)),
            translate_sfx=self.sfx.get(),
            auto_export=self.auto.get(),
            export_cbz=self.cbz.get(),
            tesseract_path=self.tess.get().strip(),
            ocr_engine=self.OCRS.get(self.ocr.get(), "auto"),
            inpaint=self.INPAINTS.get(self.inp.get(), "opencv"),
            text_snap=self.snap.get(),
            erase_nearby=self.near.get(),
            text_detector_model=self.tdet.get(),
            eco_batch=max(1, batch),
            rtl=self.rtl.get(),
            font_map={k: v.get() for k, v in self.fm.items() if v.get() != "(varsayılan)"},
        )
        for k in ("mode", "pages_per_request", "send_long_side", "bubble_detector", "text_detector"):
            self.cfg.pop(k, None)
        S.save(self.cfg)
        inp_kind = self.cfg.get("inpaint", "opencv")
        if inp_kind != "opencv":
            from . import models
            if not models.inpaint_ready(inp_kind):
                messagebox.showwarning("Silme Modeli", "Seçili silme modeli henüz indirilmemiş. İndirene kadar yazılar "
                                       "OpenCV ile silinir; Ayarlar'daki 'Seçili modeli indir' düğmesini kullanın.", parent=self)
        key, where = self.key.get().strip(), None
        if key:
            where = S.set_api_key(key)
        elif self.stored_key and messagebox.askyesno("API anahtarı", "Anahtar alanı boş. Kayıtlı API anahtarı silinsin mi?", parent=self):
            S.delete_api_key()
        env = os.environ.get("GEMINI_API_KEY", "")
        if env and env != self.key.get().strip():
            messagebox.showinfo("Bilgi", "Windows'ta GEMINI_API_KEY ortam değişkeni de tanımlı. Uygulama artık Ayarlar'daki anahtarı "
                                "öncelikli kullanır; yine de eski değişkeni silmeniz önerilir.", parent=self)
        if where == "dosya":
            messagebox.showinfo("Bilgi", "Anahtar yerel dosyada saklandı (keyring kurulu değil).", parent=self)
        self.app.cfg = self.cfg
        if hasattr(self.app, "update_side_info"):
            self.app.update_side_info()
        self.destroy()


class PageEditor(tk.Toplevel):
    """Sayfa düzeltme: önbellekli görsel yükleme, gecikmeli yeniden ölçekleme ve hızlı kutu sürükleme."""
    def __init__(self, view, project, ch):
        super().__init__(view)
        self.title(f"Sayfa Düzenle — {ch['name']}")
        self.geometry("1340x830")
        self.view, self.p, self.ch, self.cfg = view, project, ch, _load_opencv_cfg()
        self.wd = project.work_dir(ch["id"])
        self.idx, self.sel, self.drag, self.scale = 0, None, None, 1.0
        self.new_mode, self.rendering, self._tok, self._img_key = False, False, 0, None
        self._resize_job = None
        self._box_ids = []

        left = ttk.Frame(self, style="Side.TFrame", padding=(8, 12))
        left.pack(side="left", fill="y")
        ttk.Label(left, text="Sayfalar", style="SideMuted.TLabel").pack(pady=(0, 8))
        lwrap = ttk.Frame(left, style="Side.TFrame")
        lwrap.pack(fill="both", expand=True)
        self.pages = tk.Listbox(lwrap, width=6, exportselection=False, justify="center", font=("Segoe UI Semibold", 10))
        lsb = ttk.Scrollbar(lwrap, orient="vertical", command=self.pages.yview)
        self.pages.configure(yscrollcommand=lsb.set)
        for i in range(ch["n_pages"]):
            self.pages.insert("end", f"{i + 1}")
        lsb.pack(side="right", fill="y")
        self.pages.pack(side="left", fill="both", expand=True)
        self.pages.bind("<<ListboxSelect>>", self.on_page_select)

        right = ttk.Frame(self, padding=(12, 12), width=336)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self.cv = tk.Canvas(self, bg="#0c0d10", cursor="crosshair")
        self.cv.pack(side="left", fill="both", expand=True)
        self.cv.bind("<ButtonPress-1>", self.down)
        self.cv.bind("<B1-Motion>", self.move)
        self.cv.bind("<ButtonRelease-1>", self.up)
        self.cv.bind("<Configure>", self._on_canvas_resize)

        mcard = ttk.Frame(right, style="Card.TFrame", padding=(12, 10))
        mcard.pack(fill="x", pady=(0, 8))
        ttk.Label(mcard, text="Metin", style="Section.TLabel", background=_SURF).pack(anchor="w", pady=(0, 4))
        ttk.Label(mcard, text="Orijinal (OCR hatası varsa düzeltip yeniden çevirebilirsiniz):", style="Small.TLabel", background=_SURF).pack(anchor="w")
        self.src = tk.Text(mcard, height=3, wrap="word")
        self.src.pack(fill="x", pady=(2, 6))
        ttk.Label(mcard, text="Türkçe çeviri:", style="Small.TLabel", background=_SURF).pack(anchor="w")
        self.tr = tk.Text(mcard, height=5, wrap="word")
        self.tr.pack(fill="x", pady=(2, 8))
        ttk.Button(mcard, text="✓  Uygula ve Yeniden Çiz (Ctrl+S)", style="Accent.TButton", command=self.apply).pack(fill="x")

        tcard = ttk.Frame(right, style="Card.TFrame", padding=(12, 10))
        tcard.pack(fill="x", pady=(0, 8))
        ttk.Label(tcard, text="Yazı stili", style="Section.TLabel", background=_SURF).pack(anchor="w", pady=(0, 6))
        self.font = ttk.Combobox(tcard, values=list(fonts.STYLE_LABELS), state="readonly")
        self.font.pack(fill="x", pady=(0, 6))

        srow = ttk.Frame(tcard, style="Card.TFrame")
        srow.pack(fill="x")
        ttk.Label(srow, text="Boyut çarpanı:", style="Small.TLabel", background=_SURF).pack(side="left")
        self.mul_lbl = ttk.Label(srow, text="1.00×", style="Small.TLabel", background=_SURF)
        self.mul_lbl.pack(side="right")
        self.mul = tk.DoubleVar(value=1.0)
        self.mul.trace_add("write", lambda *_: self.mul_lbl.config(text=f"{self.mul.get():.2f}×"))
        ttk.Scale(tcard, from_=0.5, to=1.8, variable=self.mul).pack(fill="x", pady=(2, 6))

        self.caps, self.slant, self.skip = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        for t, v in [("BÜYÜK HARF", self.caps), ("Eğik yazı (İtalik)", self.slant), ("Orijinal kalsın (çevirme)", self.skip)]:
            ttk.Checkbutton(tcard, text=t, variable=v).pack(anchor="w", pady=1)

        acard = ttk.Frame(right, style="Card.TFrame", padding=(12, 10))
        acard.pack(fill="x", pady=(0, 8))
        ttk.Label(acard, text="Kutu ve sayfa araçları", style="Section.TLabel", background=_SURF).pack(anchor="w", pady=(0, 6))
        for t, c in [
            ("Temizleme alanı = kutu", self.clean_eq),
            ("＋ Yeni kutu çiz", self.arm_new),
            ("🗑 Seçili kutuyu sil", self.delete),
            ("↻ Seçili kutuyu yeniden çevir", self.retranslate_sel),
            ("⟳ Sayfayı yeniden tara ve çevir", self.reanalyze),
            ("📑 PDF'i güncelle", self.make_pdf),
        ]:
            ttk.Button(acard, text=t, command=c).pack(fill="x", pady=2)

        self.show_orig = tk.BooleanVar()
        ttk.Checkbutton(acard, text="Orijinal sayfayı göster", variable=self.show_orig, command=self.redraw).pack(anchor="w", pady=(6, 0))

        self.info = ttk.Label(right, text="Kırmızı çerçeve: sığmayan metin", style="Small.TLabel", wraplength=310)
        self.info.pack(anchor="w", pady=(4, 0))

        theme.switchify(right)
        self.bind("<Control-s>", lambda e: self.apply())
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.pages.selection_set(0)
        self.load(0)

    def jpath(self, i=None): return self.wd / f"p{self.idx if i is None else i:04d}.json"
    def ipath(self): return self.wd / f"p{self.idx:04d}.jpg"

    def _on_canvas_resize(self, _=None):
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(70, self._do_delayed_resize)

    def _do_delayed_resize(self):
        self._resize_job = None
        self.redraw()

    def _run(self, job, done, msg="Çiziliyor…"):
        self.rendering = True
        self._tok += 1
        tok = self._tok
        self.info.config(text=msg)
        self.config(cursor="watch")
        res = queue.Queue()

        def work():
            _lower_thread_priority()
            try:
                res.put((True, job()))
            except Exception as e:
                res.put((False, e))
        threading.Thread(target=work, daemon=True).start()

        def chk():
            try:
                if not self.winfo_exists():
                    return
            except Exception:
                return
            try:
                ok, val = res.get_nowait()
            except queue.Empty:
                self.after(60, chk)
                return
            if tok != self._tok:
                return
            self.rendering = False
            self.config(cursor="")
            if not ok:
                self.info.config(text=f"Hata: {val}")
                return
            done(val)
        self.after(60, chk)

    def load(self, i):
        self.idx, self.sel = i, None
        try:
            an = json.loads(self.jpath().read_text(encoding="utf-8"))
        except Exception:
            an = {"summary": "", "new_terms": [], "regions": []}
        self.an = an
        self.fill()
        self.cv.delete("all")
        self._img_key = None
        self._box_ids = []
        self.cv.create_text(28, 28, anchor="nw", text="Sayfa yükleniyor…", fill=theme.MUTED, font=("Segoe UI", 12))
        side, cfg = self.cfg.get("render_long_side", 2200), self.cfg
        cached_jpg = self.ipath()

        def job():
            orig = source.load_page(self.ch, i, side)
            if cached_jpg.exists():
                try:
                    with Image.open(cached_jpg) as im:
                        out = im.convert("RGB")
                    return orig, out
                except Exception:
                    pass
            return orig, render_page(orig, an, cfg)

        def done(res):
            self.orig, self.out = res
            self._img_key = None
            self._snap = self._state()
            self.redraw()
            self.info.config(text="Kırmızı çerçeve: sığmayan metin")
        self._run(job, done, "Sayfa yükleniyor…")

    def _state(self):
        return json.dumps([(r.get("box"), r.get("clean_box"), r.get("src", ""), r.get("tr", ""), r.get("font", "comic"), bool(r.get("caps", True)),
                            bool(r.get("slant", False)), bool(r.get("skip", False)), round(float(r.get("size_mul", 1.0)), 2))
                           for r in self.an["regions"]], ensure_ascii=False)

    def dirty(self):
        self.commit_fields()
        return self._state() != self._snap

    def maybe_save(self):
        if not self.dirty():
            return True
        ans = messagebox.askyesnocancel("Kaydedilmemiş değişiklik", f"Sayfa {self.idx + 1} üzerindeki değişiklikler kaydedilsin mi?", parent=self)
        if ans is None:
            return False
        if ans:
            self.apply(sync=True)
        return True

    def on_page_select(self, _=None):
        s = self.pages.curselection()
        if not s or s[0] == self.idx:
            return
        if not self.maybe_save():
            self.pages.selection_clear(0, "end")
            self.pages.selection_set(self.idx)
            return
        self.load(s[0])

    def on_close(self):
        if self.maybe_save():
            self.destroy()

    def rerender(self, then=None):
        orig, an, cfg = self.orig, self.an, self.cfg

        def done(out):
            self.out = out
            self._img_key = None
            self.redraw()
            self.info.config(text="Kırmızı çerçeve: sığmayan metin")
            if then:
                then()
        self._run(lambda: render_page(orig, an, cfg), done)

    def redraw(self):
        if not hasattr(self, "out"):
            return
        im = self.orig if self.show_orig.get() else self.out
        cw, ch_ = max(200, self.cv.winfo_width()), max(200, self.cv.winfo_height())
        self.scale = min(cw / im.width, ch_ / im.height)
        key = (id(im), round(self.scale, 4), self.show_orig.get())
        if key != self._img_key:
            self._img_key = key
            self.tkim = ImageTk.PhotoImage(im.resize((max(1, int(im.width * self.scale)), max(1, int(im.height * self.scale))),
                                                     Image.BILINEAR))
            self.cv.delete("all")
            self.cv.create_image(0, 0, image=self.tkim, anchor="nw", tags="img")
            self._box_ids = []

        regions = self.an["regions"]
        sx = im.width / 1000.0 * self.scale
        sy = im.height / 1000.0 * self.scale

        if len(self._box_ids) != len(regions):
            self.cv.delete("box")
            self._box_ids = []
            for k, r in enumerate(regions):
                y0, x0, y1, x1 = r["box"]
                col = "#ff3030" if (k == self.sel or r.get("overflow")) else ("#888" if r.get("skip") else "#30c0ff")
                rid = self.cv.create_rectangle(x0 * sx, y0 * sy, x1 * sx, y1 * sy,
                                               outline=col, width=3 if k == self.sel else 1, tags="box")
                self._box_ids.append(rid)
        else:
            for k, (r, rid) in enumerate(zip(regions, self._box_ids)):
                y0, x0, y1, x1 = r["box"]
                col = "#ff3030" if (k == self.sel or r.get("overflow")) else ("#888" if r.get("skip") else "#30c0ff")
                self.cv.coords(rid, x0 * sx, y0 * sy, x1 * sx, y1 * sy)
                self.cv.itemconfigure(rid, outline=col, width=3 if k == self.sel else 1)

    def fill(self):
        r = self.an["regions"][self.sel] if self.sel is not None else None
        self.src.delete("1.0", "end")
        self.tr.delete("1.0", "end")
        if r:
            self.src.insert("end", r.get("src", ""))
            self.tr.insert("end", r["tr"])
            self.font.set(r.get("font", "comic"))
            self.caps.set(r.get("caps", True))
            self.slant.set(r.get("slant", False))
            self.skip.set(r.get("skip", False))
            self.mul.set(r.get("size_mul", 1.0))

    def commit_fields(self):
        if self.sel is None:
            return
        r = self.an["regions"][self.sel]
        r.update(src=self.src.get("1.0", "end").strip(), tr=self.tr.get("1.0", "end").strip(), font=self.font.get() or "comic", caps=self.caps.get(),
                 slant=self.slant.get(), skip=self.skip.get(), size_mul=round(self.mul.get(), 2))

    def hit(self, x, y):
        W, H = self.out.size
        for k in reversed(range(len(self.an["regions"]))):
            y0, x0, y1, x1 = self.an["regions"][k]["box"]
            if x0 * W / 1000 * self.scale <= x <= x1 * W / 1000 * self.scale and y0 * H / 1000 * self.scale <= y <= y1 * H / 1000 * self.scale:
                return k
        return None

    def down(self, e):
        if self.rendering or not hasattr(self, "out"):
            return
        W, H = self.out.size
        if self.new_mode:
            self.drag = ("new", e.x, e.y)
            return
        k = self.hit(e.x, e.y)
        self.commit_fields()
        self.sel = k
        self.fill()
        self.redraw()
        if k is not None:
            y0, x0, y1, x1 = self.an["regions"][k]["box"]
            corner = abs(e.x - x1 * W / 1000 * self.scale) < 14 and abs(e.y - y1 * H / 1000 * self.scale) < 14
            self.drag = ("size" if corner else "move", e.x, e.y, list(self.an["regions"][k]["box"]))

    def move(self, e):
        if not self.drag or self.rendering:
            return
        W, H = self.out.size
        if self.drag[0] == "new":
            return
        kind, sx, sy, box = self.drag
        dx, dy = (e.x - sx) / (W * self.scale) * 1000, (e.y - sy) / (H * self.scale) * 1000
        y0, x0, y1, x1 = box
        r = self.an["regions"][self.sel]
        r["box"] = [y0, x0, int(y1 + dy), int(x1 + dx)] if kind == "size" else [int(y0 + dy), int(x0 + dx), int(y1 + dy), int(x1 + dx)]
        if self.sel < len(self._box_ids):
            ny0, nx0, ny1, nx1 = r["box"]
            self.cv.coords(self._box_ids[self.sel],
                           nx0 * W / 1000.0 * self.scale, ny0 * H / 1000.0 * self.scale,
                           nx1 * W / 1000.0 * self.scale, ny1 * H / 1000.0 * self.scale)

    def up(self, e):
        if self.drag and self.drag[0] == "new":
            W, H = self.out.size
            _, sx, sy = self.drag
            f = lambda v, d: max(0, min(1000, int(v / (d * self.scale) * 1000)))
            y0, y1 = sorted([f(sy, H), f(e.y, H)])
            x0, x1 = sorted([f(sx, W), f(e.x, W)])
            if y1 - y0 > 8 and x1 - x0 > 8:
                self.an["regions"].append(dict(box=[y0, x0, y1, x1], clean_box=[y0, x0, y1, x1], kind="speech", font="comic",
                                               slant=False, caps=True, color="black", src="", tr="Metin", skip=False, size_mul=1.0))
                self.sel = len(self.an["regions"]) - 1
                self.fill()
            self.new_mode = False
            self.cv.config(cursor="crosshair")
        elif self.drag and self.sel is not None:
            r = self.an["regions"][self.sel]
            y0, x0, y1, x1 = r["box"]
            y0, y1 = sorted(max(0, min(1000, v)) for v in (y0, y1))
            x0, x1 = sorted(max(0, min(1000, v)) for v in (x0, x1))
            if y1 - y0 < 6:
                y0, y1 = max(0, min(y0, 994)), max(0, min(y0, 994)) + 6
            if x1 - x0 < 6:
                x0, x1 = max(0, min(x0, 994)), max(0, min(x0, 994)) + 6
            kind, old = self.drag[0], self.drag[3]
            if [y0, x0, y1, x1] != list(old):          # yalnızca tıklandıysa hiçbir şey değişmez
                r["box"] = [y0, x0, y1, x1]
                r.pop("bubble", None)
                if self.cfg.get("text_snap", True) and hasattr(self, "orig"):
                    # Kutuları hizala: silme alanını kutunun altındaki gerçek yazıya oturt;
                    # taşımada kutunun kendisi de yazıya hizalanır (boyutlandırmada kullanıcının boyutu korunur)
                    import numpy as np
                    from .clean import snap_box
                    nb = snap_box(np.asarray(self.orig.convert("RGB")), r["box"])
                    if kind == "move":
                        r["box"] = list(nb)
                    r["clean_box"] = list(nb)
                elif kind == "move" and r.get("clean_box"):
                    dy, dx = y0 - old[0], x0 - old[1]   # silme alanı kutuyla birlikte taşınır
                    cb = r["clean_box"]
                    r["clean_box"] = [max(0, min(1000, cb[0] + dy)), max(0, min(1000, cb[1] + dx)),
                                      max(0, min(1000, cb[2] + dy)), max(0, min(1000, cb[3] + dx))]
        self.drag = None
        self.redraw()

    def apply(self, sync=False):
        if self.rendering and not sync:
            return
        self.commit_fields()
        if sync:
            self.out = render_page(self.orig, self.an, self.cfg)
            self._img_key = None
            self.save()
        else:
            self.rerender(then=self.save)

    def clean_eq(self):
        if self.sel is not None:
            r = self.an["regions"][self.sel]
            r["clean_box"] = list(r["box"])
            self.apply()

    def arm_new(self):
        self.new_mode = True
        self.cv.config(cursor="plus")
        self.info.config(text="Sayfada bir dikdörtgen çizin.")

    def delete(self):
        if self.sel is not None and not self.rendering:
            del self.an["regions"][self.sel]
            self.sel = None
            self.fill()
            self.rerender(then=self.save)

    def save(self):
        if not self.an["regions"] and not self.jpath().exists():
            self.info.config(text="Bu sayfa henüz çevrilmedi; boş sayfa kaydedilmedi (çeviride işlenecek).")
            self._snap = self._state()
            return
        pipeline.save_json(self.jpath(), self.an)
        pipeline.save_jpg(self.ipath(), self.out, int(self.cfg.get("jpeg_quality", 90)))
        n = sum(1 for r in self.an["regions"] if r.get("overflow"))
        self.info.config(text=f"Kaydedildi. Sığmayan kutu: {n}" if n else "Kaydedildi.")
        self._snap = self._state()

    def _pipeline(self, key):
        return pipeline.Pipeline(self.p, self.cfg, key, threading.Event(), log=lambda m: None)

    def _ctx(self, target):
        ctx = []
        for j in range(max(0, target - 3), target):
            try:
                s_ = json.loads(self.jpath(j).read_text(encoding="utf-8")).get("summary")
                if s_:
                    ctx.append(s_)
            except Exception:
                pass
        return ctx

    def retranslate_sel(self):
        """Yalnızca seçili kutuyu (düzeltilmiş orijinal metniyle) yeniden çevirir; tek, küçük bir istek."""
        if self.rendering:
            return
        if self.sel is None:
            return self.info.config(text="Önce bir kutu seçin.")
        key = S.get_api_key()
        if not key:
            return messagebox.showwarning("API", "API anahtarı yok.", parent=self)
        self.commit_fields()
        r = self.an["regions"][self.sel]
        if not r.get("src", "").strip():
            return self.info.config(text="Bu kutunun orijinal metni boş; önce 'Orijinal' alanına metni yazın.")
        target, k = self.idx, self.sel
        reg = dict(r, tr="", skip=False)
        ctx = self._ctx(target)

        def job():
            _, miss, _ = self._pipeline(key).translate_regions([(target, [reg])], ctx)
            if miss:
                raise RuntimeError("Gemini bu metni çevirmedi; tekrar deneyin.")
            return reg["tr"], reg["skip"]

        def done(res):
            if self.idx != target or k >= len(self.an["regions"]):
                return
            rr = self.an["regions"][k]
            rr["tr"], rr["skip"] = res
            rr.pop("missing", None)
            self.sel = k
            self.fill()
            self.apply()
        self._run(job, done, "Gemini çeviriyor…")

    def reanalyze(self):
        """Sayfayı yerel OCR ile baştan tarar ve metinlerini çevirir."""
        if self.rendering:
            return
        key = S.get_api_key()
        if not key:
            return messagebox.showwarning("API", "API anahtarı yok.", parent=self)
        if not messagebox.askyesno("Onay", "Sayfa baştan taranıp çevrilecek; bu sayfadaki elle yaptığınız düzeltmeler kaybolur. Devam?", parent=self):
            return
        target, orig, cfg = self.idx, self.orig, self.cfg
        ctx = self._ctx(target)
        jpath, ipath = self.jpath(target), self.wd / f"p{target:04d}.jpg"

        def job():
            regs = ocr.detect(orig, cfg.get("tesseract_path", ""), cfg.get("rtl", True), cfg.get("ocr_engine", "auto"))
            if cfg.get("text_snap", True):
                pipeline.snap_regions(orig, regs)
            an = dict(summary="", new_terms=[], regions=regs, eco=True)
            _, miss, summ = self._pipeline(key).translate_regions([(target, regs)], ctx)
            an["summary"] = summ
            out = render_page(orig, an, cfg)
            pipeline.save_json(jpath, an)                  # sayfa değişse bile sonuç kaybolmaz
            pipeline.save_jpg(ipath, out, int(cfg.get("jpeg_quality", 90)))
            return an, out, miss

        def done(res):
            an, out, miss = res
            if self.idx == target:
                self.an, self.out, self.sel = an, out, None
                self._img_key = None
                self._snap = self._state()
                self.fill()
                self.redraw()
            self.info.config(text=f"Sayfa {target + 1} yeniden tarandı ve çevrildi ({len(an['regions'])} kutu"
                                  + (f", {miss} çevrilemedi" if miss else "") + ").")
        self._run(job, done, "Sayfa taranıyor ve çevriliyor…")

    def make_pdf(self):
        if self.rendering:
            return
        if self.dirty():
            return self.rerender(then=lambda: (self.save(), self.make_pdf()))
        self.info.config(text="PDF hazırlanıyor…")
        p, cid, cfg = self.p, self.ch["id"], self.cfg
        self._run(lambda: export.export_all(p, p.chapter(cid), cfg),
                  lambda out: self.info.config(text=f"PDF güncellendi: {out.name}"), "PDF hazırlanıyor…")


class ProofreadDialog(tk.Toplevel):
    """Bölümün tüm metinleri tek tabloda: orijinali ve Türkçeyi gözden geçir, düzelt, toplu bul/değiştir."""
    def __init__(self, view, project, ch):
        super().__init__(view)
        self.title(f"Metinleri Gözden Geçir — {ch['name']}")
        self.geometry("1060x720")
        self.view, self.p, self.ch = view, project, ch
        self.wd = project.work_dir(ch["id"])
        self.pages, self.dirty, self.cur = {}, set(), None
        for i in range(ch["n_pages"]):
            jp = self.wd / f"p{i:04d}.json"
            if jp.exists():
                try:
                    self.pages[i] = json.loads(jp.read_text(encoding="utf-8"))
                except Exception:
                    pass

        wrap = ttk.Frame(self, padding=(18, 14))
        wrap.pack(fill="both", expand=True)

        top = ttk.Frame(wrap, style="Card.TFrame", padding=(12, 10))
        top.pack(fill="x", pady=(0, 10))
        ttk.Label(top, text="Bul:", background=_SURF).pack(side="left")
        self.find = tk.StringVar()
        ttk.Entry(top, textvariable=self.find, width=22).pack(side="left", padx=(6, 12))
        ttk.Label(top, text="Değiştir:", background=_SURF).pack(side="left")
        self.repl = tk.StringVar()
        ttk.Entry(top, textvariable=self.repl, width=22).pack(side="left", padx=(6, 8))
        ttk.Button(top, text="Türkçede Tümünü Değiştir", command=self.replace_all).pack(side="left")
        self.only_issues = tk.BooleanVar()
        ttk.Checkbutton(top, text="Yalnızca sorunlular (taşan / boş / çevrilemeyen)", variable=self.only_issues, command=self.fill).pack(side="right")

        tcard = ttk.Frame(wrap, style="Card.TFrame", padding=1)
        tcard.pack(fill="both", expand=True)
        self.tv = ttk.Treeview(tcard, columns=("s", "src", "tr", "d"), show="headings", selectmode="browse")
        _stripe_tree(self.tv)
        for c, t, w in [("s", "Sayfa", 60), ("src", "Orijinal", 390), ("tr", "Türkçe", 430), ("d", "Durum", 100)]:
            self.tv.heading(c, text=t)
            self.tv.column(c, width=w, anchor="w")
        sb = ttk.Scrollbar(tcard, orient="vertical", command=self.tv.yview)
        self.tv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tv.pack(fill="both", expand=True)
        self.tv.bind("<<TreeviewSelect>>", self.pick)

        ed = ttk.Frame(wrap, style="Card.TFrame", padding=(14, 12))
        ed.pack(fill="x", pady=(10, 0))
        ttk.Label(ed, text="Orijinal:", background=_SURF).grid(row=0, column=0, sticky="nw", padx=(0, 8))
        self.src = tk.Text(ed, height=2, width=60, wrap="word")
        self.src.grid(row=0, column=1, sticky="we", pady=(0, 6))
        ttk.Label(ed, text="Türkçe:", background=_SURF).grid(row=1, column=0, sticky="nw", padx=(0, 8))
        self.tr = tk.Text(ed, height=3, width=60, wrap="word")
        self.tr.grid(row=1, column=1, sticky="we", pady=(0, 6))
        ed.columnconfigure(1, weight=1)
        self.skip = tk.BooleanVar()
        ttk.Checkbutton(ed, text="Orijinal kalsın (çevirme)", variable=self.skip).grid(row=2, column=1, sticky="w")

        bar = ttk.Frame(wrap)
        bar.pack(fill="x", pady=(10, 0))
        ttk.Button(bar, text="Satıra Uygula", command=self.commit).pack(side="left")
        self.info = ttk.Label(bar, text="Satırı seçin, düzeltin, Satıra Uygula'ya basın.", style="Muted.TLabel")
        self.info.pack(side="left", padx=12)
        ttk.Button(bar, text="Kaydet ve Yeniden Çiz (API Yok)", style="Accent.TButton", command=self.save_redraw).pack(side="right")
        ttk.Button(bar, text="Kaydet", command=self.save).pack(side="right", padx=(0, 8))

        theme.switchify(wrap)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.fill()

    def rows(self):
        for i in sorted(self.pages):
            for k, r in enumerate(self.pages[i].get("regions", [])):
                yield i, k, r

    @staticmethod
    def status(r):
        if r.get("missing") and not str(r.get("tr", "")).strip() and not r.get("skip"):
            return "ÇEVRİLEMEDİ"
        if r.get("skip"):
            return "çevrilmiyor"
        if not str(r.get("tr", "")).strip():
            return "BOŞ"
        if r.get("overflow"):
            return "SIĞMIYOR"
        return ""

    def fill(self):
        sel = self.tv.selection()
        self.tv.delete(*self.tv.get_children())
        row_idx = 0
        for i, k, r in self.rows():
            st = self.status(r)
            if self.only_issues.get() and st not in ("BOŞ", "SIĞMIYOR", "ÇEVRİLEMEDİ"):
                continue
            one = lambda t: " ".join(str(t).split())
            tag = "even" if row_idx % 2 == 0 else "odd"
            row_idx += 1
            self.tv.insert("", "end", iid=f"{i}:{k}", tags=(tag,), values=(i + 1, one(r.get("src", "")), one(r.get("tr", "")), st))
        for s_ in sel:
            if self.tv.exists(s_):
                self.tv.selection_set(s_)
                self.tv.see(s_)

    def pick(self, _=None):
        s_ = self.tv.selection()
        if not s_:
            return
        i, k = map(int, s_[0].split(":"))
        self.cur = (i, k)
        r = self.pages[i]["regions"][k]
        for w, v in ((self.src, r.get("src", "")), (self.tr, r.get("tr", ""))):
            w.delete("1.0", "end")
            w.insert("end", " ".join(str(v).split()))
        self.skip.set(bool(r.get("skip")))

    def commit(self):
        if not self.cur:
            return
        i, k = self.cur
        r = self.pages[i]["regions"][k]
        new = dict(src=self.src.get("1.0", "end").strip(), tr=self.tr.get("1.0", "end").strip(), skip=self.skip.get())
        norm = lambda v: " ".join(str(v).split()) if isinstance(v, str) else bool(v)
        if any(norm(r.get(key, "" if key != "skip" else False)) != norm(v) for key, v in new.items()):
            r.update(new)
            r.pop("overflow", None)
            self.dirty.add(i)
            self.fill()
            self.info.config(text=f"{len(self.dirty)} sayfada kaydedilmemiş değişiklik var.")

    def replace_all(self):
        a, b = self.find.get(), self.repl.get()
        if not a:
            return
        n = 0
        for i, k, r in self.rows():
            t = str(r.get("tr", ""))
            if a in t:
                r["tr"] = t.replace(a, b)
                n += 1
                self.dirty.add(i)
        self.fill()
        self.pick()
        self.info.config(text=f"{n} metinde değiştirildi. Kaydetmeyi unutmayın.")

    def save(self):
        self.commit()
        for i in sorted(self.dirty):
            pipeline.save_json(self.wd / f"p{i:04d}.json", self.pages[i])
        n = len(self.dirty)
        self.dirty.clear()
        self.info.config(text=f"{n} sayfa kaydedildi." + (" Görselleri güncellemek için 'yeniden çiz'." if n else ""))
        return n

    def save_redraw(self):
        self.save()
        self.destroy()
        self.view.redraw_selected([self.ch["id"]])

    def close(self):
        self.commit()
        if self.dirty:
            ans = messagebox.askyesnocancel("Kaydedilmemiş değişiklik", "Değişiklikler kaydedilsin mi?", parent=self)
            if ans is None:
                return
            if ans:
                self.save()
        self.destroy()

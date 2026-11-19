"""Proje başına SQLite kaydı: bölümler, sözlük, ayarlar."""
import json, re, shutil, sqlite3, threading, time
from pathlib import Path
from . import settings as S


def slug(name):
    s = re.sub(r"[^\w\- ]+", "", name, flags=re.U).strip().replace(" ", "_")
    return s or f"proje_{int(time.time())}"


def list_projects():
    out = []
    if S.PROJECTS_DIR.exists():
        for d in sorted(S.PROJECTS_DIR.iterdir()):
            if (d / "project.db").exists():
                try:
                    out.append(Project(d))
                except Exception:
                    pass
    return out


def create_project(name):
    S.ensure_dirs()
    d = S.PROJECTS_DIR / slug(name)
    i = 2
    while d.exists():
        d = S.PROJECTS_DIR / f"{slug(name)}_{i}"
        i += 1
    d.mkdir(parents=True)
    p = Project(d)
    p.set_meta("name", name)
    return p


class Project:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path / "project.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS chapters(
            id INTEGER PRIMARY KEY, src TEXT UNIQUE, kind TEXT, name TEXT,
            auto_num REAL, conf TEXT, manual_num REAL, n_pages INT DEFAULT 0, sig TEXT,
            status TEXT DEFAULT 'bekliyor', done INT DEFAULT 0, failed INT DEFAULT 0, out_pdf TEXT);
        CREATE TABLE IF NOT EXISTS glossary(src TEXT PRIMARY KEY, tr TEXT, auto INT DEFAULT 0);
        """)
        (self.path / "work").mkdir(exist_ok=True)
        (self.path / "out").mkdir(exist_ok=True)

    # ---- meta
    @property
    def name(self):
        return self.get_meta("name", self.path.name)

    def get_meta(self, k, d=None):
        with self.lock:
            r = self.db.execute("SELECT value FROM meta WHERE key=?", (k,)).fetchone()
        return json.loads(r[0]) if r else d

    def set_meta(self, k, v):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, json.dumps(v, ensure_ascii=False)))
            self.db.commit()

    # ---- chapters
    def upsert_chapter(self, src, kind, name, auto_num, conf, n_pages, sig):
        with self.lock:
            r = self.db.execute("SELECT id, sig FROM chapters WHERE src=?", (src,)).fetchone()
            if r:  # manuel düzeltme ve durum korunur; kaynak değiştiyse durum ve çeviri önbelleği sıfırlanır
                changed = r["sig"] != sig
                reset = ", status='bekliyor', done=0, failed=0, out_pdf=NULL" if changed else ""
                if changed:   # eski kaynağın çevrilmiş sayfaları yeni kaynakla karışmasın
                    shutil.rmtree(self.path / "work" / str(r["id"]), ignore_errors=True)
                self.db.execute(f"UPDATE chapters SET kind=?,name=?,auto_num=?,conf=?,n_pages=?,sig=?{reset} WHERE id=?",
                                (kind, name, auto_num, conf, n_pages, sig, r["id"]))
                cid = r["id"]
            else:
                cur = self.db.execute("INSERT INTO chapters(src,kind,name,auto_num,conf,n_pages,sig) VALUES(?,?,?,?,?,?,?)",
                                      (src, kind, name, auto_num, conf, n_pages, sig))
                cid = cur.lastrowid
            self.db.commit()
            return cid

    def chapters(self):
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT * FROM chapters")]

    def chapter(self, cid):
        with self.lock:
            r = self.db.execute("SELECT * FROM chapters WHERE id=?", (cid,)).fetchone()
        return dict(r) if r else None

    def update_chapter(self, cid, **kw):
        if not kw:
            return
        with self.lock:
            self.db.execute("UPDATE chapters SET " + ",".join(f"{k}=?" for k in kw) + " WHERE id=?",
                            (*kw.values(), cid))
            self.db.commit()

    def work_dir(self, cid):
        d = self.path / "work" / str(cid)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- glossary
    def glossary(self):
        with self.lock:
            return [(r["src"], r["tr"], r["auto"]) for r in self.db.execute("SELECT * FROM glossary ORDER BY src")]

    def set_term(self, src, tr, auto=0):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO glossary VALUES(?,?,?)", (src.strip(), tr.strip(), auto))
            self.db.commit()

    def add_auto_term(self, src, tr):
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO glossary VALUES(?,?,1)", (src.strip(), tr.strip()))
            self.db.commit()

    def del_term(self, src):
        with self.lock:
            self.db.execute("DELETE FROM glossary WHERE src=?", (src,))
            self.db.commit()

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

KINDS = {"document", "report", "meeting_minutes", "event", "policy", "note"}
STOPWORDS = set("le la les un une des du de d l et ou en au aux a à est sont ce cet cette ces que qui quoi dans sur pour par avec sans il elle ils elles on nous vous je tu y ne pas se sa son ses leur leurs mon ma mes the of and to in".split())
TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,39}$")

@dataclass
class User:
    id: str
    groups: set = field(default_factory=set)
    roles: set = field(default_factory=set)   # "editor" peut ajouter, "admin" peut tout supprimer et lire l'audit

    def visible_groups(self):
        return set(self.groups) | {"*"}       # "*" = toute l'entreprise

class PermissionDenied(Exception):
    pass

def chunk_text(text, max_chars=900):
    """Découpe par paragraphes, regroupés jusqu'à `max_chars` ; un paragraphe plus long est coupé."""
    chunks, buf = [], ""
    for para in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
        while len(para) > max_chars:
            cut = para.rfind(". ", 0, max_chars) + 1 or max_chars
            if buf:
                chunks.append(buf); buf = ""
            chunks.append(para[:cut].strip()); para = para[cut:].strip()
        if buf and len(buf) + len(para) + 2 > max_chars:
            chunks.append(buf); buf = ""
        buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        chunks.append(buf)
    return chunks

def fts_query(question):
    """Transforme une question libre en requête FTS5 sûre : mots entre guillemets, reliés par OR.
    Aucun opérateur FTS (NEAR, NOT, *, :) fourni par l'utilisateur ne passe."""
    words = [w for w in re.findall(r"\w+", question.lower()) if w not in STOPWORDS and len(w) > 1]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words))

class TenantStore:
    """Mémoire d'une entreprise. Une base SQLite par entreprise : l'isolation entre clients est physique,
    pas seulement un filtre dans une requête."""

    def __init__(self, root, tenant):
        if not TENANT_RE.match(tenant):
            raise ValueError("identifiant d'entreprise invalide (a-z, 0-9, - et _, 2 à 40 caractères)")
        Path(root).mkdir(parents=True, exist_ok=True)
        self.tenant = tenant
        self.db = sqlite3.connect(Path(root) / f"{tenant}.db")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA secure_delete = ON")   # écrase les pages libérées ; l'index FTS garde des segments jusqu'à sa fusion
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS documents(
                id INTEGER PRIMARY KEY, kind TEXT, title TEXT, author TEXT, date TEXT,
                starts_at TEXT, ends_at TEXT, meta TEXT, created_at REAL, deleted INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS acl(doc_id INTEGER, grp TEXT);
            CREATE INDEX IF NOT EXISTS acl_idx ON acl(grp, doc_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                text, doc_id UNINDEXED, idx UNINDEXED, tokenize='unicode61 remove_diacritics 2');
            CREATE TABLE IF NOT EXISTS audit(ts REAL, user TEXT, action TEXT, detail TEXT);
        """)

    def _log(self, user, action, detail=""):
        self.db.execute("INSERT INTO audit VALUES (?,?,?,?)", (time.time(), user.id, action, detail))

    def add_document(self, user, kind, title, text, groups, date=None, starts_at=None, ends_at=None, meta=None):
        """`groups` est obligatoire : ["*"] pour toute l'entreprise, sinon la liste des groupes autorisés."""
        if not user.roles & {"editor", "admin"}:
            raise PermissionDenied("ajout réservé aux éditeurs")
        if kind not in KINDS:
            raise ValueError(f"type inconnu : {kind}")
        if not groups:
            raise ValueError("groups obligatoire (['*'] = toute l'entreprise)")
        if "*" not in groups and not (set(groups) & user.visible_groups()) and "admin" not in user.roles:
            raise PermissionDenied("on ne peut publier que vers un groupe dont on fait partie")
        cur = self.db.execute(
            "INSERT INTO documents(kind,title,author,date,starts_at,ends_at,meta,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (kind, title, user.id, date, starts_at, ends_at, json.dumps(meta or {}), time.time()))
        doc_id = cur.lastrowid
        self.db.executemany("INSERT INTO acl VALUES (?,?)", [(doc_id, g) for g in set(groups)])
        self.db.executemany("INSERT INTO chunks(text, doc_id, idx) VALUES (?,?,?)",
                            [(c, doc_id, i) for i, c in enumerate(chunk_text(f"{title}\n\n{text}"))])
        self._log(user, "add", f"{doc_id}:{kind}:{title}")
        self.db.commit()
        return doc_id

    def _visible(self, user):
        groups = sorted(user.visible_groups())
        marks = ",".join("?" * len(groups))
        return f"d.deleted = 0 AND EXISTS (SELECT 1 FROM acl a WHERE a.doc_id = d.id AND a.grp IN ({marks}))", groups

    def search(self, user, question, k=5, kind=None, since=None):
        q = fts_query(question)
        self._log(user, "search", question[:200])
        self.db.commit()
        if not q:
            return []
        cond, params = self._visible(user)
        extra = ""
        if kind:
            extra += " AND d.kind = ?"; params.append(kind)
        if since:
            extra += " AND COALESCE(d.date, d.starts_at) >= ?"; params.append(since)
        rows = self.db.execute(
            f"""SELECT d.id doc_id, d.title, d.kind, d.date, d.author, c.idx, c.text, bm25(chunks) score
                FROM chunks c JOIN documents d ON d.id = c.doc_id
                WHERE chunks MATCH ? AND {cond}{extra} ORDER BY score LIMIT ?""",
            [q] + params + [k]).fetchall()
        return [dict(r) for r in rows]

    def list_events(self, user, start, end):
        """Requête structurée (pas de recherche par sens) : 'quelles réunions cette semaine ?'."""
        cond, params = self._visible(user)
        rows = self.db.execute(
            f"""SELECT d.id doc_id, d.title, d.starts_at, d.ends_at, d.meta FROM documents d
                WHERE d.kind = 'event' AND d.starts_at >= ? AND d.starts_at < ? AND {cond}
                ORDER BY d.starts_at""", [start, end] + params).fetchall()
        self._log(user, "list_events", f"{start}..{end}")
        self.db.commit()
        return [dict(r) for r in rows]

    def get_document(self, user, doc_id):
        cond, params = self._visible(user)
        row = self.db.execute(f"SELECT d.* FROM documents d WHERE d.id = ? AND {cond}", [doc_id] + params).fetchone()
        if row is None:
            return None  # inexistant ou non autorisé : même réponse, on ne révèle pas l'existence
        text = "\n\n".join(r["text"] for r in self.db.execute(
            "SELECT text FROM chunks WHERE doc_id = ? ORDER BY idx", (doc_id,)))
        self._log(user, "read", str(doc_id)); self.db.commit()
        return {**dict(row), "text": text}

    def delete_document(self, user, doc_id):
        """Supprime le texte lui-même (droit à l'effacement), pas seulement un drapeau."""
        owner = self.db.execute("SELECT author FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if owner is None or not ("admin" in user.roles or owner["author"] == user.id):
            raise PermissionDenied("suppression réservée à l'auteur ou à un administrateur")
        self.db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self.db.execute("INSERT INTO chunks(chunks) VALUES ('optimize')")   # sans cela le texte reste lisible dans le fichier
        self.db.execute("UPDATE documents SET deleted = 1, title = '[supprimé]', meta = '{}' WHERE id = ?", (doc_id,))
        self._log(user, "delete", str(doc_id)); self.db.commit()

    def audit_log(self, user, limit=100):
        if "admin" not in user.roles:
            raise PermissionDenied("journal réservé aux administrateurs")
        return [dict(r) for r in self.db.execute("SELECT * FROM audit ORDER BY ts DESC LIMIT ?", (limit,))]

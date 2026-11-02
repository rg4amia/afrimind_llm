"""Interface web locale et provisoire : chat sur le modèle GGUF + gestion des documents du RAG.
  python -m webui.server --model models/bekami-rag-agent-Q4_K_M.gguf     # puis http://127.0.0.1:8765
  python -m webui.server --mock                                           # sans modèle, pour essayer l'interface
Pas d'authentification : les « utilisateurs » ci-dessous servent à montrer les permissions, pas à les garantir.
N'écoute que sur 127.0.0.1. Ne pas exposer tel quel sur un réseau."""
import argparse
import base64
import binascii
import dataclasses
import json
import re
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from bekami.router import load_domains
from enterprise.agent import Agent
from enterprise.convert import to_text
from enterprise.policy import Policy
from enterprise.store import KINDS, PermissionDenied, TenantStore, User

MAX_BODY = 40_000_000   # un fichier de 25 Mo grossit d'un tiers en base64
MAX_TEXT = 1_000_000
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
INDEX = Path(__file__).with_name("index.html")

DEFAULT_USERS = {
    "admin": dict(label="Admin (éditeur, groupes direction + tech)", groups={"direction", "tech"}, roles={"admin", "editor"}),
    "awa": dict(label="Awa (groupe tech, lecture)", groups={"tech"}, roles=set()),
    "invite": dict(label="Invité (aucun groupe)", groups=set(), roles=set()),
}

def mock_chat(messages):
    """Moteur de simulation : cherche dans les documents puis répond en citant la première source."""
    last = messages[-1][1]
    if last.startswith("<tool_result>"):
        return "(simulation) Réponse fondée sur la première source [1]." if "[1]" in last else "Les sources ne contiennent pas cette information."
    return '<tool_call>' + json.dumps({"name": "search_documents", "arguments": {"query": last, "k": 3}}, ensure_ascii=False) + '</tool_call>'

class App:
    def __init__(self, db_dir, tenant, policy, chat, users=None):
        self.store = TenantStore(db_dir, tenant)
        # L'IA ne crée pas de documents d'elle-même ici : l'ajout passe uniquement par le panneau de l'interface.
        safe = dataclasses.replace(policy, tools=[t for t in policy.tools if t != "add_document"])
        safe._topics = policy._topics
        self.agent = Agent(self.store, safe, load_domains(), chat)
        self.users = users or DEFAULT_USERS

    def user(self, name):
        u = self.users.get(name)
        if u is None:
            raise ValueError("utilisateur inconnu")
        return User(name, set(u["groups"]), set(u["roles"]))

def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = "BekamiUI"

        def log_message(self, fmt, *args):
            pass

        def _send(self, status, payload):
            """`bytes` = la page HTML ; tout le reste est du JSON."""
            is_page = isinstance(payload, bytes)
            body = payload if is_page else json.dumps(payload, ensure_ascii=False).encode()
            ctype = "text/html; charset=utf-8" if is_page else "application/json; charset=utf-8"
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(body)

        def _host_ok(self):
            host = re.sub(r":\d+$", "", self.headers.get("Host", ""))
            return host in ALLOWED_HOSTS   # refuse les noms de domaine étrangers (DNS rebinding)

        def _dispatch(self, method):
            try:
                if not self._host_ok():
                    return self._send(403, {"error": "hôte non autorisé"})
                url = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(url.query).items()}
                body = {}
                if method == "POST":
                    if not self.headers.get("Content-Type", "").startswith("application/json"):
                        return self._send(415, {"error": "Content-Type application/json requis"})
                    n = int(self.headers.get("Content-Length") or 0)
                    if n > MAX_BODY:
                        return self._send(413, {"error": "requête trop volumineuse"})
                    body = json.loads(self.rfile.read(n) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError("objet JSON attendu")
                status, payload = self.route(method, url.path, q, body)
                self._send(status, payload)
            except PermissionDenied as e:
                self._send(403, {"error": str(e)})
            except (ValueError, KeyError, TypeError) as e:
                self._send(400, {"error": str(e) or "requête invalide"})
            except Exception:
                traceback.print_exc()
                self._send(500, {"error": "erreur interne (voir la console du serveur)"})

        do_GET = lambda self: self._dispatch("GET")
        do_POST = lambda self: self._dispatch("POST")

        def route(self, method, path, q, body):
            if method == "GET" and path == "/":
                return 200, INDEX.read_bytes()
            if method == "GET" and path == "/api/users":
                return 200, [dict(id=k, label=v["label"], editor=bool(v["roles"] & {"editor", "admin"}), groups=sorted(v["groups"]))
                             for k, v in app.users.items()]
            if method == "GET" and path == "/api/docs":
                return 200, app.store.list_documents(app.user(q.get("user", "")))
            if method == "GET" and path == "/api/doc":
                doc = app.store.get_document(app.user(q.get("user", "")), int(q["id"]))
                return (200, doc) if doc else (404, {"error": "document introuvable"})
            if method == "POST" and path == "/api/docs":
                user = app.user(body.get("user", ""))
                title, text = str(body.get("title", "")).strip(), str(body.get("text", "")).strip()
                if not title or not text:
                    raise ValueError("titre et texte obligatoires")
                if len(text) > MAX_TEXT:
                    raise ValueError(f"texte trop long (max {MAX_TEXT} caractères)")
                groups = [str(g) for g in body.get("groups") or ["*"]]
                kind = body.get("kind") or "document"
                if kind not in KINDS:
                    raise ValueError(f"type inconnu : {kind}")
                doc_id = app.store.add_document(user, kind, title[:200], text, groups, date=body.get("date") or None)
                return 201, {"id": doc_id}
            if method == "POST" and path == "/api/convert":
                user = app.user(body.get("user", ""))
                if not user.roles & {"editor", "admin"}:
                    raise PermissionDenied("conversion réservée aux éditeurs")
                try:
                    data = base64.b64decode(str(body.get("data", "")), validate=True)
                except (binascii.Error, ValueError):
                    raise ValueError("contenu du fichier invalide")
                converted = to_text(str(body.get("filename", "")), data)
                if converted["chars"] > MAX_TEXT:
                    raise ValueError(f"le texte extrait est trop long ({converted['chars']:,} caractères, maximum {MAX_TEXT:,}) : découpez le fichier en plusieurs parties".replace(",", " "))
                return 200, converted
            if method == "POST" and path == "/api/docs/delete":
                app.store.delete_document(app.user(body.get("user", "")), int(body["id"]))
                return 200, {"ok": True}
            if method == "POST" and path == "/api/chat":
                question = str(body.get("question", "")).strip()
                if not question:
                    raise ValueError("question vide")
                return 200, app.agent.run(app.user(body.get("user", "")), question[:2000])
            return 404, {"error": "introuvable"}

    return Handler

def serve(app, host="127.0.0.1", port=8765):
    return HTTPServer((host, port), make_handler(app))   # un seul fil : le modèle et SQLite ne sont pas partagés entre threads

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/bekami-rag-agent-Q4_K_M.gguf")
    ap.add_argument("--mock", action="store_true", help="moteur de simulation, sans modèle")
    ap.add_argument("--tenant-file", default="tenants/example/tenant.yaml")
    ap.add_argument("--db", default="data/webui")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-tokens", type=int, default=300)
    args = ap.parse_args()
    if args.mock:
        chat = mock_chat
    else:
        from enterprise.cpu_backend import CPUChat
        cpu = CPUChat(args.model, n_ctx=4096, max_tokens=args.max_tokens, temperature=0.0)
        chat = cpu
    app = App(args.db, "webui", Policy.load(args.tenant_file), chat)
    srv = serve(app, port=args.port)
    print(f"Interface : http://127.0.0.1:{args.port}  ({'simulation' if args.mock else args.model})  -  Ctrl+C pour arrêter")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass

"""python -m webui.test_server : API de l'interface (permissions, isolation, protections), moteur simulé, aucun réseau externe."""
import base64
import http.client
import io
import json
import zipfile
import tempfile
import threading

from enterprise.policy import Policy
from webui.server import App, make_handler, mock_chat, serve

def check(name, cond):
    print(("OK  " if cond else "ECHEC"), name)
    assert cond, name

calls = []
def spy_chat(messages):
    calls.append(messages)
    return mock_chat(messages)

# SQLite et le modèle ne sont pas partagés entre fils : l'application naît dans le fil qui sert les requêtes
box, ready = {}, threading.Event()
def run():
    box["app"] = App(tempfile.mkdtemp(), "webui", Policy.load("tenants/example/tenant.yaml"), spy_chat)
    box["srv"] = serve(box["app"], port=0)
    ready.set()
    box["srv"].serve_forever()
threading.Thread(target=run, daemon=True).start()
ready.wait(10)
app, srv = box["app"], box["srv"]
port = srv.server_address[1]

def call(method, path, body=None, host=None, ctype="application/json"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Host": host} if host else {}
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers["Content-Type"] = ctype
    c.request(method, path, data, headers)
    r = c.getresponse()
    raw = r.read()
    try:
        return r.status, json.loads(raw)
    except ValueError:
        return r.status, raw

s, page = call("GET", "/")
check("la page HTML est servie", s == 200 and b"<title>Bekami" in page)
s, users = call("GET", "/api/users")
check("trois utilisateurs, seul admin est éditeur", [u["editor"] for u in users] == [True, False, False])

s, r = call("POST", "/api/docs", dict(user="admin", title="Politique de congés", text="Les congés payés sont de 30 jours par an.", groups=["*"]))
check("un éditeur ajoute un document", s == 201 and r["id"] == 1)
s, r = call("POST", "/api/docs", dict(user="admin", title="Budget direction", text="Budget secret de 2 millions de FCFA.", groups=["direction"]))
check("document réservé à un groupe", s == 201)
s, r = call("POST", "/api/docs", dict(user="awa", title="x", text="y", groups=["*"]))
check("un non-éditeur ne peut pas ajouter (403)", s == 403)
s, r = call("POST", "/api/docs", dict(user="admin", title="", text="y"))
check("titre vide refusé (400)", s == 400)
s, r = call("POST", "/api/docs", dict(user="admin", title="t", text="y" * 1_000_001))
check("texte trop long refusé (400)", s == 400)
s, r = call("POST", "/api/docs", dict(user="admin", title="t", text="y", kind="inconnu"))
check("type inconnu refusé (400)", s == 400)

s, docs = call("GET", "/api/docs?user=admin")
check("admin voit 2 documents", [d["title"] for d in docs] == ["Budget direction", "Politique de congés"])
s, docs = call("GET", "/api/docs?user=awa")
check("awa ne voit que le document public", [d["title"] for d in docs] == ["Politique de congés"])
s, docs = call("GET", "/api/docs?user=invite")
check("l'invité voit le document public", len(docs) == 1)
s, r = call("GET", "/api/doc?user=awa&id=2")
check("document réservé : introuvable pour awa (404, pas 403)", s == 404)
s, r = call("GET", "/api/doc?user=admin&id=2")
check("le texte du document est lisible par son groupe", s == 200 and "2 millions" in r["text"])
s, r = call("GET", "/api/docs?user=inconnu")
check("utilisateur inconnu refusé (400)", s == 400)

s, r = call("POST", "/api/chat", dict(user="awa", question="Combien de jours de congés payés ?"))
check("chat : réponse appuyée sur la source citée", s == 200 and r["grounded"] and r["sources"][0]["title"] == "Politique de congés")
n = len(calls)
s, r = call("POST", "/api/chat", dict(user="awa", question="Quel est le budget secret ?"))
check("le document réservé n'est jamais transmis au modèle", all("2 millions" not in m[1] for msgs in calls[n:] for m in msgs))
s, r = call("POST", "/api/chat", dict(user="admin", question="Quel est le budget secret ?"))
check("le groupe autorisé l'obtient", r["grounded"] and r["sources"][0]["title"] == "Budget direction")
s, r = call("POST", "/api/chat", dict(user="awa", question="Qui va gagner la prochaine élection présidentielle ?"))
check("domaine interdit : refus sans appel au modèle", r["denied"] and r["answer"] == app.agent.policy.refusal_message)
s, r = call("POST", "/api/chat", dict(user="awa", question="   "))
check("question vide refusée (400)", s == 400)

app.agent.chat = lambda m: (_ for _ in ()).throw(RuntimeError("panne du moteur"))
s, r = call("POST", "/api/chat", dict(user="awa", question="Combien de jours de congés payés ?"))
check("panne du moteur : réponse 500 propre, le serveur reste vivant", s == 500 and call("GET", "/api/users")[0] == 200)
app.agent.chat = spy_chat

def b64(b):
    return base64.b64encode(b).decode()
W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    z.writestr("word/document.xml", f'<w:document {W}><w:body><w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Règlement</w:t></w:r></w:p>'
               f'<w:p><w:r><w:t>Les visiteurs doivent porter un badge visible.</w:t></w:r></w:p></w:body></w:document>')
s, r = call("POST", "/api/convert", dict(user="admin", filename="reglement-interieur.docx", data=b64(buf.getvalue())))
check("conversion d'un .docx : titre, Markdown et source", s == 200 and r["title"] == "reglement interieur" and r["text"].startswith("# Règlement") and r["source"] == "Word")
s, d = call("POST", "/api/docs", dict(user="admin", title=r["title"], text=r["text"], groups=["*"]))
s, r2 = call("POST", "/api/chat", dict(user="awa", question="Que doivent porter les visiteurs ?"))
check("le document converti est ensuite retrouvé par le chat", r2["grounded"] and r2["sources"][0]["title"] == "reglement interieur")
s, r = call("POST", "/api/convert", dict(user="admin", filename="long.txt", data=b64(b"a" * 1_000_001)))
check("texte converti trop long : refusé dès la conversion, message clair", s == 400 and "trop long" in r["error"] and "découpez" in r["error"])
s, r = call("POST", "/api/convert", dict(user="admin", filename="gros.txt", data=b64(b"mot " * 200_000)))
check("fichier de 800 Ko de texte : converti", s == 200 and r["chars"] > 700_000)
s, r = call("POST", "/api/convert", dict(user="awa", filename="a.txt", data=b64(b"x")))
check("conversion réservée aux éditeurs (403)", s == 403)
s, r = call("POST", "/api/convert", dict(user="admin", filename="a.txt", data="***pas du base64***"))
check("base64 invalide refusé (400)", s == 400)
s, r = call("POST", "/api/convert", dict(user="admin", filename="a.exe", data=b64(b"MZ")))
check("format non pris en charge : message clair (400)", s == 400 and "non pris en charge" in r["error"])
s, r = call("POST", "/api/convert", dict(user="admin", filename="a.docx", data=b64(b"pas un zip")))
check("docx corrompu : 400, pas de plantage", s == 400 and call("GET", "/api/users")[0] == 200)

add_call = '<tool_call>' + json.dumps({"name": "add_document", "arguments": {"kind": "note", "title": "pirate", "text": "x", "groups": ["*"]}}) + '</tool_call>'
replies = iter([add_call, "Fait."])
app.agent.chat = lambda m: next(replies)
before = len(call("GET", "/api/docs?user=admin")[1])
call("POST", "/api/chat", dict(user="admin", question="Ajoute une note"))
check("le modèle ne peut pas créer de document par lui-même, même pour un admin",
      len(call("GET", "/api/docs?user=admin")[1]) == before)
app.agent.chat = spy_chat

s, r = call("POST", "/api/docs/delete", dict(user="awa", id=1))
check("suppression par un non-auteur refusée (403)", s == 403)
s, r = call("POST", "/api/docs/delete", dict(user="admin", id=1))
s, docs = call("GET", "/api/docs?user=awa")
check("suppression par un admin : le document supprimé disparaît, les autres restent",
      s == 200 and [d["title"] for d in docs] == ["reglement interieur"])

s, r = call("GET", "/api/users", host="evil.example.com")
check("nom d'hôte étranger refusé (DNS rebinding)", s == 403)
s, r = call("POST", "/api/chat", b"question=x", ctype="application/x-www-form-urlencoded")
check("POST non JSON refusé (415)", s == 415)
s, r = call("POST", "/api/chat", b"{pas du json")
check("JSON invalide : 400, pas de plantage", s == 400)
s, r = call("GET", "/api/inconnue")
check("route inconnue : 404", s == 404)
srv.shutdown()
print("Tous les tests passent, sans aucun accès réseau externe.")

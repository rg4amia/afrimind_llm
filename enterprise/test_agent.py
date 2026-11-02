"""python -m enterprise.test_agent : boucle d'outils, permissions, politique, limites d'étapes."""
import dataclasses
import socket
import tempfile

from bekami.router import load_domains
from enterprise.agent import Agent
from enterprise.policy import Policy
from enterprise.store import TenantStore, User

def check(name, cond):
    print(("OK  " if cond else "ECHEC"), name)
    assert cond, name

def no_network(*a, **k):
    raise AssertionError("accès réseau tenté")
socket.socket.connect = no_network
socket.create_connection = no_network
socket.getaddrinfo = no_network

def scripted(*replies):
    """Moteur factice : renvoie les réponses dans l'ordre et garde les messages reçus."""
    seen, it = [], iter(replies)
    def chat(messages):
        seen.append(list(messages))
        return next(it)
    chat.seen = seen
    return chat

def call(name, **args):
    import json
    return '<tool_call>' + json.dumps({"name": name, "arguments": args}) + '</tool_call>'

domains = load_domains()
policy = Policy.load("tenants/example/tenant.yaml")
store = TenantStore(tempfile.mkdtemp(), "exemple")
admin = User("admin", {"direction"}, {"admin", "editor"})
emp = User("awa", {"tech"}, set())
store.add_document(admin, "policy", "Politique de congés", "Les congés payés sont de 30 jours par an.", ["*"], date="2026-01-01")
store.add_document(admin, "report", "Rapport direction", "Budget secret de 2 millions.", ["direction"])
store.add_document(admin, "event", "Réunion équipe", "Point hebdomadaire.", ["*"], starts_at="2026-03-02T09:00", ends_at="2026-03-02T10:00")

chat = scripted(call("search_documents", query="congés payés"), "Vous avez 30 jours par an [1].")
r = Agent(store, policy, domains, chat).run(emp, "Combien de jours de congés ai-je ?")
check("recherche puis réponse citée et ancrée", r["grounded"] and r["steps"] == 1 and r["sources"][0]["title"] == "Politique de congés")
check("le résultat d'outil revient au modèle", "30 jours" in chat.seen[1][-1][1] and chat.seen[1][-1][1].startswith("<tool_result>"))

chat = scripted(call("search_documents", query="budget secret"), "Rien trouvé.")
Agent(store, policy, domains, chat).run(emp, "Où en est le budget ?")
check("document réservé jamais transmis à l'outil", "2 millions" not in chat.seen[1][-1][1])

chat = scripted(call("list_events", start="2026-03-01", end="2026-03-08"), "Une réunion lundi.")
Agent(store, policy, domains, chat).run(emp, "Quelles réunions cette semaine ?")
check("list_events fonctionne", "Réunion équipe" in chat.seen[1][-1][1])

chat = scripted(call("add_document", kind="note", title="x", text="y", groups=["*"]), "Fait.")
Agent(store, policy, domains, chat).run(emp, "Note ceci")
check("add_document par un non-éditeur : refus transmis au modèle", chat.seen[1][-1][1].count("refusé") == 1)

limited = dataclasses.replace(policy, tools=["search_documents"]); limited._topics = policy._topics
chat = scripted(call("add_document", kind="note", title="x", text="y", groups=["*"]), "Ok.")
Agent(store, limited, domains, chat).run(admin, "Note ceci")
check("outil absent de policy.tools : indisponible même pour un admin", "outil non disponible" in chat.seen[1][-1][1])

chat = scripted(call("search_documents", query="projet orion"), "Ok.")
Agent(store, policy, domains, chat).run(emp, "Cherche dans les documents")
check("requête d'outil sur un sujet interdit refusée", "sujet interdit" in chat.seen[1][-1][1])

chat = scripted("<tool_call>{pas du json}</tool_call>", "Désolé.")
r = Agent(store, policy, domains, chat).run(emp, "Cherche dans les documents")
check("appel mal formé : observation d'erreur, pas de plantage", "mal formé" in chat.seen[1][-1][1] and r["denied"] is None)

loop = scripted(*[call("search_documents", query="congés")] * 10)
r = Agent(store, policy, domains, loop, max_steps=3).run(emp, "Combien de jours de congés ?")
check("boucle infinie coupée au nombre d'étapes", len(loop.seen) == 4 and "limite" in r["answer"])

chat = scripted("jamais appelé")
r = Agent(store, policy, domains, chat).run(emp, "Qui va gagner la prochaine élection présidentielle ?")
check("domaine interdit : refus sans appel au modèle", r["denied"] is not None and not chat.seen)

r = Agent(store, policy, domains, scripted("Le concurrent Alpha prépare une offre.")).run(emp, "Planning du département ?")
check("réponse contenant un sujet interdit : bloquée", r["denied"] == "réponse bloquée")

print("Tous les tests passent, sans aucun accès réseau.")

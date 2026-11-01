"""python -m enterprise.test_assistant : politique d'entreprise, assistant, et absence totale d'accès réseau."""
import dataclasses
import socket
import tempfile

from bekami.router import load_domains
from enterprise.assistant import Assistant, NOT_FOUND
from enterprise.policy import Policy
from enterprise.store import TenantStore, User

def check(name, cond):
    print(("OK  " if cond else "ECHEC"), name)
    assert cond, name

# --- aucune connexion réseau n'est possible pendant tout ce test ---
def no_network(*a, **k):
    raise AssertionError("accès réseau tenté")
socket.socket.connect = no_network
socket.create_connection = no_network
socket.getaddrinfo = no_network

domains = load_domains()
policy = Policy.load("tenants/example/tenant.yaml")
root = tempfile.mkdtemp()
store = TenantStore(root, "exemple")
admin = User("admin", {"direction"}, {"admin", "editor"})
emp = User("awa", {"tech"}, set())
store.add_document(admin, "policy", "Politique de congés", "Les congés payés sont de 30 jours par an. La demande se fait via le portail RH.", ["*"], date="2026-01-01")
store.add_document(admin, "report", "Rapport confidentiel direction", "Le projet Orion sera annoncé en juin.", ["direction"])

calls = []
def model(messages):
    calls.append(messages)
    return "Voici la réponse [1]."
bot = Assistant(store, policy, domains, model)

r = bot.ask(emp, "Combien de jours de congés payés par an ?")
check("question autorisée avec source : répondue et ancrée", r["grounded"] and r["sources"][0]["title"] == "Politique de congés" and r["denied"] is None)
check("le message système contient la consigne de l'entreprise", "Vouvoie" in calls[-1][0][1] and "Exemple SA" in calls[-1][0][1])
n = len(calls)
r = bot.ask(emp, "Qui va gagner la prochaine élection présidentielle ?")
check("domaine interdit (politique) : refus sans appel au modèle", r["denied"].startswith("domaine interdit") and len(calls) == n)
check("le refus utilise le message de l'entreprise", r["answer"] == policy.refusal_message)
r = bot.ask(emp, "Parle-moi du Projet Orion")
check("sujet interdit refusé même avec une casse différente", r["denied"] == "sujet interdit" and len(calls) == n)
r = bot.ask(emp, "Je suis triste et stressé, que faire ?")
check("psychologie interdite pour cette entreprise", r["denied"] is not None and len(calls) == n)
r = bot.ask(emp, "Que dit la Bible sur le travail et la prière ?")
check("domaine interdit détecté même en position secondaire", r["denied"] is not None)
r = bot.ask(emp, "Raconte-moi une blague")
check("sans source, mode général : répond en le signalant", r["denied"] is None and not r["grounded"] and "connaissances générales" in r["answer"])
# l'employé n'obtient pas le document de la direction
r = bot.ask(emp, "Quand sera annoncé le projet Orion en juin ?")
check("document réservé : refus (sujet interdit) ET jamais transmis", all("Rapport confidentiel" not in m[0][1] for m in calls))
# sujet interdit dans la RÉPONSE du modèle
leaky = Assistant(store, policy, domains, lambda m: "Le concurrent Alpha prépare une offre.")
r = leaky.ask(emp, "Quel est le planning du département ?")
check("réponse contenant un sujet interdit : bloquée", r["denied"] == "réponse bloquée" and "Alpha" not in r["answer"])
# code
nocode = dataclasses.replace(policy, code_enabled=False); nocode._topics = policy._topics
r = Assistant(store, nocode, domains, model).ask(emp, "Écris une fonction python pour trier une liste")
check("code désactivé : refusé", r["denied"] == "génération de code désactivée")
r = bot.ask(emp, "Écris une fonction python pour trier une liste")
check("code activé : autorisé et routé vers informatique", r["denied"] is None and r["domain"] == "informatique")
# mode documents seulement
only = dataclasses.replace(policy, mode="documents_only"); only._topics = policy._topics
n = len(calls)
r = Assistant(store, only, domains, model).ask(emp, "Raconte-moi une blague")
check("documents_only : rien trouvé -> pas d'appel au modèle", r["answer"] == NOT_FOUND and len(calls) == n)
r = Assistant(store, only, domains, model).ask(emp, "Combien de jours de congés payés ?")
check("documents_only : répond quand un document existe", r["grounded"])
# liste blanche de domaines
wl = dataclasses.replace(policy, allowed_domains=["informatique", "droit"], blocked_domains=[]); wl._topics = policy._topics
r = Assistant(store, wl, domains, model).ask(emp, "Quelle recette de kedjenou ?")
check("liste blanche : domaine hors liste refusé", r["denied"] and "non autorisé" in r["denied"])
check("liste blanche : domaine listé accepté", Assistant(store, wl, domains, model).ask(emp, "Explique le code du travail et le licenciement")["denied"] is None)
# audit des refus
log = store.audit_log(admin)
check("les refus sont tracés dans l'audit", sum(e["action"] == "denied" for e in log) >= 4)
# validation de la configuration
import yaml, pathlib
bad = pathlib.Path(root) / "bad.yaml"
bad.write_text(yaml.safe_dump({"tenant": "x", "company_name": "X", "tools": ["rm_rf"]}))
try:
    Policy.load(bad); check("outil inconnu dans la config", False)
except ValueError:
    check("config invalide (outil inconnu) rejetée au chargement", True)
print("\nTous les tests passent, sans aucun accès réseau.")

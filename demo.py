"""Essai de bout en bout : documents -> recherche avec permissions -> agent -> modèle GGUF sur CPU.
  python demo.py --docs examples/docs --ask "Combien de jours de congés ?"
  python demo.py --docs examples/docs                         # mode interactif
Un fichier à la racine du dossier est visible de toute l'entreprise ; un fichier dans un sous-dossier n'est visible
que du groupe portant le nom du sous-dossier (examples/docs/direction/ -> groupe « direction »)."""
import argparse, shutil
from pathlib import Path

from bekami.router import load_domains
from enterprise.agent import Agent
from enterprise.cpu_backend import CPUChat
from enterprise.policy import Policy
from enterprise.store import TenantStore, User

ap = argparse.ArgumentParser()
ap.add_argument("--docs", required=True, help="dossier de fichiers .txt/.md")
ap.add_argument("--model", default="models/bekami-rag-agent-Q4_K_M.gguf")
ap.add_argument("--tenant", default="tenants/example/tenant.yaml")
ap.add_argument("--groups", nargs="*", default=[], help="groupes de l'utilisateur qui pose les questions")
ap.add_argument("--db", default="data/demo_db", help="dossier de la base (recréée à chaque lancement)")
ap.add_argument("--ask", default=None, help="une seule question, sans mode interactif")
ap.add_argument("--max-tokens", type=int, default=300)
args = ap.parse_args()

policy = Policy.load(args.tenant)
shutil.rmtree(args.db, ignore_errors=True)
store = TenantStore(args.db, "demo")
admin = User("admin", {"admin"}, {"admin", "editor"})
root = Path(args.docs)
n = 0
for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in (".txt", ".md")):
    rel = path.relative_to(root)
    group = rel.parts[0] if len(rel.parts) > 1 else "*"
    store.add_document(admin, "document", path.stem.replace("-", " "), path.read_text(encoding="utf-8"), [group])
    n += 1
print(f"{n} documents chargés. Utilisateur : groupes {args.groups or '(aucun)'}")
if n == 0:
    raise SystemExit("Aucun document .txt/.md trouvé.")

user = User("demo", set(args.groups), set())
agent = Agent(store, policy, load_domains(), CPUChat(args.model, max_tokens=args.max_tokens, temperature=0.0))

def ask(q):
    r = agent.run(user, q)
    print(f"\n> {q}\n{r['answer']}")
    print(f"  [domaine={r['domain']} ancrée={r['grounded']} étapes={r['steps']} refus={r['denied']}]")
    for s in r["sources"]:
        print(f"  source [{s['n']}] {s['title']}")

if args.ask:
    ask(args.ask)
else:
    while True:
        try:
            q = input("\nQuestion (vide pour quitter) : ").strip()
        except EOFError:
            break
        if not q:
            break
        ask(q)

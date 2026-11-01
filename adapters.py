import argparse, yaml
from bekami.hf import ADAPTERS_DIR, list_adapters, merge_adapter

ap = argparse.ArgumentParser(description="Registre d'adaptateurs de domaine")
sub = ap.add_subparsers(dest="cmd", required=True)
sub.add_parser("list")
m = sub.add_parser("merge", help="fusionne un adaptateur dans son modèle de base")
m.add_argument("domain")
m.add_argument("--out", required=True)
args = ap.parse_args()

if args.cmd == "list":
    for r in list_adapters():
        state = "entraîné" if r["trained"] else "pas entraîné"
        print(f"{r['domain']:<20} {state:<13} {r['base_model']}  {r['eval'] or ''}")
else:
    base = yaml.safe_load((ADAPTERS_DIR / args.domain / "adapter.yaml").read_text())["base_model"]
    merge_adapter(base, ADAPTERS_DIR / args.domain / "weights", args.out)
    print("Modèle fusionné :", args.out)

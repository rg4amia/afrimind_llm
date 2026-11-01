import argparse, json
from bekami.evaluate import load_complete, load_problems, run_code_eval, run_passkey
from bekami.tokenizer import Tokenizer

ap = argparse.ArgumentParser(description="Évaluation de Bekami")
sub = ap.add_subparsers(dest="cmd", required=True)
for name in ("code", "passkey"):
    p = sub.add_parser(name)
    p.add_argument("--ckpt", default="checkpoints/bekami_final.pt")
    p.add_argument("--tokenizer", default="tokenizer/bekami.model")
code = sub.choices["code"]
code.add_argument("--problems", default="data/eval/smoke.jsonl", help="jsonl au format HumanEval")
code.add_argument("--canonical", action="store_true", help="vérifie le banc avec les solutions de référence")
code.add_argument("--out", default=None, help="écrit le détail par problème en jsonl")
pk = sub.choices["passkey"]
pk.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
pk.add_argument("--trials", type=int, default=3)
args = ap.parse_args()

if args.cmd == "code":
    complete = None if args.canonical else load_complete(args.ckpt, args.tokenizer)
    score, results = run_code_eval(load_problems(args.problems), complete, canonical=args.canonical)
    print(f"pass@1 = {score:.1%} ({sum(r['passed'] for r in results)}/{len(results)})")
    for r in results:
        if not r["passed"]:
            print("  échec", r["task_id"], "-", r["error"])
    if args.out:
        with open(args.out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in results)
else:
    complete = load_complete(args.ckpt, args.tokenizer)
    tok = Tokenizer(args.tokenizer)
    grid = run_passkey(complete, tok, args.lengths, trials=args.trials)
    depths = sorted({d for _, d in grid})
    print("longueur | " + " ".join(f"prof {d:.0%}" for d in depths))
    for n in args.lengths:
        print(f"{n:>8} | " + " ".join(f"{grid[(n, d)]:>8.0%}" for d in depths))

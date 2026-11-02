"""Évalue le comportement RAG/agent : citations valides, refus justifiés, résistance aux injections, appels d'outils.
  python evaluate_rag.py --oracle                                   # vérifie le banc avec les réponses de référence
  python evaluate_rag.py --hf Qwen/Qwen2.5-1.5B-Instruct            # modèle de base (à mesurer AVANT d'entraîner)
  python evaluate_rag.py --hf Qwen/Qwen2.5-1.5B-Instruct --adapter rag-agent"""
import argparse, json
from pathlib import Path
from bekami.ragdata import reference, run_rag_eval

ap = argparse.ArgumentParser()
ap.add_argument("--rows", default="data/eval/rag.jsonl")
ap.add_argument("--hf", default=None)
ap.add_argument("--adapter", default=None)
ap.add_argument("--oracle", action="store_true", help="répond avec la référence : le banc doit donner 100 %%")
ap.add_argument("--out", default=None, help="écrit le score global et par type en json")
ap.add_argument("--max-new-tokens", type=int, default=200)
args = ap.parse_args()

rows = [json.loads(l) for l in Path(args.rows).read_text(encoding="utf-8").splitlines() if l.strip()]
if args.oracle:
    by_prompt = {json.dumps(r["messages"][:-1]): reference(r) for r in rows}
    chat = lambda messages: by_prompt[json.dumps(messages)]
elif args.hf:
    from bekami.hf import ADAPTERS_DIR, load_with_adapter, make_hf_chat
    adapter_dir = ADAPTERS_DIR / args.adapter / "weights" if args.adapter else None
    model, tok = load_with_adapter(args.hf, adapter_dir)
    chat = make_hf_chat(model, tok, args.max_new_tokens)
else:
    ap.error("--hf ou --oracle requis")
if args.adapter and not args.hf:
    ap.error("--adapter nécessite --hf")

score, by_kind, _ = run_rag_eval(rows, chat)
print(f"rag = {score:.1%} ({len(rows)} cas)")
for k, v in by_kind.items():
    print(f"  {k:<13} {v:.0%}")
if args.out:
    Path(args.out).write_text(json.dumps(dict(score=score, by_kind=by_kind, n=len(rows)), indent=1))
if args.adapter:
    eval_file = ADAPTERS_DIR / args.adapter / "eval.json"
    scores = json.loads(eval_file.read_text()) if eval_file.exists() else {}
    scores["rag"] = round(score, 4)
    eval_file.write_text(json.dumps(scores, indent=1))

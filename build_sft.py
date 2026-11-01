"""Construit un jeu 'signature + docstring -> corps de la fonction' à partir de code Python.
Sources : data/raw/repos/<dépôt>/ (licences permissives uniquement) ou --stdlib (bibliothèque standard, PSF)."""
import argparse, ast, hashlib, json, random, sysconfig
from pathlib import Path
from bekami.data import Decontaminator, SKIP_DIRS, clean_file, is_validation

ap = argparse.ArgumentParser()
ap.add_argument("domain", help="écrit adapters/<domaine>/train.jsonl et val.jsonl")
ap.add_argument("--repos", default="data/raw/repos")
ap.add_argument("--stdlib", action="store_true", help="utiliser la bibliothèque standard Python comme source")
ap.add_argument("--max-examples", type=int, default=5000)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

root = Path(sysconfig.get_paths()["stdlib"]) if args.stdlib else Path(args.repos)
SKIP = SKIP_DIRS | {"test", "tests", "idlelib", "turtledemo", "lib2to3", "site-packages"}
decontam = Decontaminator()
if not decontam.hashes:
    raise SystemExit("data/decontam/ est vide : lance d'abord python fetch_humaneval.py")

def functions(text):
    """Fonctions de premier niveau avec docstring : (prompt, corps), dédentés."""
    tree, lines = ast.parse(text), text.splitlines(keepends=True)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.decorator_list or node.name.startswith("test"):
            continue
        doc = node.body[0]
        if not (isinstance(doc, ast.Expr) and isinstance(getattr(doc, "value", None), ast.Constant)
                and isinstance(doc.value.value, str)) or len(node.body) < 2:
            continue
        prompt = "".join(lines[node.lineno - 1:doc.end_lineno])
        body = "".join(lines[doc.end_lineno:node.end_lineno])
        n_body = len(body.splitlines())
        if len(doc.value.value) < 30 or not 3 <= n_body <= 60 or len(prompt) + len(body) > 2500:
            continue
        yield prompt, body

groups = {"train": [], "val": []}
seen, stats = set(), dict(files=0, bad=0, contaminated=0, dup=0)
for path in sorted(root.rglob("*.py")):
    rel = path.relative_to(root)
    if any(p in SKIP for p in rel.parts[:-1]):
        continue
    group = rel.parts[0]
    try:
        text = clean_file(path.read_text(encoding="utf-8"))
        funcs = list(functions(text)) if text else []
    except (SyntaxError, UnicodeDecodeError, ValueError):
        stats["bad"] += 1
        continue
    stats["files"] += 1
    for prompt, body in funcs:
        if decontam.is_contaminated(prompt + body):
            stats["contaminated"] += 1
            continue
        digest = hashlib.sha1((prompt + body).encode()).hexdigest()
        if digest in seen:
            stats["dup"] += 1
            continue
        seen.add(digest)
        groups["val" if is_validation(group, 5) else "train"].append({"prompt": prompt, "completion": body})

rng = random.Random(args.seed)
out = Path("adapters") / args.domain
out.mkdir(parents=True, exist_ok=True)
for split, cap in (("train", args.max_examples), ("val", max(50, args.max_examples // 20))):
    rng.shuffle(groups[split])
    rows = groups[split][:cap]
    (out / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{split}: {len(rows)} exemples (sur {len(groups[split])} disponibles)")
print("stats :", stats)

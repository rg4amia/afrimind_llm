"""Génère le jeu de comportement RAG/agent : adapters/rag-agent/{train,val}.jsonl et le banc data/eval/rag.jsonl.
Sans modèle ni réseau. Les sujets du banc d'évaluation ne figurent jamais dans l'entraînement."""
import argparse, json
from pathlib import Path
from bekami.ragdata import generate

ap = argparse.ArgumentParser()
ap.add_argument("--domain", default="rag-agent")
ap.add_argument("--n-train", type=int, default=3000)
ap.add_argument("--n-val", type=int, default=150)
ap.add_argument("--n-eval", type=int, default=140)
ap.add_argument("--curated", default="data/curated", help="dossier de .jsonl {messages} relus par des humains, ajoutés à l'entraînement")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

def write(path, rows, keep):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps({k: r[k] for k in keep}, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{path}: {len(rows)} lignes")

def load_curated(folder):
    """Exemples écrits ou relus par des humains (langues africaines, documents réels). Refuse tout enregistrement mal formé."""
    rows = []
    for path in sorted(Path(folder).glob("*.jsonl")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            msgs = json.loads(line).get("messages")
            ok = (isinstance(msgs, list) and len(msgs) >= 2 and msgs[-1].get("role") == "assistant"
                  and all(m.get("role") in ("system", "user", "assistant") and isinstance(m.get("content"), str) and m["content"].strip() for m in msgs))
            if not ok:
                raise SystemExit(f"{path}:{i} : enregistrement invalide (messages avec rôles system/user/assistant, dernier = assistant)")
            rows.append(dict(messages=msgs))
    return rows

def dedupe(rows, banned=()):
    """Retire les doublons de prompt et tout prompt présent dans `banned` (pas de fuite entre jeux)."""
    seen, kept = set(banned), []
    for r in rows:
        key = json.dumps(r["messages"][:-1], ensure_ascii=False)
        if key not in seen:
            seen.add(key); kept.append(r)
    return kept, seen

keys = lambda rows: {json.dumps(r["messages"][:-1], ensure_ascii=False) for r in rows}
out = Path("adapters") / args.domain
curated = load_curated(args.curated)
print(f"{len(curated)} exemples relus ajoutés")
evalset, _ = dedupe(generate(args.n_eval * 2, args.seed + 2, "eval"))
evalset = evalset[:args.n_eval]
train, seen = dedupe(generate(args.n_train * 2, args.seed, "train") + curated, keys(evalset))
train = train[:args.n_train]
val, _ = dedupe(generate(args.n_val * 3, args.seed + 1, "train"), keys(train) | keys(evalset))
val = val[:args.n_val]
write(out / "train.jsonl", train, ["messages"])
write(out / "val.jsonl", val, ["messages"])
write("data/eval/rag.jsonl", evalset, ["kind", "messages", "expect"])
if not (out / "adapter.yaml").exists():
    (out / "adapter.yaml").write_text("""# Comportement RAG/agent : citer [n], dire « pas dans les sources », résister aux injections, appeler les outils.
# Les messages système sont déjà dans les données (identiques à ceux de l'exécution) : system_prompt reste null.
base_model: Qwen/Qwen2.5-1.5B-Instruct    # Apache-2.0, multilingue ; à confirmer (3B ou 7B si le GPU le permet)
quantize: 4bit
format: chat
system_prompt: null
train_file: train.jsonl
val_file: val.jsonl
max_length: 2048
epochs: 1
batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 0.0002
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
seed: 42
""", encoding="utf-8")
    print("créé", out / "adapter.yaml")

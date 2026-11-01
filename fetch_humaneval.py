"""Télécharge HumanEval (OpenAI, licence MIT) pour l'évaluation et pour la décontamination."""
import gzip, json, subprocess
from pathlib import Path

URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
Path("data/eval").mkdir(parents=True, exist_ok=True)
Path("data/decontam").mkdir(parents=True, exist_ok=True)
# curl plutôt que urllib : utilise les certificats du système (urllib échoue sur un Python macOS sans certificats)
raw = gzip.decompress(subprocess.run(["curl", "-fsSL", "--max-time", "60", URL], check=True, capture_output=True).stdout).decode("utf-8")
problems = [json.loads(l) for l in raw.splitlines() if l.strip()]
Path("data/eval/humaneval.jsonl").write_text(raw if raw.endswith("\n") else raw + "\n", encoding="utf-8")
# prompt + solution + tests : tout ce qui ne doit jamais apparaître dans les données d'entraînement
Path("data/decontam/humaneval.txt").write_text(
    "\n\n".join(p["prompt"] + p["canonical_solution"] + p["test"] for p in problems), encoding="utf-8")
print(f"{len(problems)} problèmes -> data/eval/humaneval.jsonl et data/decontam/humaneval.txt")

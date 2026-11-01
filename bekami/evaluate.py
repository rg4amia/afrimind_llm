import json
import os
import random
import re
import resource
import signal
import subprocess
import sys
import tempfile

import torch

from bekami.model import GPTConfig, BekamiGPT
from bekami.tokenizer import Tokenizer

# ---------- modèle -> fonction complete(prompt, max_new_tokens) -> texte ----------

def load_complete(ckpt_path, tokenizer_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = BekamiGPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return make_complete(model, Tokenizer(tokenizer_path), device)

def make_complete(model, tok, device):
    def complete(prompt, max_new_tokens=256):
        ids = tok.encode(prompt)[-(model.cfg.context_length - 1):]
        room = model.cfg.context_length - len(ids)
        out = model.generate(torch.tensor([ids], device=device),
                             max_new_tokens=min(max_new_tokens, room), temperature=0.0, top_k=1)
        return tok.decode(out[0, len(ids):].tolist())
    return complete

# ---------- long contexte : passkey / needle-in-a-haystack ----------

FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "

def passkey_prompt(tok, n_tokens, depth, key):
    """Prompt d'environ `n_tokens` tokens avec la phrase-clé enfouie à la profondeur `depth` (0..1)."""
    needle = tok.encode(f" The pass key is {key}. Remember it. {key} is the pass key. ")
    question = tok.encode("\nWhat is the pass key? The pass key is")
    filler = tok.encode(FILLER)
    n_fill = max(0, n_tokens - len(needle) - len(question))
    fill = (filler * (n_fill // len(filler) + 1))[:n_fill]
    cut = int(len(fill) * depth)
    return tok.decode(fill[:cut] + needle + fill[cut:] + question)

def run_passkey(complete, tok, lengths, depths=(0.1, 0.5, 0.9), trials=3, seed=0):
    rng = random.Random(seed)
    grid = {}
    for n in lengths:
        for d in depths:
            ok = 0
            for _ in range(trials):
                key = str(rng.randint(10000, 99999))
                ok += key in complete(passkey_prompt(tok, n, d, key), 12)
            grid[(n, d)] = ok / trials
    return grid

# ---------- code : pass@1 par exécution des tests ----------

STOPS = ("\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\nassert ")

def truncate(completion):
    cut = min([completion.find(s) for s in STOPS if s in completion] or [len(completion)])
    return completion[:cut]

def _limits(cpu_seconds, mem_bytes):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
        try:  # non appliqué par macOS ; effectif sous Linux (Colab, cluster)
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass
    return apply

def run_program(source, timeout=10, mem_bytes=2 << 30):
    """Exécute `source` dans un sous-processus Python isolé. Renvoie (réussi, message).
    Limites : temps, fichiers, mémoire (Linux). Pas d'isolation réseau ni système de fichiers :
    n'exécuter du code généré que dans un conteneur ou une session Colab jetable."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prog.py")
        with open(path, "w") as f:
            f.write(source)
        proc = subprocess.Popen(
            [sys.executable, "-I", path], cwd=tmp, env={"PATH": os.environ.get("PATH", "")},
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            start_new_session=True, preexec_fn=_limits(timeout + 1, mem_bytes))
        try:
            _, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return False, "timeout"
        return proc.returncode == 0, err.strip().splitlines()[-1] if err.strip() else ""

def load_problems(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def run_code_eval(problems, complete=None, canonical=False, timeout=10, max_new_tokens=256):
    """Format HumanEval : prompt, test, entry_point (et canonical_solution pour `canonical=True`,
    qui vérifie le banc lui-même en exécutant les solutions de référence)."""
    results = []
    for p in problems:
        body = p["canonical_solution"] if canonical else truncate(complete(p["prompt"], max_new_tokens))
        source = f'{p["prompt"]}{body}\n\n{p["test"]}\n\ncheck({p["entry_point"]})\n'
        ok, msg = run_program(source, timeout)
        results.append(dict(task_id=p.get("task_id"), passed=ok, error=msg))
    return sum(r["passed"] for r in results) / len(results), results

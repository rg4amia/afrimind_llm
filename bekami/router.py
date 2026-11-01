import re
import unicodedata
from pathlib import Path

import yaml

def normalize(text):
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")

def load_domains(path="domains.yaml"):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for d in cfg["domains"].values():
        # une expression de plusieurs mots pèse plus qu'un mot isolé : « code du travail » l'emporte sur « code »
        d["_patterns"] = [(re.compile(r"(?<!\w)" + re.escape(normalize(k)) + r"(?!\w)"), len(k.split()))
                          for k in d.get("keywords", [])]
    return cfg

def route(text, cfg, min_score=1):
    """Classe les domaines par poids total des mots-clés trouvés (1 par mot du mot-clé). Renvoie [(domaine, score)] trié, ou
    [("general", 0)] si rien ne dépasse `min_score`. Routeur volontairement simple : à remplacer par un
    classifieur entraîné quand on aura des exemples réels."""
    t = normalize(text)
    scores = {name: sum(w for p, w in d["_patterns"] if p.search(t)) for name, d in cfg["domains"].items()}
    ranked = sorted(((n, s) for n, s in scores.items() if s >= min_score), key=lambda x: -x[1])
    return ranked or [("general", 0)]

def build_system_prompt(cfg, domain):
    d = cfg["domains"][domain]
    return (d["system_prompt"].strip() + "\n\n" + cfg["common_policy"].strip()).replace("\n\n\n", "\n\n")

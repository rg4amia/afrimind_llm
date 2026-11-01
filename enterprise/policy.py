from dataclasses import dataclass
from pathlib import Path

import yaml

from bekami.router import normalize, route
import re

CODE_DOMAINS = {"informatique", "cybersecurite"}

@dataclass
class Policy:
    tenant: str
    company_name: str
    mode: str = "documents_plus_general"
    allowed_domains: list = None
    blocked_domains: list = ()
    blocked_topics: list = ()
    code_enabled: bool = True
    tools: list = ()
    extra_instructions: str = ""
    refusal_message: str = "Cette demande sort du cadre défini pour cet assistant."

    @classmethod
    def load(cls, path):
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        policy = cls(**raw)
        if policy.mode not in ("documents_only", "documents_plus_general"):
            raise ValueError(f"mode inconnu : {policy.mode}")
        unknown = set(policy.tools) - {"search_documents", "list_events", "add_document"}
        if unknown:
            raise ValueError(f"outils inconnus : {unknown}")
        policy._topics = [re.compile(r"(?<!\w)" + re.escape(normalize(t)) + r"(?!\w)") for t in policy.blocked_topics]
        return policy

    def blocked_topic_in(self, text):
        t = normalize(text)
        return next((p.pattern for p in self._topics if p.search(t)), None)

@dataclass
class Decision:
    allowed: bool
    domain: str
    reason: str = ""

def decide(question, policy, domains_cfg):
    """Décision avant tout appel au modèle. Prudente : si UN des domaines détectés est interdit, on refuse."""
    if policy.blocked_topic_in(question):
        return Decision(False, "general", "sujet interdit")
    ranked = route(question, domains_cfg)
    top = ranked[0][0]
    for name, _ in ranked:
        if name in policy.blocked_domains:
            return Decision(False, top, f"domaine interdit : {name}")
        if name in CODE_DOMAINS and not policy.code_enabled:
            return Decision(False, top, "génération de code désactivée")
    if policy.allowed_domains is not None and top != "general" and top not in policy.allowed_domains:
        return Decision(False, top, f"domaine non autorisé : {top}")
    return Decision(True, top)

def check_output(text, policy):
    """Seconde barrière : la réponse du modèle ne doit pas contenir de sujet interdit."""
    return policy.blocked_topic_in(text) is None

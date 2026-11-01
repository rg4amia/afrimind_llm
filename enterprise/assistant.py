from bekami.router import build_system_prompt
from enterprise.policy import check_output, decide
from enterprise.qa import build_messages

NOT_FOUND = "Je n'ai trouvé aucun document autorisé qui répond à cette question."
UNGROUNDED_NOTE = "\n\n(Réponse fondée sur des connaissances générales, pas sur les documents de l'entreprise.)"

class Assistant:
    """Assemble : politique de l'entreprise -> recherche autorisée -> modèle -> contrôle de la réponse.
    `chat(messages) -> texte` est n'importe quel moteur (llama.cpp sur CPU, API, etc.)."""

    def __init__(self, store, policy, domains_cfg, chat, k=5):
        self.store, self.policy, self.domains, self.chat, self.k = store, policy, domains_cfg, chat, k

    def ask(self, user, question):
        decision = decide(question, self.policy, self.domains)
        if not decision.allowed:
            self.store._log(user, "denied", decision.reason); self.store.db.commit()
            return dict(answer=self.policy.refusal_message, sources=[], domain=decision.domain,
                        denied=decision.reason, grounded=False)
        hits = self.store.search(user, question, k=self.k)
        if not hits and self.policy.mode == "documents_only":
            return dict(answer=NOT_FOUND, sources=[], domain=decision.domain, denied=None, grounded=False)
        domain_prompt = build_system_prompt(self.domains, decision.domain)
        messages = build_messages(self.policy.company_name, question, hits, domain_prompt, self.policy.extra_instructions)
        text = self.chat(messages)
        if not check_output(text, self.policy):
            self.store._log(user, "output_blocked", decision.domain); self.store.db.commit()
            return dict(answer=self.policy.refusal_message, sources=[], domain=decision.domain,
                        denied="réponse bloquée", grounded=False)
        if not hits:
            text += UNGROUNDED_NOTE
        return dict(answer=text, domain=decision.domain, denied=None, grounded=bool(hits),
                    sources=[dict(n=i, doc_id=h["doc_id"], title=h["title"], date=h["date"]) for i, h in enumerate(hits, 1)])

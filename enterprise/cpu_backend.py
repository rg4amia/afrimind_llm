class CPUChat:
    """Moteur d'inférence CPU hors ligne (llama.cpp, modèle GGUF quantifié). Même contrat que les autres moteurs :
    chat(messages) -> texte, avec messages = [(rôle, texte), ...]. Aucun accès réseau : le modèle est un fichier local."""

    def __init__(self, model_path, n_ctx=4096, n_threads=None, max_tokens=512, temperature=0.2):
        from llama_cpp import Llama
        self.llm = Llama(model_path=str(model_path), n_ctx=n_ctx, n_threads=n_threads, verbose=False)
        self.max_tokens, self.temperature = max_tokens, temperature

    def __call__(self, messages):
        out = self.llm.create_chat_completion(
            messages=[{"role": r, "content": c} for r, c in messages],
            max_tokens=self.max_tokens, temperature=self.temperature)
        return out["choices"][0]["message"]["content"].strip()

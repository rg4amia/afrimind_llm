"""python -m bekami.test_ragdata : données RAG/agent déterministes, banc sans fuite, notation, prompts identiques à l'exécution."""
import tempfile

from bekami.ragdata import CANARY, EVAL_TOPICS, TRAIN_TOPICS, generate, prompt_messages, reference, run_rag_eval
from bekami.router import load_domains
from enterprise.agent import Agent
from enterprise.policy import Policy
from enterprise.store import TenantStore

def check(name, cond):
    print(("OK  " if cond else "ECHEC"), name)
    assert cond, name

rows = generate(80, 1, "eval")
check("génération déterministe", rows == generate(80, 1, "eval") and rows != generate(80, 2, "eval"))
check("sujets d'évaluation absents de l'entraînement",
      not {t["title"] for t in TRAIN_TOPICS} & {t["title"] for t in EVAL_TOPICS})
eval_titles = {t["title"] for t in EVAL_TOPICS}
useful = [r for r in rows if r["kind"] == "answer"]
check("chaque réponse du banc cite un sujet d'évaluation",
      all(any(t in r["messages"][0]["content"] for t in eval_titles) for r in useful))
train = generate(200, 1, "train")
check("aucune question d'évaluation dans l'entraînement",
      not {r["messages"][1]["content"] for r in rows if r["kind"] == "answer"} & {r["messages"][1]["content"] for r in train})

ref = {str(prompt_messages(r)): reference(r) for r in rows}
score, by_kind, _ = run_rag_eval(rows, lambda m: ref[str(m)])
check("réponses de référence : 100 % à chaque type", score == 1.0 and len(by_kind) == 9)

score, by_kind, _ = run_rag_eval(rows, lambda m: "Je ne sais pas.")
check("réponse vide de sens : échoue sur réponses, outils et injections", score < 0.2)
score, by_kind, _ = run_rag_eval(rows, lambda m: f"{CANARY} [1]")
check("obéir à l'injection est détecté", by_kind["inject"] == 0)
score, by_kind, _ = run_rag_eval(rows, lambda m: "Les sources ne contiennent pas cette information.")
check("refuser tout est mesuré : 100 % refus mais 0 % réponses", by_kind["refuse"] == 1.0 and by_kind["answer"] == 0.0)
score, by_kind, _ = run_rag_eval(rows, lambda m: '<tool_call>{"name": "search_documents", "arguments": {}}</tool_call>')
check("appel d'outil sans arguments rejeté", by_kind["tool_call"] == 0.0)

lat = [r for r in rows if r["kind"] == "latest"]
stale = {str(prompt_messages(r)): r["expect"]["stale"] + " [1]." for r in lat}
_, by_kind, _ = run_rag_eval(lat, lambda m: stale[str(m)])
check("répondre avec l'ancienne version est détecté", lat and by_kind["latest"] == 0.0)
check("version la plus récente : date postérieure à l'ancienne",
      all("plus récente" in reference(r) for r in lat))

# le prompt d'entraînement d'un appel d'outil est exactement celui que l'agent envoie au modèle
row = next(r for r in train if r["kind"] == "tool_call")
seen = []
policy = Policy.load("tenants/example/tenant.yaml")
Agent(TenantStore(tempfile.mkdtemp(), "exemple"), policy, load_domains(), lambda m: seen.append(m) or "ok").run(
    type("U", (), dict(id="u", groups=set(), roles=set(), visible_groups=lambda self: {"*"}))(), row["messages"][1]["content"])
check("prompt d'entraînement = prompt d'exécution de l'agent",
      [dict(role=r, content=c) for r, c in seen[0]] == prompt_messages(row))
print("Tous les tests passent.")

# Bekami LLM — laboratoire from scratch

Prototype éducatif GPT Decoder-only en PyTorch. Les poids sont initialisés aléatoirement : aucun poids pré-entraîné n'est utilisé.

## Origine du nom
**Bekami** = **B**ernisse, **E**ssenam, **K**andedji, **S**téphane, **A**mia, **M**igan — les prénoms et noms de la famille Amia (Thierry Stéphane, Bernisse Klahon, Essenam Ludivine Jemima, Kandedji Shekinaelle Shiphra, Migan Shilo Doxa).

## Important
- La configuration fournie vise un petit modèle expérimental (~20–30M paramètres).
- Un modèle entraîné sur quelques millions de tokens ne devient pas automatiquement un assistant fiable.
- Ne pas utiliser de données personnelles, confidentielles ou protégées sans autorisation.
- Vérifier les licences et la qualité de chaque corpus.
- Prévoir un espace de stockage persistant (Google Drive) et sauvegarder les checkpoints.

## Démarrage dans Google Colab

1. Activer `Exécution > Modifier le type d'exécution > GPU`.
2. Vérifier les ressources :
```python
!nvidia-smi
!free -h
```
3. Décompresser le projet et installer :
```bash
!pip install -r requirements.txt
```
4. Déposer les données dans `data/raw/` : un dossier par dépôt dans `data/raw/repos/<nom>/` (code, licences permissives uniquement) et/ou des `.txt` UTF-8 pour la documentation. Pour décontaminer, placer le texte des benchmarks d'évaluation dans `data/decontam/*.txt`.
5. Entraîner le tokenizer :
```bash
!python train_tokenizer.py
```
6. Encoder les corpus :
```bash
!python prepare_data.py
```
7. Vérifier que `config.yaml` et le tokenizer ont le même vocabulaire, puis entraîner :
```bash
!python train.py
```
8. Générer :
```bash
!python generate.py
```

## Premier test conseillé
Commencer avec 5 à 20 Mo de texte propre pour valider toute la chaîne. Cela sert uniquement à tester le pipeline, pas à obtenir un modèle généraliste.

## Données et langues
Commencer par des corpus publics et licenciés, avec des fichiers UTF-8 distincts. Garder un registre `source, langue, licence, volume, date, filtrage`. Réserver des documents de validation indépendants de l'entraînement. Pour les langues africaines, rechercher des corpus vérifiés et des locuteurs compétents pour l'évaluation.

## Limites techniques de cette V1
- Pas encore de instruction tuning, de chat template, de filtrage avancé ni d'évaluation linguistique formelle.
- Le modèle est un prototype causal de recherche.
- Le nettoyage écarte les fichiers générés, minifiés ou contenant des secrets, dédoublonne à l'identique et décontamine par fenêtres de 10 mots. Il ne fait pas encore de dédoublonnage approximatif (MinHash), de détection de données personnelles ni de tri des fichiers par dépendances.
- Le tokenizer (BPE octet, symboles FIM, `<file_sep>`, `<repo_name>`, outils, chat) doit être entraîné sur du code réel avant d'utiliser `configs/s.yaml` ou `configs/m.yaml` (vocabulaire de 64 000).
- Le vocabulaire est configurable, mais doit être identique dans le tokenizer et le modèle.

## Piste B : adapter un modèle de code open-weights (QLoRA)
Base par défaut : `deepseek-ai/deepseek-coder-1.3b-instruct` (contexte 16 384, licence « deepseek-license » : lire ses conditions avant tout usage commercial). Pour le 6,7 B, changer `base_model` dans `adapter.yaml`.

```bash
pip install -r requirements-ft.txt
cp -r adapters/_template adapters/php-laravel     # un dossier par domaine
# remplacer adapters/php-laravel/train.jsonl par vos données :
#   {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}   ou   {"prompt": ..., "completion": ...}
python finetune.py php-laravel                    # écrit adapters/php-laravel/weights/
python adapters.py list
python evaluate.py code --hf deepseek-ai/deepseek-coder-1.3b-instruct --adapter php-laravel --problems data/eval/humaneval.jsonl
python adapters.py merge php-laravel --out merged/php-laravel   # modèle autonome
```
Le template de DeepSeek impose par défaut un message système « informatique uniquement » : pour un domaine hors code, renseigner `system_prompt` dans `adapter.yaml`.

Seul le dernier tour de l'assistant contribue à la perte. Mesurer le modèle de base avec `evaluate.py` avant d'entraîner : un adaptateur qui fait baisser le score de base est un échec, même si la perte d'entraînement diminue.

## Architecture et orientation

Bekami a deux pistes, et une couche qui les sert :

| Couche | Rôle | GPU |
|---|---|---|
| `bekami/` + `train.py` (piste A) | GPT from scratch : apprendre, valider la chaîne. `config.yaml` (~16 M) tient sur un Colab gratuit ; `s.yaml` (~124 M) et `m.yaml` (~1,3 B) demandent bien plus de données et de calcul qu'un prototype n'en a | oui |
| `finetune.py` + `adapters/` (piste B) | **Voie principale** : un adaptateur QLoRA par domaine sur un modèle ouvert de 1–7 B, fusionnable | un seul GPU 16 Go suffit |
| `enterprise/` | Exécution : politique -> RAG avec permissions -> modèle -> contrôle de sortie ; boucle d'agent (`agent.py`) | non : llama.cpp sur CPU |

Principe : un petit modèle ouvert, spécialisé par adaptateurs, ne retient pas les faits (c'est le rôle du RAG) mais le **comportement** : citer ses sources, refuser quand elles manquent, appeler les outils au bon format.

### Agent
`enterprise/agent.py` exécute `search_documents`, `list_events`, `add_document` avec les droits de l'utilisateur. Format d'appel : `<tool_call>{"name", "arguments"}</tool_call>` -> `<tool_result>…</tool_result>` (mêmes symboles que le tokenizer). Seuls les outils de `policy.tools` existent, le nombre d'étapes est borné, et les résultats d'outils sont traités comme des données.

```python
from enterprise.agent import Agent
Agent(store, policy, domains, chat).run(user, "Quelles réunions cette semaine ?")
```

### Reprise d'entraînement (piste A)
`python train.py config.yaml --resume` repart de `checkpoints/last.pt` (poids + optimiseur + pas) après une coupure Colab.

### Feuille de route (par ordre de valeur)
1. **Données de comportement RAG/agent** (démarré : `python build_rag_sft.py`, voir ci-dessous) : générer des exemples `{sources, question} -> réponse citée [n]`, `-> « pas dans les sources »`, et des trajectoires d'appels d'outils ; les mettre dans `adapters/rag-agent/train.jsonl`. C'est ce qui rend le modèle utile en agent, bien plus que d'ajouter des paramètres.
2. **Évaluation RAG** (`data/eval/rag.jsonl`) : taux de citations valides, taux de refus correct, taux d'appels d'outils bien formés. Mesurer le modèle de base avant tout entraînement.
3. **Récupération hybride** : garder BM25 (FTS5), ajouter un petit modèle d'embeddings sur CPU (ex. multilingual-e5-small) et fusionner les classements (RRF). Les permissions restent appliquées dans la requête.
4. **Quantification et livraison** : fusionner l'adaptateur, convertir en GGUF Q4_K_M, brancher `CPUChat`, empaqueter avec `deploy/build_bundle.sh`.
5. **Piste A** : ne passer à `s.yaml` qu'avec un corpus propre de plusieurs milliards de tokens ; sinon rester sur la piste B.

### Jeu de données et banc RAG/agent
```bash
python build_rag_sft.py                       # adapters/rag-agent/{train,val}.jsonl + data/eval/rag.jsonl
python evaluate_rag.py --oracle               # vérifie le banc (doit afficher 100 %)
python evaluate_rag.py --hf Qwen/Qwen2.5-1.5B-Instruct                    # modèle de base, AVANT entraînement
python finetune.py rag-agent
python evaluate_rag.py --hf Qwen/Qwen2.5-1.5B-Instruct --adapter rag-agent
```
9 types de cas : réponse citée, version la plus récente quand deux sources se contredisent, refus (sources sans réponse), résistance à une instruction injectée dans une source, appel `search_documents` / `list_events`, réponse après résultat d'outil, résultat vide. Les prompts sont construits par les mêmes fonctions que l'exécution (`enterprise.qa`, `enterprise.agent.agent_head`) ; les sujets du banc n'apparaissent jamais dans l'entraînement. Sur Colab, `Bekami_Colab_RAG_Agent.ipynb` enchaîne ces étapes (régénération, mesure de base, entraînement, mesure avec adaptateur, tableau comparatif par comportement). `--out fichier.json` de `evaluate_rag.py` enregistre les scores.

Les données comptent 20 sujets d'entraînement (6 pour le banc), plusieurs formulations par sujet, l'accord singulier/pluriel, des préfixes de politesse, des distracteurs et des documents sans date. `build_rag_sft.py` retire les doublons de prompt et garantit qu'aucun prompt n'est commun à l'entraînement, à la validation et au banc. Limite : ce sont toujours des gabarits en français : ils apprennent le **format** et la discipline de citation, pas la variété du langage réel. Pour les documents réels de l'entreprise et les langues africaines, déposer des exemples relus par des locuteurs dans `data/curated/*.jsonl` (voir `data/curated/README.md`) : ils sont ajoutés à l'entraînement au prochain `build_rag_sft.py`. Ne pas y mettre de traduction automatique non relue.

Tests : `python -m bekami.test_ragdata`, `python -m bekami.test_router`, `python -m enterprise.test_store`, `python -m enterprise.test_assistant`, `python -m enterprise.test_agent`.

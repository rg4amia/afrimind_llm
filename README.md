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

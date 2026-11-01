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
4. Déposer des corpus texte UTF-8 dans `data/raw/`, un fichier par langue ou source.
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
- Le script de préparation charge chaque fichier texte en mémoire : pour de très grands corpus, le remplacer par un pipeline streaming/sharding.
- Le vocabulaire est configurable, mais doit être identique dans le tokenizer et le modèle.

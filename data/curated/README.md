# Exemples relus par des humains

Déposez ici des fichiers `.jsonl`, une ligne par exemple :

    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

`python build_rag_sft.py` les ajoute à l'entraînement (le dernier message doit être celui de l'assistant, seul tour appris).
Pour les langues africaines (fon, éwé, yoruba, wolof, lingala, swahili, bambara…) : texte écrit ou **relu par un locuteur compétent**,
avec le même message système que l'exécution (voir `enterprise/qa.py` et `enterprise/agent.py`), sources citées `[n]`,
et des cas de refus (« les sources ne contiennent pas cette information », dans la langue). Ne pas utiliser de traduction automatique non relue.
Réserver une part de ces exemples à un banc d'évaluation séparé, jamais à l'entraînement.

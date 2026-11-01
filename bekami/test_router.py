"""python -m bekami.test_router"""
from bekami.router import load_domains, route
cfg = load_domains()
cases = {
 "Mon patron veut me licencier sans préavis, que dit le code du travail ?": "droit",
 "Comment épargner 50 000 FCFA par mois avec le mobile money ?": "finance_economie",
 "Aide-moi à faire un business plan pour une boutique": "entrepreneuriat_marketing",
 "Explique-moi les dérivées pour mon examen du bac": "education|mathematiques_logique",
 "Calcule la probabilité d'obtenir deux fois pile": "mathematiques_logique",
 "Quel engrais pour une plantation de cacao ?": "agriculture_environnement",
 "Pourquoi la colonisation a-t-elle pris fin ? histoire de l'indépendance": "histoire_geographie",
 "Corrige l'orthographe de ce texte": "redaction_langue",
 "Prépare-moi pour un entretien d'embauche, voici mon CV": "emploi_carriere",
 "Je suis très stressé et je me sens seul depuis le deuil": "psychologie_bienetre",
 "Je suis triste et stressé, que faire ?": "psychologie_bienetre",
 "Une recette de kedjenou avec les quantités": "nutrition_cuisine",
 "Mon compte a été piraté, comment changer mon mot de passe ?": "cybersecurite",
 "Comment installer un onduleur et une batterie solaire ?": "ingenierie_btp_energie",
 "Quelles pièces pour un extrait de naissance à la mairie ?": "administratif_ci",
 "Quel album de zouglou me conseilles-tu ?": "arts_culture_loisirs",
 "Programme de musculation pour débutant": "sport",
 "Je cherche un hôtel et un billet d'avion pour les vacances": "voyage_tourisme",
 "Entraîne un modèle de machine learning en Python": "informatique",
 "J'ai de la fièvre depuis deux jours": "medecine",
 "Que dit la Bible sur le pardon ?": "foi_spiritualite",
 "Comment dit-on bonjour en baoulé ?": "langues_africaines",
 "Parle-moi des coutumes de l'ethnie Sénoufo en Côte d'Ivoire": "cultures_ethnies_ci",
 "Explique la photosynthèse et l'ADN": "biologie_science",
 "Raconte-moi une blague": "general",
 "Bonjour, comment ça va ?": "general",
 "Donne-moi un conseil pour bien dormir": "general",
 "Je vais tourner à droite au carrefour": "general",
}
bad = [(q, route(q, cfg)[:2]) for q, w in cases.items() if route(q, cfg)[0][0] not in w.split("|")]
for q, got in bad: print("KO", got, "<-", q)
print(f"{len(cases) - len(bad)}/{len(cases)}")
assert not bad

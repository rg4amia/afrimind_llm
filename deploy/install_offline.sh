#!/usr/bin/env bash
# À lancer chez le client, SANS internet, dans le dossier décompressé du paquet.
set -euo pipefail
cd "$(dirname "$0")"
echo "1/4 Vérification de l'intégrité des fichiers"
shasum -a 256 -c SHA256SUMS --quiet
echo "2/4 Création de l'environnement et installation depuis le paquet local (aucun accès réseau)"
python3 -m venv .venv
.venv/bin/pip install --no-index --find-links wheelhouse -r requirements-runtime.txt -q
echo "3/4 Auto-test de la politique et des permissions"
.venv/bin/python -m enterprise.test_store >/dev/null && .venv/bin/python -m enterprise.test_assistant >/dev/null
echo "4/4 Terminé. Modèle(s) disponible(s) : $(ls models)"

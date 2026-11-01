#!/usr/bin/env bash
# À lancer sur une machine AVEC internet, de même système et même version de Python que le serveur du client.
# Usage : deploy/build_bundle.sh <modele.gguf> [dossier_sortie]
# Pour un autre système cible, définir PIP_PLATFORM_ARGS, par exemple :
#   PIP_PLATFORM_ARGS="--platform manylinux2014_x86_64 --python-version 3.11 --only-binary=:all:"
set -euo pipefail
MODEL="${1:?modèle .gguf requis}"
OUT="${2:-dist/bekami-enterprise}"
rm -rf "$OUT" && mkdir -p "$OUT/wheelhouse" "$OUT/models"
python3 -m pip download -q -r requirements-runtime.txt -d "$OUT/wheelhouse" \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu ${PIP_PLATFORM_ARGS:-}
cp "$MODEL" "$OUT/models/"
cp -r bekami enterprise tenants domains.yaml requirements-runtime.txt deploy/install_offline.sh "$OUT/"
find "$OUT" -name __pycache__ -prune -exec rm -rf {} +
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS)
tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
shasum -a 256 "$OUT.tar.gz" | tee "$OUT.tar.gz.sha256"
echo "Paquet : $OUT.tar.gz  (à transmettre avec son empreinte .sha256, par un canal distinct si possible)"

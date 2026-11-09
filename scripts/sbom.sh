#!/usr/bin/env bash
# CycloneDX SBOM of the active environment -> dist/sbom.cdx.json
# Needs cyclonedx-bom (in the dev extra). Run inside the dev venv.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
python -m cyclonedx_py environment --pyproject pyproject.toml \
  --output-reproducible --of JSON -o dist/sbom.cdx.json
echo "wrote dist/sbom.cdx.json"

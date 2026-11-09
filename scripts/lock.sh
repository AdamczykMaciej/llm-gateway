#!/usr/bin/env bash
# Regenerate constraints/dev.txt from the active venv, which must have
# ".[dev,service,azure,vertex]" installed. Use it with: pip install -c constraints/dev.txt -e ".[dev,...]"
set -euo pipefail
cd "$(dirname "$0")/.."
{
  echo "# Pinned test/dev environment (pip freeze). Regenerate with scripts/lock.sh."
  echo "# Not a runtime constraint: the published package keeps its own bounds."
  python -m pip freeze --exclude-editable | LC_ALL=C sort -f
} > constraints/dev.txt

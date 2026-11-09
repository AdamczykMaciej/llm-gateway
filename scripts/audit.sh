#!/usr/bin/env bash
# Local dependency audit (no CI minutes needed): pip-audit on the active
# environment, plus osv-scanner on constraints/dev.txt when it is installed.
#
# Usage: scripts/audit.sh            (run inside the venv with ".[dev,service,azure,vertex]")
# Ignore list: scripts/audit-ignore.txt, one advisory id per line:
#     <ID>  <expiry YYYY-MM-DD>  <reason>
# e.g. GHSA-xxxx-xxxx-xxxx  2026-12-31  no fix released, not reachable from our code
# Blank lines and lines starting with # are skipped. An entry past its expiry
# is an error, so ignores cannot be forgotten.
set -euo pipefail
cd "$(dirname "$0")/.."
IGNORES="scripts/audit-ignore.txt"
args=()
today=$(date +%Y-%m-%d)
if [ -f "$IGNORES" ]; then
  while read -r id expiry reason; do
    case "${id:-}" in ""|\#*) continue ;; esac
    if [ -z "${expiry:-}" ] || [ -z "${reason:-}" ]; then
      echo "audit-ignore.txt: '$id' needs an expiry date and a reason" >&2; exit 2
    fi
    if [[ "$expiry" < "$today" ]]; then
      echo "audit-ignore.txt: ignore for $id expired on $expiry" >&2; exit 2
    fi
    args+=(--ignore-vuln "$id")
  done < "$IGNORES"
fi
python -m pip_audit --skip-editable "${args[@]+"${args[@]}"}"
if command -v osv-scanner >/dev/null 2>&1; then
  osv-scanner scan --lockfile requirements.txt:constraints/dev.txt
else
  echo "osv-scanner not installed; skipped (optional)"
fi

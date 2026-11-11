#!/usr/bin/env bash
set -euo pipefail
repo="${1:-$PWD}"
out="${2:-$repo/real_material_upload_bundle.txt}"
ps="$repo/RefractiveIndices/Polystyrene.txt"
pmma="$repo/RefractiveIndices/PolymethylMethacrylate.txt"
for f in "$ps" "$pmma"; do [[ -f "$f" ]] || { echo "Missing: $f" >&2; exit 1; }; done
{
  echo '===== BEGIN Polystyrene.txt ====='
  cat "$ps"
  echo
  echo '===== END Polystyrene.txt ====='
  echo '===== BEGIN PolymethylMethacrylate.txt ====='
  cat "$pmma"
  echo
  echo '===== END PolymethylMethacrylate.txt ====='
} > "$out"
echo "Wrote $out"

#!/usr/bin/env bash

# Exit on error, unset var, or pipefail
set -euo pipefail

# Accept the map file as first argument, or default if not provided
MAP_FILE="${1:-.github/kfp-pipelines-map.json}"
TMP_DIR="$(mktemp -d)"

if [[ ! -f "$MAP_FILE" ]]; then
  echo "❌ Pipeline map file not found: $MAP_FILE"
  exit 1
fi

# Loop through JSON map entries using jq
jq -r 'to_entries[] | "\(.key)\t\(.value)"' "$MAP_FILE" \
| while IFS=$'\t' read -r py_file yaml_file; do
    echo "→ Checking $py_file → $yaml_file"

    # verify source and target exist
    if [[ ! -f "$py_file" ]]; then
      echo "❌ Python file not found: $py_file"
      exit 1
    fi
    if [[ ! -f "$yaml_file" ]]; then
      echo "❌ Expected YAML missing: $yaml_file"
      exit 1
    fi

    # compile to temp and show diff if any
    kfp dsl compile --py "$py_file" --output "$TMP_DIR/tmp.yaml"
    if ! diff -u "$yaml_file" "$TMP_DIR/tmp.yaml"; then
      echo "❌ $yaml_file is out of date with $py_file"
      echo "   → update by running:"
      echo "     kfp dsl compile --py $py_file --output $yaml_file"
      exit 1
    else
      echo "✅ $yaml_file is up to date"
    fi
done
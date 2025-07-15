#!/usr/bin/env bash

set -euo pipefail

# Parse command line arguments
TAG="${1:-latest}"

# Shift past the TAG argument if it was provided
if [[ $# -gt 0 ]]; then
  shift
fi

TAG=${TAG} docker buildx bake --push

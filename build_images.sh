#!/usr/bin/env bash

set -euo pipefail

# Function to print bold text
print_bold() {
  echo -e "\033[1m$1\033[0m"
}

# Default platform is linux/arm64 if not specified
PLATFORM="linux/arm64"

# Parse named parameters
while [[ $# -gt 0 ]]; do
  case $1 in
    --platform)
      PLATFORM="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: $0 [--platform PLATFORM]"
      exit 1
      ;;
  esac
done

cd pipeline
print_bold "🔨 Building pipeline image..."
docker build --platform "$PLATFORM" -t "quay.io/hbelmiro/fraud-detection-e2e-demo-pipeline:latest" -f Containerfile .
print_bold "🚀 Pushing pipeline image..."
docker push quay.io/hbelmiro/fraud-detection-e2e-demo-pipeline:latest
echo "✅ Pipeline image successfully built and pushed"

cd ../feature_engineering
print_bold "🔨 Building feature engineering image..."
docker build --platform "$PLATFORM" -t "quay.io/hbelmiro/fraud-detection-e2e-demo-feature-engineering:latest" -f Containerfile .
print_bold "🚀 Pushing feature engineering image..."
docker push quay.io/hbelmiro/fraud-detection-e2e-demo-feature-engineering:latest
echo "✅ Feature engineering image successfully built and pushed"

cd ../train
print_bold "🔨 Building train image..."
docker build --platform "$PLATFORM" -t "quay.io/hbelmiro/fraud-detection-e2e-demo-train:latest" -f Containerfile .
print_bold "🚀 Pushing train image..."
docker push quay.io/hbelmiro/fraud-detection-e2e-demo-train:latest
echo "✅ Train image successfully built and pushed"

cd ../rest_predictor
print_bold "🔨 Building REST predictor image..."
docker build --platform "$PLATFORM" -t "quay.io/hbelmiro/fraud-detection-e2e-demo-rest-predictor:latest" -f Containerfile .
print_bold "🚀 Pushing REST predictor image..."
docker push quay.io/hbelmiro/fraud-detection-e2e-demo-rest-predictor:latest
echo "✅ REST predictor image successfully built and pushed"

cd ..

#!/usr/bin/env bash

set -euo pipefail

# Function to print bold text
print_bold() {
  echo -e "\033[1m$1\033[0m"
}

# Default platform is linux/amd64,linux/arm64 if not specified
PLATFORM="linux/amd64,linux/arm64"

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
docker buildx build --platform "$PLATFORM" -t "quay.io/hbelmiro/fraud-detection-e2e-demo-pipeline:latest" -f Containerfile --push .
echo "✅ Pipeline image successfully built and pushed"

cd ../data_preparation
print_bold "🔨 Building data preparation image..."
docker buildx build --platform "$PLATFORM" -t "quay.io/hbelmiro/fraud-detection-e2e-demo-data-preparation:latest" -f Containerfile --push .
echo "✅ Data preparation image successfully built and pushed"

cd ../feature_engineering
print_bold "🔨 Building feature engineering image..."
docker buildx build --platform "$PLATFORM" -t "quay.io/hbelmiro/fraud-detection-e2e-demo-feature-engineering:latest" -f Containerfile --push .
echo "✅ Feature engineering image successfully built and pushed"

cd ../train
print_bold "🔨 Building train image..."
docker buildx build --platform "$PLATFORM" -t "quay.io/hbelmiro/fraud-detection-e2e-demo-train:latest" -f Containerfile --push .
echo "✅ Train image successfully built and pushed"

cd ../rest_predictor
print_bold "🔨 Building REST predictor image..."
docker buildx build --platform "$PLATFORM" -t "quay.io/hbelmiro/fraud-detection-e2e-demo-rest-predictor:latest" -f Containerfile --push .
echo "✅ REST predictor image successfully built and pushed"

cd ..

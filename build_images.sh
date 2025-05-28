#!/usr/bin/env bash

set -euo pipefail

# Function to print bold text
print_bold() {
  echo -e "\033[1m$1\033[0m"
}

# Default platform is linux/arm64 if not specified
PLATFORM="linux/arm64"
# Default target is development if not specified
TARGET="development"

# Parse named parameters
while [[ $# -gt 0 ]]; do
  case $1 in
    --platform)
      PLATFORM="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: $0 [--platform PLATFORM] [--target TARGET]"
      echo "       TARGET can be 'development' or 'ci'"
      exit 1
      ;;
  esac
done

# Validate TARGET parameter
if [[ "$TARGET" != "development" && "$TARGET" != "ci" ]]; then
  echo "Error: TARGET must be either 'development' or 'ci'"
  echo "Usage: $0 [--platform PLATFORM] [--target TARGET]"
  echo "       TARGET can be 'development' or 'ci'"
  exit 1
fi

# Set image tag suffix based on TARGET
TAG_SUFFIX=""
if [[ "$TARGET" == "ci" ]]; then
  # Override platform to linux/amd64 for CI
  PLATFORM="linux/amd64"
  TAG_SUFFIX="-ci"
  print_bold "🔧 Building for CI with platform $PLATFORM"
else
  print_bold "🔧 Building for development with platform $PLATFORM"
fi

cd pipeline
print_bold "🔨 Building pipeline image..."
docker build --platform "$PLATFORM" -t "quay.io/hbelmiro/fraud-detection-e2e-demo-pipeline${TAG_SUFFIX}:latest" -f Containerfile .
print_bold "🚀 Pushing pipeline image..."
docker push quay.io/hbelmiro/fraud-detection-e2e-demo-pipeline${TAG_SUFFIX}:latest
echo "✅ Pipeline image successfully built and pushed"

cd ../data_preparation
print_bold "🔨 Building data preparation image..."
docker build --platform "$PLATFORM" -t "quay.io/hbelmiro/fraud-detection-e2e-demo-data-preparation${TAG_SUFFIX}:latest" -f Containerfile .
print_bold "🚀 Pushing data preparation image..."
docker push quay.io/hbelmiro/fraud-detection-e2e-demo-data-preparation${TAG_SUFFIX}:latest
echo "✅ Data preparation image successfully built and pushed"

cd ../feature_engineering
print_bold "🔨 Building feature engineering image..."
docker build --platform "$PLATFORM" -t "quay.io/hbelmiro/fraud-detection-e2e-demo-feature-engineering${TAG_SUFFIX}:latest" -f Containerfile .
print_bold "🚀 Pushing feature engineering image..."
docker push quay.io/hbelmiro/fraud-detection-e2e-demo-feature-engineering${TAG_SUFFIX}:latest
echo "✅ Feature engineering image successfully built and pushed"

cd ../train
print_bold "🔨 Building train image..."
docker build --platform "$PLATFORM" -t "quay.io/hbelmiro/fraud-detection-e2e-demo-train${TAG_SUFFIX}:latest" -f Containerfile .
print_bold "🚀 Pushing train image..."
docker push quay.io/hbelmiro/fraud-detection-e2e-demo-train${TAG_SUFFIX}:latest
echo "✅ Train image successfully built and pushed"

cd ../rest_predictor
print_bold "🔨 Building REST predictor image..."
docker build --platform "$PLATFORM" -t "quay.io/hbelmiro/fraud-detection-e2e-demo-rest-predictor${TAG_SUFFIX}:latest" -f Containerfile .
print_bold "🚀 Pushing REST predictor image..."
docker push quay.io/hbelmiro/fraud-detection-e2e-demo-rest-predictor${TAG_SUFFIX}:latest
echo "✅ REST predictor image successfully built and pushed"

cd ..

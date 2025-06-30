#!/usr/bin/env bash

set -euo pipefail

# Function to print bold text
print_bold() {
  echo -e "\033[1m$1\033[0m"
}

cd pipeline
print_bold "🔨 Building pipeline image..."
docker buildx build --platform linux/amd64 -t "quay.io/hbelmiro/fraud-detection-e2e-demo-pipeline-rhoai:latest" -f Containerfile --push .
echo "✅ Pipeline image successfully built and pushed"

cd ../data_preparation
print_bold "🔨 Building data preparation image..."
docker buildx build --platform linux/amd64 -t "quay.io/hbelmiro/fraud-detection-e2e-demo-data-preparation-rhoai:latest" -f Containerfile --push .
echo "✅ Data preparation image successfully built and pushed"

cd ../feature_engineering
print_bold "🔨 Building feature engineering image..."
docker buildx build --platform linux/amd64 -t "quay.io/hbelmiro/fraud-detection-e2e-demo-feature-engineering-rhoai:latest" -f Containerfile --push .
echo "✅ Feature engineering image successfully built and pushed"

cd ../train
print_bold "🔨 Building train image..."
docker buildx build --platform linux/amd64 -t "quay.io/hbelmiro/fraud-detection-e2e-demo-train-rhoai:latest" -f Containerfile --push .
echo "✅ Train image successfully built and pushed"

cd ../rest_predictor
print_bold "🔨 Building REST predictor image..."
docker buildx build --platform linux/amd64 -t "quay.io/hbelmiro/fraud-detection-e2e-demo-rest-predictor-rhoai:latest" -f Containerfile --push .
echo "✅ REST predictor image successfully built and pushed"

cd ..

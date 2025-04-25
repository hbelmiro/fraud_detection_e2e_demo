#!/usr/bin/env bash

set -euo pipefail

# Configure MinIO Client
mc alias set local http://localhost:9000 minio minio123

# Create directory structure
mc mb local/mlpipeline/artifacts/feature_repo/data/input --p

# Upload data files
mc cp \
  feature_engineering/feature_repo/data/input/raw_transaction_datasource.csv \
  feature_engineering/feature_repo/data/input/test.csv \
  feature_engineering/feature_repo/data/input/train.csv \
  feature_engineering/feature_repo/data/input/validate.csv \
  local/mlpipeline/artifacts/feature_repo/data/input/

# Upload feature store configuration
mc cp \
  feature_engineering/feature_repo/feature_store.yaml \
  local/mlpipeline/artifacts/feature_repo/

# Verify the upload (optional)
mc ls --recursive local/mlpipeline/artifacts/

# Install Model Registry
kubectl apply -k "https://github.com/kubeflow/model-registry/manifests/kustomize/overlays/db?ref=v0.2.16"

# Install KServe
kubectl create namespace kserve
oc project kserve
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.15/hack/quick_install.sh" | bash
oc project kubeflow

# Adjust RBAC policies
kubectl apply -k ./manifests -n kubeflow

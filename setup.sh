#!/usr/bin/env bash

set -euo pipefail

# Get MinIO credentials from the cluster
ACCESS_KEY=$(kubectl get secret ds-pipeline-s3-sample -n fraud-detection -oyaml | grep accesskey | awk '{print $2}' | base64 --decode)
SECRET_KEY=$(kubectl get secret ds-pipeline-s3-sample -n fraud-detection -oyaml | grep secretkey | awk '{print $2}' | base64 --decode)

echo "Using MinIO credentials:"
echo "Access Key: $ACCESS_KEY"
echo "Secret Key: $SECRET_KEY"

# Configure MinIO Client with correct credentials
mc alias set local http://localhost:9000 "$ACCESS_KEY" "$SECRET_KEY"

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

# Install Kubeflow Spark Operator
helm install spark-operator spark-operator/spark-operator \
    --namespace spark-operator \
    --create-namespace \
    --set webhook.enable=true

# Make sure the Kubeflow Spark Operator is watching the fraud-detection namespace. Run this command to let it watch all namespaces:
helm upgrade spark-operator spark-operator/spark-operator --set spark.jobNamespaces={} --namespace spark-operator

# Apply OpenShift compatibility patches for Spark Operator deployments
# These patches remove security context fields that conflict with OpenShift SCCs
kubectl patch deployment spark-operator-controller -n spark-operator --type='json' -p='[{"op": "remove", "path": "/spec/template/spec/securityContext/fsGroup"}]' || true
kubectl patch deployment spark-operator-controller -n spark-operator --type='json' -p='[{"op": "remove", "path": "/spec/template/spec/containers/0/securityContext/seccompProfile"}]' || true
kubectl patch deployment spark-operator-webhook -n spark-operator --type='json' -p='[{"op": "remove", "path": "/spec/template/spec/securityContext/fsGroup"}]' || true
kubectl patch deployment spark-operator-webhook -n spark-operator --type='json' -p='[{"op": "remove", "path": "/spec/template/spec/containers/0/securityContext/seccompProfile"}]' || true

kubectl apply -k ./manifests

# Label the fraud-detection namespace for Istio mesh connectivity
kubectl label namespace fraud-detection maistra.io/member-of=istio-system --overwrite

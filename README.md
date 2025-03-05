This is a work in progress.

## MinIO

### Port forward

```shell
kubectl port-forward --namespace kubeflow svc/minio-service 9000:9000
```

### Get MinIO credentials

```shell
ACCESS_KEY=kubectl get secret mlpipeline-minio-artifact -oyaml | yq .data.accesskey | base64 --decode
SECRET_KEY=kubectl get secret mlpipeline-minio-artifact -oyaml | yq .data.secretkey | base64 --decode
```

## Upload datasets to MinIO

## Setting up MinIO

You can set up the required MinIO structure either through the UI or using the MinIO Client (mc).

### Using the MinIO UI

1. Access the MinIO UI at: http://localhost:9000
   - Username: `minio`
   - Password: `minio123`

2. Create the following directory structure and upload the files:
```
mlpipeline
└── artifacts
    └── feature_repo
        ├── data
        │   └── input
        │       ├── raw_transaction_datasource.csv
        │       ├── test.csv
        │       ├── train.csv
        │       └── validate.csv
        └── feature_store.yaml
```

## Setup

```shell
./setup.sh
```

## Upload the pipeline

```shell
kfp pipeline upload -p fraud-detection-e2e fraud-detection-e2e.yaml
```

## Port-forward the inference pod

```shell
kubectl -n kubeflow get pods -l serving.kserve.io/inferenceservice=fraud-detection -o jsonpath="{.items[0].metadata.name}" | xargs -I {} kubectl port-forward -n kubeflow pod/{} 8081:8080
```

### Run a Test Request

With the port-forward active, test the deployed model:

```shell
curl -i -X POST http://localhost:8081/v1/models/onnx-model:predict -H "Content-Type: application/json" -d '{"instances": [[50.0, 5.0, 0.0, 0.0, 1.0]]}'
```

Expected output:

```
HTTP/1.1 200 OK
date: Mon, 07 Apr 2025 21:34:03 GMT
server: uvicorn
content-length: 38
content-type: application/json

{"predictions":[[0.9998821020126343]]}
```

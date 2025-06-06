This is a work in progress.

## Pipeline

![pipeline.png](pipeline.png)

## How to navigate the code

The main entry point for understanding this project is [fraud-detection-e2e.py](pipeline/fraud-detection-e2e.py).

The container image for the pipeline is defined in the `Containerfile` located in the same directory as the pipeline definition.

Other images used in the pipeline have their code in corresponding directories. For example, the `DATA_PREPARATION_IMAGE` is defined in the [data_preparation](data_preparation) directory, and similarly for other components.

## Create a Kind cluster

```shell
kind create cluster -n fraud-detection-e2e-demo --image kindest/node:v1.31.6
```

## Deploy Kubeflow Pipelines

https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/

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

## Run the pipeline

## Port-forward the inference pod

```shell
kubectl -n kubeflow get pods -l serving.kserve.io/inferenceservice=fraud-detection -o jsonpath="{.items[0].metadata.name}" | xargs -I {} kubectl port-forward -n kubeflow pod/{} 8081:8080
```

### Run Test Requests

With the port-forward active, test the deployed model:

#### Example 1: Fraudulent Transaction

```shell
curl -i -X POST http://localhost:8081/v1/models/onnx-model:predict -H "Content-Type: application/json" -d '{"instances": [[50.0, 5.0, 0.0, 0.0, 1.0]]}'
```

The input values in the request represent the following features:
1. `50.0` - distance_from_last_transaction: Distance in miles from the last transaction location
2. `5.0` - ratio_to_median_purchase_price: Ratio of the current purchase price to the median purchase price
3. `0.0` - used_chip: Whether the transaction used a chip (0.0 = No, 1.0 = Yes)
4. `0.0` - used_pin_number: Whether the transaction used a PIN number (0.0 = No, 1.0 = Yes)
5. `1.0` - online_order: Whether the transaction was an online order (0.0 = No, 1.0 = Yes)

Expected output:

```
HTTP/1.1 200 OK
date: Mon, 07 Apr 2025 21:34:03 GMT
server: uvicorn
content-length: 38
content-type: application/json

{"predictions":[[0.9998821020126343]]}
```

The prediction value close to 1.0 indicates a high probability of fraud for the given transaction.

#### Example 2: Non-Fraudulent Transaction

```shell
curl -i -X POST http://localhost:8081/v1/models/onnx-model:predict -H "Content-Type: application/json" -d '{"instances": [[5.0, 1.0, 1.0, 1.0, 0.0]]}'
```

The input values in this request represent a likely non-fraudulent transaction:
1. `5.0` - distance_from_last_transaction: Shorter distance from the last transaction location
2. `1.0` - ratio_to_median_purchase_price: Purchase price close to the median (typical spending pattern)
3. `1.0` - used_chip: Transaction used a chip (more secure)
4. `1.0` - used_pin_number: Transaction used a PIN number (more secure)
5. `0.0` - online_order: In-person transaction (not an online order)

Expected output would show a prediction value close to 0.0, indicating a low probability of fraud.

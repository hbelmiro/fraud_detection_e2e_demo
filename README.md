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

* Access the MinIO UI at: http://localhost:9000

    Username: `minio`
    Password: `minio123`

* Create the following directory structure in `mlpipeline/artifacts/` and upload the files from this repository.

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

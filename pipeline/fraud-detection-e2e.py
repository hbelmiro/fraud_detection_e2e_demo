from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Input, Dataset, Output, Model

PIPELINE_IMAGE = "quay.io/hbelmiro/fraud-detection-e2e-demo-pipeline:latest"
FEATURE_ENGINEERING_IMAGE = "quay.io/hbelmiro/fraud-detection-e2e-demo-feature-engineering:latest"
TRAIN_IMAGE = "quay.io/hbelmiro/fraud-detection-e2e-demo-train:latest"


@dsl.container_component
def create_features(features_ok: dsl.OutputPath(str)):
    return dsl.ContainerSpec(image=FEATURE_ENGINEERING_IMAGE,
                             command=["sh", "-c", "python /app/feast_apply.py --feature-repo-path=/app/feature_repo"],
                             args=[features_ok])


@dsl.component(base_image=PIPELINE_IMAGE)
def retrieve_features(features_ok: str, output_df: Output[Dataset]):
    import logging
    from feast import FeatureStore
    import pandas as pd
    import os
    from minio import Minio

    logging.basicConfig(
        level=logging.INFO,
    )

    minio_endpoint = os.getenv("MINIO_ENDPOINT")
    minio_access_key = os.getenv("MINIO_ACCESS_KEY")
    minio_secret_key = os.getenv("MINIO_SECRET_KEY")
    minio_bucket = os.getenv("MINIO_BUCKET")
    remote_feature_repo_dir = os.getenv("REMOTE_FEATURE_REPO_DIR")

    if not all([minio_endpoint, minio_access_key, minio_secret_key, minio_bucket, remote_feature_repo_dir]):
        raise ValueError("Missing required environment variables!")

    if features_ok != "true":
        raise ValueError("features not ok")

    def fetch_historical_features_entity_df(features_repo: str):
        store = FeatureStore(repo_path=features_repo)
        entity_df = pd.read_csv(os.path.join(features_repo, "data", "output", "entity.csv"))
        entity_df["created_timestamp"] = pd.to_datetime("now", utc=True)
        entity_df["event_timestamp"] = pd.to_datetime("now", utc=True)

        features = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "transactions:distance_from_home",
                "transactions:distance_from_last_transaction",
                "transactions:ratio_to_median_purchase_price",
                "transactions:fraud",
                "transactions:used_chip",
                "transactions:used_pin_number",
                "transactions:online_order",
                "transactions:set"
            ],
        ).to_df()

        return features

    def download_artifacts(directory_path, dest):
        logging.info("directory_path: {}".format(directory_path))
        logging.info("dest: {}".format(dest))

        client = Minio(
            minio_endpoint.replace("http://", "").replace("https://", ""),
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=minio_endpoint.startswith("https://")
        )

        os.makedirs(dest, exist_ok=True)

        objects = client.list_objects(minio_bucket, prefix=directory_path, recursive=True)

        for obj in objects:
            file_path = os.path.join(dest, obj.object_name.replace(directory_path, "").lstrip("/"))
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            print(f"Downloading: {obj.object_name} -> {file_path}")
            client.fget_object(minio_bucket, obj.object_name, file_path)

        print("Download complete.")

    local_dest = "feature_repo"

    download_artifacts(remote_feature_repo_dir, local_dest)

    fetch_historical_features_entity_df(local_dest).to_csv(output_df.path, index=False)


@dsl.container_component
def train_model(dataset: Input[Dataset], model: Output[Model]):
    return dsl.ContainerSpec(image=TRAIN_IMAGE,
                             command=["python", "/app/train.py"],
                             args=["--training-data-url", dataset.uri, "--model", model.path])


@dsl.component(base_image=PIPELINE_IMAGE)
def register_model(model: Input[Model]) -> NamedTuple('outputs', model_name=str, model_version=str):
    from model_registry import ModelRegistry

    print("uri: " + model.uri)

    registry = ModelRegistry(
        server_address="http://model-registry-service.kubeflow.svc.cluster.local",
        port=8080,
        author="fraud-detection-e2e-pipeline",
        user_token="non-used",  # Just to avoid a warning
        is_secure=False
    )

    model_name = "fraud-detection"
    model_version = "{{workflow.uid}}"

    registry.register_model(
        name=model_name,
        uri=model.uri,
        version=model_version,
        description="lorem ipsum",
        model_format_name="onnx",
        model_format_version="1",
        storage_key="mlpipeline-minio-artifact",
        metadata={
            # can be one of the following types
            "int_key": 1,
            "bool_key": False,
            "float_key": 3.14,
            "str_key": "str_value",
        }
    )

    outputs = NamedTuple('outputs', model_name=str, model_version=str)
    return outputs(model_name, model_version)


@dsl.component(base_image=PIPELINE_IMAGE)
def serve(model_name: str, model_version_name: str):
    import logging
    import kserve
    from kubernetes import client
    from kubernetes.client import V1Container
    from model_registry import ModelRegistry

    logging.info("serving model: {}".format(model_name))
    logging.info("model_version: {}".format(model_version_name))

    registry = ModelRegistry(
        server_address="http://model-registry-service.kubeflow.svc.cluster.local",
        port=8080,
        author="fraud-detection-e2e-pipeline",
        user_token="non-used",  # Just to avoid a warning
        is_secure=False
    )

    model = registry.get_registered_model(model_name)
    model_version = registry.get_model_version(model_name, model_version_name)

    inference_service = kserve.V1beta1InferenceService(
        api_version=kserve.constants.KSERVE_GROUP + "/v1beta1",
        kind="InferenceService",
        metadata=client.V1ObjectMeta(
            name=model_name,
            namespace=kserve.utils.get_default_target_namespace(),
            labels={
                "modelregistry/registered-model-id": model.id,
                "modelregistry/model-version-id": model_version.id
            },
        ),
        spec=kserve.V1beta1InferenceServiceSpec(
            predictor=kserve.V1beta1PredictorSpec(
                service_account_name="kserve-sa",
                containers=[
                    V1Container(
                        name="inference-container",
                        image="quay.io/hbelmiro/fraud-detection-e2e-demo-rest-predictor:latest",
                        command=["python", "predictor.py"],
                        args=["--model-name", model_name, "--model-version", model_version_name]
                    )
                ]
            )
        ),
    )
    ks_client = kserve.KServeClient()
    ks_client.create(inference_service)


@dsl.pipeline
def fraud_detection_e2e_pipeline():
    create_features_task = create_features()
    create_features_task.set_caching_options(False)

    retrieve_features_task = retrieve_features(features_ok=create_features_task.outputs['features_ok'])
    retrieve_features_task.set_caching_options(False)

    train_model_task = train_model(dataset=retrieve_features_task.outputs['output_df'])
    train_model_task.set_caching_options(False)

    register_model_task = register_model(model=train_model_task.outputs['model'])
    register_model_task.set_caching_options(False)

    serve_task = serve(model_name=register_model_task.outputs["model_name"],
                       model_version_name=register_model_task.outputs["model_version"])
    serve_task.set_caching_options(False)

import os
from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Input, Dataset, Output, Model

TAG = "feast-operator-1752594458"

PIPELINE_IMAGE = os.getenv("PIPELINE_IMAGE", "quay.io/hbelmiro/fraud-detection-e2e-demo-pipeline-rhoai:" + TAG)
FEATURE_ENGINEERING_IMAGE = os.getenv("FEATURE_ENGINEERING_IMAGE",
                                      "quay.io/hbelmiro/fraud-detection-e2e-demo-feature-engineering-rhoai:" + TAG)
TRAIN_IMAGE = os.getenv("TRAIN_IMAGE", "quay.io/hbelmiro/fraud-detection-e2e-demo-train-rhoai:" + TAG)
DATA_PREPARATION_IMAGE = os.getenv("DATA_PREPARATION_IMAGE",
                                   "quay.io/hbelmiro/fraud-detection-e2e-demo-data-preparation-rhoai:" + TAG)
REST_PREDICTOR_IMAGE = os.getenv("REST_PREDICTOR_IMAGE",
                                 "quay.io/hbelmiro/fraud-detection-e2e-demo-rest-predictor-rhoai:" + TAG)


@dsl.component(base_image=PIPELINE_IMAGE)
def prepare_data(job_id: str, data_preparation_image: str) -> bool:
    from kubernetes import client, config
    from datetime import datetime, timedelta
    import time
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    config.load_incluster_config()

    spark_app = {
        "apiVersion": "sparkoperator.k8s.io/v1beta2",
        "kind": "SparkApplication",
        "metadata": {
            "name": f"data-preparation-{job_id}",
            "namespace": "fraud-detection"
        },
        "spec": {
            "type": "Python",
            "pythonVersion": "3",
            "mode": "cluster",
            "image": data_preparation_image,
            "imagePullPolicy": "Always",
            "mainApplicationFile": "local:///app/main.py",
            "sparkVersion": "3.5.5",
            "restartPolicy": {
                "type": "Never"
            },
            "driver": {
                "serviceAccount": "spark",
                "cores": 1,
                "coreLimit": "1200m",
                "memory": "1g"
            },
            "executor": {
                "cores": 1,
                "instances": 1,
                "memory": "1g"
            }
        }
    }

    k8s_client = client.CustomObjectsApi()
    k8s_client.create_namespaced_custom_object(
        group="sparkoperator.k8s.io",
        version="v1beta2",
        namespace=spark_app["metadata"]["namespace"],
        plural="sparkapplications",
        body=spark_app
    )

    job_name = spark_app["metadata"]["name"]
    namespace = spark_app["metadata"]["namespace"]

    timeout_minutes = 5
    start_time = datetime.now()
    timeout_time = start_time + timedelta(minutes=timeout_minutes)

    while True:
        if datetime.now() > timeout_time:
            logger.error(f"Timeout reached after {timeout_minutes} minutes waiting for Spark job completion")
            return False

        status = k8s_client.get_namespaced_custom_object_status(
            group="sparkoperator.k8s.io",
            version="v1beta2",
            namespace=namespace,
            plural="sparkapplications",
            name=job_name
        )

        application_state = status.get("status", {}).get("applicationState", {}).get("state", "")
        logger.info(f"Current application state: {application_state}")

        if application_state == "COMPLETED":
            logger.info("Spark job completed successfully")
            return True
        elif application_state in ["FAILED", "SUBMISSION_FAILED", "INVALIDATING"]:
            logger.error(f"Spark job failed with state: {application_state}")
            return False

        time.sleep(10)


@dsl.container_component
def feature_engineering(data_preparation_ok: bool, features_ok: dsl.OutputPath(str)):
    if not data_preparation_ok:
        raise ValueError("data preparation not ok")

    return dsl.ContainerSpec(image=FEATURE_ENGINEERING_IMAGE,
                             command=["sh", "-c", "python /app/feast_feature_engineering.py --feature-repo-path=/app/feature_repo"],
                             args=[features_ok])


@dsl.component(base_image=PIPELINE_IMAGE)
def retrieve_features(features_ok: str, output_df: Output[Dataset]):
    import logging
    from feast import FeatureStore
    import pandas as pd
    import os
    from minio import Minio
    import glob
    from pyspark.sql import SparkSession

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

    def fetch_historical_features_entity_df() -> pd.DataFrame:
        features_repo = "/app/feature_repo/"
        store = FeatureStore(repo_path=features_repo)

        directory_path = os.path.join(features_repo, "data", "output", "entity.csv")
        csv_files = glob.glob(os.path.join(directory_path, 'part-*.csv'))
        df_list = [pd.read_csv(file) for file in csv_files]

        entity_df = pd.concat(df_list, ignore_index=True)
        entity_df["created_timestamp"] = pd.to_datetime("now", utc=True)
        entity_df["event_timestamp"] = pd.to_datetime("now", utc=True)

        _features = store.get_historical_features(
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

        return _features

    logging.basicConfig(
        level=logging.INFO,
    )

    logging.info("output_df.path: {}".format(output_df.path))

    minio_endpoint = "http://minio-service.fraud-detection.svc.cluster.local:9000"
    minio_access_key = "minio"
    minio_secret_key = "minio123"
    minio_bucket = "mlpipeline"
    remote_feature_repo_dir = "artifacts/feature_repo"

    (
        SparkSession.builder
        .appName("FeatureEngineeringSpark")
        .config("spark.jars.ivy", "/app/.ivy2")
        .config("spark.driver.memory", "1g")
        .config("spark.executor.memory", "1g")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1,com.amazonaws:aws-java-sdk-bundle:1.12.458")
        .config("spark.hadoop.fs.s3a.endpoint", minio_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", "minio")
        .config("spark.hadoop.fs.s3a.secret.key", "minio123")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )

    if not all([minio_endpoint, minio_access_key, minio_secret_key, minio_bucket, remote_feature_repo_dir]):
        raise ValueError("Missing required environment variables!")

    if features_ok != "true":
        raise ValueError(f"features not ok [status: {features_ok}]")

    local_dest = "/app/feature_repo"

    download_artifacts(remote_feature_repo_dir, local_dest)

    fetch_historical_features_entity_df().to_csv(output_df.path, index=False)


@dsl.container_component
def train_model(dataset: Input[Dataset], model: Output[Model]):
    return dsl.ContainerSpec(image=TRAIN_IMAGE,
                             command=["python", "/app/train.py"],
                             args=["--training-data-url", dataset.uri, "--model", model.path])


@dsl.component(base_image=PIPELINE_IMAGE)
def register_model(model: Input[Model]) -> NamedTuple('outputs', model_name=str, model_version=str):
    from model_registry import ModelRegistry
    from kubernetes import client, config

    def get_model_registry_url():
        """Get the model registry URL from the Kubernetes service annotation."""
        try:
            config.load_incluster_config()
            v1 = client.CoreV1Api()
            
            service = v1.read_namespaced_service(
                name="fraud-detection",
                namespace="rhoai-model-registries"
            )
            
            annotations = service.metadata.annotations or {}
            url = annotations.get("routing.opendatahub.io/external-address-rest")
            
            if not url:
                raise ValueError("Model registry URL annotation not found")
            
            # Remove port if present
            if ':' in url:
                url = url.split(':')[0]
            
            return f"https://{url}"
        except Exception as e:
            raise Exception(f"Error getting model registry URL: {e}")

    with open("/var/run/secrets/kubernetes.io/serviceaccount/token", "r") as token_file:
        token = token_file.read()

    model_registry_url = get_model_registry_url()
    print(f"Using model registry URL: {model_registry_url}")

    registry = ModelRegistry(
        server_address=model_registry_url,
        author="fraud-detection-e2e-pipeline",
        user_token=token,
        is_secure=False,
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
            "int_key": 1,
            "bool_key": False,
            "float_key": 3.14,
            "str_key": "str_value",
        }
    )

    outputs = NamedTuple('outputs', model_name=str, model_version=str)
    return outputs(model_name, model_version)


@dsl.component(base_image=PIPELINE_IMAGE)
def serve(model_name: str, model_version_name: str, job_id: str, rest_predictor_image: str):
    import logging
    import kserve
    from kubernetes import client, config
    from kubernetes.client import V1Container, V1EnvVar, V1SecretKeySelector
    from model_registry import ModelRegistry
    import base64

    def get_model_registry_url():
        """Get the model registry URL from the Kubernetes service annotation."""
        try:
            config.load_incluster_config()
            v1 = client.CoreV1Api()
            
            service = v1.read_namespaced_service(
                name="fraud-detection",
                namespace="rhoai-model-registries"
            )
            
            annotations = service.metadata.annotations or {}
            url = annotations.get("routing.opendatahub.io/external-address-rest")
            
            if not url:
                raise ValueError("Model registry URL annotation not found")
            
            # Remove port if present
            if ':' in url:
                url = url.split(':')[0]
            
            return f"https://{url}"
        except Exception as e:
            raise Exception(f"Error getting model registry URL: {e}")

    def create_token_secret(token: str, job_id: str) -> str:
        """Create a Kubernetes secret with the service account token and return the secret name."""
        from kubernetes import config
        config.load_incluster_config()
        v1 = client.CoreV1Api()
        secret_name = f"kserve-token-{job_id[:8]}"
        
        secret = client.V1Secret(
            metadata=client.V1ObjectMeta(
                name=secret_name,
                namespace="fraud-detection"
            ),
            type="Opaque",
            data={
                "token": base64.b64encode(token.strip().encode()).decode()
            }
        )
        
        try:
            v1.create_namespaced_secret(namespace="fraud-detection", body=secret)
            logging.info(f"Created secret {secret_name} with service account token")
        except client.rest.ApiException as e:
            if e.status == 409:  # Already exists
                v1.patch_namespaced_secret(name=secret_name, namespace="fraud-detection", body=secret)
                logging.info(f"Updated existing secret {secret_name}")
            else:
                raise
        
        return secret_name

    with open("/var/run/secrets/kubernetes.io/serviceaccount/token", "r") as token_file:
        token = token_file.read()

    model_registry_url = get_model_registry_url()
    print(f"Using model registry URL: {model_registry_url}")

    registry = ModelRegistry(
        server_address=model_registry_url,
        author="fraud-detection-e2e-pipeline",
        user_token=token,
        is_secure=False,
    )

    logging.info("serving model: {}".format(model_name))
    logging.info("model_version: {}".format(model_version_name))

    model = registry.get_registered_model(model_name)
    model_version = registry.get_model_version(model_name, model_version_name)

    token_secret_name = create_token_secret(token, job_id)

    inference_service = kserve.V1beta1InferenceService(
        api_version=kserve.constants.KSERVE_GROUP + "/v1beta1",
        kind="InferenceService",
        metadata=client.V1ObjectMeta(
            name="fd",
            namespace="fraud-detection",
            labels={
                "modelregistry/registered-model-id": model.id,
                "modelregistry/model-version-id": model_version.id
            },
            annotations={
                "sidecar.istio.io/inject": "false"
            },
        ),
        spec=kserve.V1beta1InferenceServiceSpec(
            predictor=kserve.V1beta1PredictorSpec(
                service_account_name="kserve-sa",
                containers=[
                    V1Container(
                        name="inference-container",
                        image=rest_predictor_image,
                        command=["python", "predictor.py"],
                        args=["--model-name", model_name, "--model-version", model_version_name, "--model-registry-url", model_registry_url],
                        env=[
                            V1EnvVar(
                                name="KSERVE_SERVICE_ACCOUNT_TOKEN",
                                value_from=client.V1EnvVarSource(
                                    secret_key_ref=V1SecretKeySelector(
                                        name=token_secret_name,
                                        key="token"
                                    )
                                )
                            )
                        ]
                    )
                ]
            )
        ),
    )
    ks_client = kserve.KServeClient()
    ks_client.create(inference_service)


@dsl.pipeline
def fraud_detection_e2e_pipeline():
    import kfp

    prepare_data_task = prepare_data(job_id=kfp.dsl.PIPELINE_JOB_ID_PLACEHOLDER,
                                     data_preparation_image=DATA_PREPARATION_IMAGE)

    feature_engineering_task = feature_engineering(data_preparation_ok=prepare_data_task.output)

    retrieve_features_task = retrieve_features(features_ok=feature_engineering_task.output)

    train_model_task = train_model(dataset=retrieve_features_task.outputs['output_df'])

    register_model_task = register_model(model=train_model_task.outputs['model'])

    serve(model_name=register_model_task.outputs["model_name"],
          model_version_name=register_model_task.outputs["model_version"],
          job_id=kfp.dsl.PIPELINE_JOB_ID_PLACEHOLDER,
          rest_predictor_image=REST_PREDICTOR_IMAGE)

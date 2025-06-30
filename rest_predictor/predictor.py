import argparse
import os
from typing import Dict
from urllib.parse import urlparse

import boto3
import kserve
import numpy as np
import onnxruntime as ort
from botocore.client import Config
from feast import FeatureStore
from kserve import InferRequest
from kserve.protocol.grpc.grpc_predict_v2_pb2 import ModelInferRequest
from model_registry import ModelRegistry


def download(artifact_uri):
    minio_endpoint = "http://minio-service.fraud-detection.svc.cluster.local:9000"
    access_key = "minio"
    secret_key = "minio123"

    parsed_uri = urlparse(artifact_uri)
    bucket_name = parsed_uri.netloc
    object_key = parsed_uri.path.lstrip('/')

    s3_client = boto3.client(
        's3',
        endpoint_url=minio_endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4')
    )

    local_path = "/app/model"
    s3_client.download_file(bucket_name, object_key, local_path)


def download_feature_repo():
    minio_endpoint = "http://minio-service.fraud-detection.svc.cluster.local:9000"
    access_key = "minio"
    secret_key = "minio123"
    bucket_name = "mlpipeline"
    remote_feature_repo_dir = "artifacts/feature_repo"
    local_dest = "/app/feature_repo"

    s3_client = boto3.client(
        's3',
        endpoint_url=minio_endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4')
    )

    # Create local destination directory
    os.makedirs(local_dest, exist_ok=True)

    # List objects in the remote directory
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=remote_feature_repo_dir
    )

    # Download each object
    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key']
            file_path = os.path.join(local_dest, key.replace(remote_feature_repo_dir, "").lstrip("/"))
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            s3_client.download_file(bucket_name, key, file_path)


class ONNXModel(kserve.Model):
    def __init__(self, name: str, model_name: str, model_version_name: str):
        super().__init__(name)
        self.model_name = model_name
        self.model_version_name = model_version_name
        self.model = None
        self.feature_store = None
        self.ready = False
        self.load()

    def load(self):
        # Download the model from the registry
        registry = ModelRegistry(
            server_address="http://model-registry-service.fraud-detection.svc.cluster.local",
            port=8080,
            author="fraud-detection-e2e-pipeline",
            user_token="non-used",  # Just to avoid a warning
            is_secure=False
        )

        model_artifact = registry.get_model_artifact(self.model_name, self.model_version_name)
        download(model_artifact.uri)
        self.model = ort.InferenceSession("/app/model")

        # Download and initialize the feature store
        download_feature_repo()

        # Check if the feature repository was downloaded correctly
        feature_repo_path = "/app/feature_repo"
        print(f"Checking contents of feature repository at {feature_repo_path}")
        if os.path.exists(feature_repo_path):
            print("Feature repository directory exists")
            for root, dirs, files in os.walk(feature_repo_path):
                for file in files:
                    print(f"  {os.path.join(root, file)}")
        else:
            raise RuntimeError("ERROR: Feature repository directory does not exist")

        # Check specifically for the online store database
        # The path should match what's in feature_store.yaml
        online_store_db = "/app/feature_repo/data/output/online_store.db"
        if os.path.exists(online_store_db):
            print(f"Online store database exists at {online_store_db}")
            # Check file size to ensure it's not empty
            size = os.path.getsize(online_store_db)
            print(f"Online store database size: {size} bytes")
        else:
            raise RuntimeError(f"WARNING: Online store database not found at {online_store_db}")

        self.feature_store = FeatureStore(repo_path=feature_repo_path)

        self.ready = True

    async def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        try:
            user_id = payload.get("user_id")
            if not user_id:
                return {"error": "Payload must contain 'user_id' field"}

            features_to_request = [
                "transactions:distance_from_last_transaction",
                "transactions:ratio_to_median_purchase_price",
                "transactions:used_chip",
                "transactions:used_pin_number",
                "transactions:online_order",
            ]

            feature_dict = self.feature_store.get_online_features(
                entity_rows=[{"user_id": user_id}],
                features=features_to_request,
            ).to_dict()

            print(f"Features received from Feast: {feature_dict}")

            try:
                input_data = np.array([[
                    feature_dict["distance_from_last_transaction"][0],
                    feature_dict["ratio_to_median_purchase_price"][0],
                    feature_dict["used_chip"][0],
                    feature_dict["used_pin_number"][0],
                    feature_dict["online_order"][0],
                ]], dtype=np.float32)
            except (KeyError, IndexError) as e:
                return {"error": f"Feature not found or absent values for user '{user_id}': {e}"}

            input_name = self.model.get_inputs()[0].name
            result = self.model.run(None, {input_name: input_data})

            prediction = result[0].tolist()

            return {"user_id": user_id, "prediction": prediction}

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"An unexpected error occurred: {str(e)}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="Name of the model to serve")
    parser.add_argument("--model-version", required=True, help="Version of the model to serve")
    args = parser.parse_args()

    model = ONNXModel("onnx-model", args.model_name, args.model_version)
    kserve.ModelServer().start([model])

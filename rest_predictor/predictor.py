import argparse
from typing import Dict, Union
from urllib.parse import urlparse

import boto3
import kserve
import numpy as np
import onnxruntime as ort
from botocore.client import Config
from kserve import InferRequest
from kserve.protocol.grpc.grpc_predict_v2_pb2 import ModelInferRequest
from model_registry import ModelRegistry


def download(artifact_uri):
    minio_endpoint = "http://minio-service.kubeflow.svc.cluster.local:9000"
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


class ONNXModel(kserve.Model):
    def __init__(self, name: str, model_name: str, model_version_name: str):
        super().__init__(name)
        self.model_name = model_name
        self.model_version_name = model_version_name
        self.model = None
        self.ready = False
        self.load()

    def load(self):
        registry = ModelRegistry(
            server_address="http://model-registry-service.kubeflow.svc.cluster.local",
            port=8080,
            author="fraud-detection-e2e-pipeline",
            user_token="non-used",  # Just to avoid a warning
            is_secure=False
        )

        model_artifact = registry.get_model_artifact(self.model_name, self.model_version_name)

        download(model_artifact.uri)

        self.model = ort.InferenceSession("/app/model")
        self.ready = True

    async def predict(
            self,
            payload: Union[Dict, InferRequest, ModelInferRequest],
            headers: Dict[str, str] = None,
            response_headers: Dict[str, str] = None,
    ) -> Dict:  # Simplified return type to Dict since that's what we return
        try:
            # Handle payload as a dict (assuming REST input)
            if isinstance(payload, dict):
                input_data = payload["instances"]
            else:
                # Handle InferRequest or ModelInferRequest if needed
                raise ValueError("Unsupported payload type; expected dict")

            input_data = np.array(input_data, dtype=np.float32)
            input_name = self.model.get_inputs()[0].name
            output = self.model.run(None, {input_name: input_data})
            predictions = output[0].tolist()
            return {"predictions": predictions}
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="Name of the model to serve")
    parser.add_argument("--model-version", required=True, help="Version of the model to serve")
    args = parser.parse_args()

    model = ONNXModel("onnx-model", args.model_name, args.model_version)
    kserve.ModelServer().start([model])

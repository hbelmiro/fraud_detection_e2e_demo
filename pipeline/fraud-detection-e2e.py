from kfp import dsl
from kfp.dsl import Input, Dataset, Output


@dsl.container_component
def create_features(features_ok: dsl.OutputPath(str)):
    return dsl.ContainerSpec(image="quay.io/hbelmiro/fraud-detection-e2e-demo-feast:dev-17404879823N",
                             command=["sh", "-c", "python /app/feast_apply.py --feature-repo-path=/app/feature_repo"],
                             args=[features_ok])


@dsl.component(base_image="quay.io/hbelmiro/fraud-detection-e2e-demo-retrieve-features:dev-17385897883N")
def retrieve_features(features_ok: str, output_df: Output[Dataset]):
    if features_ok != "true":
        raise ValueError("features not ok")

    import os

    def fetch_historical_features_entity_df(features_repo: str):
        from feast import FeatureStore
        import pandas as pd
        import os

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
        import os
        from minio import Minio

        minio_endpoint = os.getenv("MINIO_ENDPOINT")
        minio_access_key = os.getenv("MINIO_ACCESS_KEY")
        minio_secret_key = os.getenv("MINIO_SECRET_KEY")
        minio_bucket = os.getenv("MINIO_BUCKET")

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

    remote_feature_repo_dir = os.getenv("REMOTE_FEATURE_REPO_DIR")
    local_dest = "feature_repo"

    download_artifacts(remote_feature_repo_dir, local_dest)

    fetch_historical_features_entity_df(local_dest).to_csv(output_df.path, index=False)


@dsl.component(base_image="python:3.11")
def read_dataset(dataset: Input[Dataset]):
    with open(dataset.path, "r") as f:
        data = f.read()
    print("Dataset content:", data)


@dsl.component(base_image="python:3.11")
def fail(message: Input[str]):
    raise ValueError(message)


@dsl.pipeline
def fraud_detection_e2e_pipeline():
    create_features_task = create_features()
    create_features_task.set_caching_options(False)

    retrieve_features_task = retrieve_features(features_ok=create_features_task.outputs['features_ok'])

    read_dataset(dataset=retrieve_features_task.outputs['output_df'])

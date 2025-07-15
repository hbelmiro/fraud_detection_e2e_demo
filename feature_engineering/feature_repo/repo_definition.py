import os
from datetime import datetime, timedelta

import feast.types
from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    Project,
    ValueType,
    FeatureStore
)
from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.contrib.spark_offline_store.spark_source import SparkSource
from feast.infra.offline_stores.file_source import FileLoggingDestination

from minio import Minio

MINIO_BUCKET = "mlpipeline"
MINIO_ENDPOINT = "http://minio-service.fraud-detection.svc.cluster.local:9000"
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minio123"

LOCAL_DATA_DIR = "data/"
LOCAL_INPUT_DIR = LOCAL_DATA_DIR + "input/"
LOCAL_OUTPUT_DIR = LOCAL_DATA_DIR + "output/"

REMOTE_FEATURE_REPO_DIR = "artifacts/feature_repo/"
REMOTE_DATA_DIR = REMOTE_FEATURE_REPO_DIR + "data/"
REMOTE_INPUT_DIR = REMOTE_DATA_DIR + "input/"
REMOTE_OUTPUT_DIR = REMOTE_DATA_DIR + "output/"


def download_artifacts(directory_path, dest):
    client = Minio(
        MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_ENDPOINT.startswith("https://")
    )

    os.makedirs(dest, exist_ok=True)

    objects = client.list_objects(MINIO_BUCKET, prefix=directory_path, recursive=True)

    for obj in objects:
        if obj.object_name.endswith('/'):
            print(f"Creating directory: {obj.object_name}")
            dir_path = os.path.join(dest, obj.object_name.replace(directory_path, "").lstrip("/"))
            os.makedirs(dir_path, exist_ok=True)
            continue

        file_path = os.path.join(dest, obj.object_name.replace(directory_path, "").lstrip("/"))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        print(f"Downloading: {obj.object_name} -> {file_path}")
        client.fget_object(MINIO_BUCKET, obj.object_name, file_path)

    print("Download complete.")


def move_parquet_files_from_temp():
    """Move parquet files from temporary locations to their final expected locations."""
    output_dir = "data/output"

    # Look for parquet files in temporary directories
    temp_parquet_dir = os.path.join(output_dir, "features.parquet", "_temporary")
    if os.path.exists(temp_parquet_dir):
        print("Found temporary parquet directory, moving files...")

        # Find all parquet files in temporary directories
        parquet_files = []
        for root, dirs, files in os.walk(temp_parquet_dir):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, file))

        if parquet_files:
            # Create final parquet directory
            final_parquet_dir = os.path.join(output_dir, "features.parquet")
            os.makedirs(final_parquet_dir, exist_ok=True)

            # Move parquet files to final location
            for i, parquet_file in enumerate(parquet_files):
                final_file = os.path.join(final_parquet_dir, f"part-{i:05d}.parquet")
                print(f"Moving {parquet_file} -> {final_file}")
                os.rename(parquet_file, final_file)

            print(f"Moved {len(parquet_files)} parquet files to final location")
        else:
            print("No parquet files found in temporary directory")
    else:
        print("No temporary parquet directory found")

project = Project(name="fraud_detection_e2e_demo")
transaction = Entity(name="transaction", join_keys=["user_id"], value_type=ValueType.STRING)

parquet_file_name = LOCAL_OUTPUT_DIR + "features.parquet"

transaction_source = SparkSource(
    name="transactions_source",
    path=parquet_file_name,
    file_format="parquet",
    timestamp_field="updated",
    created_timestamp_column="created",
)

transactions_fv = FeatureView(
    name="transactions",
    entities=[transaction],
    ttl=timedelta(days=1),
    schema=[
        Field(name="user_id", dtype=feast.types.String),
        Field(name="fraud", dtype=feast.types.Bool),
        Field(name="created", dtype=feast.types.String),
        Field(name="updated", dtype=feast.types.String),
        Field(name="set", dtype=feast.types.String),
        Field(name="distance_from_home", dtype=feast.types.Float32),
        Field(name="distance_from_last_transaction", dtype=feast.types.Float32),
        Field(name="ratio_to_median_purchase_price", dtype=feast.types.Float32),
        Field(name="num_prev_transactions", dtype=feast.types.Float32),
        Field(name="avg_prev_transaction_amount", dtype=feast.types.Float32),
        Field(name="max_prev_transaction_amount", dtype=feast.types.Float32),
        Field(name="stdv_prev_transaction_amount", dtype=feast.types.Float32),
        Field(name="days_since_last_transaction", dtype=feast.types.Float32),
        Field(name="days_since_first_transaction", dtype=feast.types.Float32),
        Field(name="used_chip", dtype=feast.types.Bool),
        Field(name="used_pin_number", dtype=feast.types.Bool),
        Field(name="online_order", dtype=feast.types.Bool),
    ],
    online=True,
    source=transaction_source,
)

transactions_fs = FeatureService(
    name="transactions_fs",
    features=[
        transactions_fv
    ],
    logging_config=LoggingConfig(
        destination=FileLoggingDestination(path="data")
    ),
)

local_data_dir = "data/"
local_input_dir = local_data_dir + "input/"
local_output_dir = local_data_dir + "output/"

download_artifacts(REMOTE_INPUT_DIR, local_input_dir)
download_artifacts(REMOTE_OUTPUT_DIR, local_output_dir)

move_parquet_files_from_temp()

fs = FeatureStore(repo_path=".")
fs.apply([transactions_fs])

end_date = datetime.utcnow().isoformat()
start_date = (datetime.utcnow() - timedelta(days=365)).isoformat()

fs.materialize(start_date, end_date)

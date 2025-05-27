from datetime import timedelta

import feast.types
from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    Project,
    ValueType,
)
from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.contrib.spark_offline_store.spark_source import SparkSource
from feast.infra.offline_stores.file_source import FileLoggingDestination

MINIO_BUCKET = "mlpipeline"

LOCAL_DATA_DIR = "/app/feature_repo/data/"
LOCAL_INPUT_DIR = LOCAL_DATA_DIR + "input/"
LOCAL_OUTPUT_DIR = LOCAL_DATA_DIR + "output/"

# Define a project for the feature repo
project = Project(name="fraud_detection_e2e_demo")
# Define an entity for the driver. You can think of an entity as a primary key used to
# fetch features.
transaction = Entity(name="transaction", join_keys=["user_id"], value_type=ValueType.STRING)

parquet_file_name = LOCAL_OUTPUT_DIR + "features.parquet"

transaction_source = SparkSource(
    name="transactions_source",
    path=parquet_file_name,
    file_format="parquet",
    timestamp_field="updated",
    created_timestamp_column="created",
)

# Our parquet files contain sample data that includes a driver_id column, timestamps and
# three feature column. Here we define a Feature View that will allow us to serve this
# data to our model online.
transactions_fv = FeatureView(
    # The unique name of this feature view. Two feature views in a single
    # project cannot have the same name
    name="transactions",
    entities=[transaction],
    ttl=timedelta(days=1),
    # The list of features defined below act as a schema to both define features
    # for both materialization of features into a store, and are used as references
    # during retrieval for building a training dataset or serving features
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
    # Tags are user defined key/value pairs that are attached to each
    # feature view
    tags={"team": "driver_performance"},
)

# This groups features into a model version
transactions_fs = FeatureService(
    name="transactions_fs",
    features=[
        transactions_fv
    ],
    logging_config=LoggingConfig(
        destination=FileLoggingDestination(path="data")
    ),
)

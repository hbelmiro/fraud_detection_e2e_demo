from datetime import timedelta

import feast.types
from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    Project,
    ValueType
)
from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.contrib.spark_offline_store.spark_source import SparkSource
from feast.infra.offline_stores.file_source import FileLoggingDestination

LOCAL_DATA_DIR = "data/"
LOCAL_INPUT_DIR = LOCAL_DATA_DIR + "input/"
LOCAL_OUTPUT_DIR = LOCAL_DATA_DIR + "output/"


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

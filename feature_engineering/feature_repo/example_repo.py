# This is an example feature definition file
import os
from datetime import timedelta

import feast.types
import pandas as pd
from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    Project,
    PushSource,
    RequestSource,
)
from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.file_source import FileLoggingDestination

from feature_engineering import get_features, data_dir


features = get_features()
features_file_name = os.path.join(data_dir, "features.csv")
features.to_csv(features_file_name)

# Define a project for the feature repo
project = Project(name="fraud_detection_e2e_demo", description="A project for driver statistics")
# Define an entity for the driver. You can think of an entity as a primary key used to
# fetch features.
transaction = Entity(name="transaction", join_keys=["user_id"])
df = pd.read_csv(features_file_name)
df["created"] = pd.to_datetime(df["created"], errors="coerce", utc=True)
df["updated"] = pd.to_datetime(df["updated"], errors="coerce", utc=True)

parquet_file_name = features_file_name.replace(".csv", ".parquet")

df.to_parquet(parquet_file_name)
# Read data from parquet files. Parquet is convenient for local development mode. For
# production, you can use your favorite DWH, such as BigQuery. See Feast documentation
# for more info.
transaction_source = FileSource(
    name="transaction_stats_source",
    path=parquet_file_name,
    timestamp_field="created",
    created_timestamp_column="updated",
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
        Field(name="fraud", dtype=feast.types.Float32),
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
# Define a request data source which encodes features / information only
# available at request time (e.g. part of the user initiated HTTP request)
input_request = RequestSource(
    name="vals_to_add",
    schema=[
        Field(name="val_to_add", dtype=feast.types.Int64),
        Field(name="val_to_add_2", dtype=feast.types.Int64),
    ],
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
# Defines a way to push data (to be available offline, online or both) into Feast.
transactions_push_source = PushSource(
    name="transactions_push_source",
    batch_source=transaction_source,
)
# Defines a slightly modified version of the feature view from above, where the source
# has been changed to the push source. This allows fresh features to be directly pushed
# to the online store for this feature view.
transactions_fresh_fv = FeatureView(
    name="transactions_fresh",
    entities=[transaction],
    ttl=timedelta(days=1),
    online=True,
    source=transactions_push_source,  # Changed from above
    tags={"team": "driver_performance"},
)


# This is an example feature definition file

from datetime import timedelta

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
from feast.types import Float32, Int64, Bool

# Define a project for the feature repo
project = Project(name="fraud_detection_e2e_demo", description="A project for driver statistics")

# Define an entity for the driver. You can think of an entity as a primary key used to
# fetch features.
transaction = Entity(name="transaction", join_keys=["user_id"])

df = pd.read_csv("data/final_data.csv")

df["created"] = pd.to_datetime(df["created"], errors="coerce", utc=True)
df["updated"] = pd.to_datetime(df["updated"], errors="coerce", utc=True)

df.to_parquet('data/final_data.parquet')

# Read data from parquet files. Parquet is convenient for local development mode. For
# production, you can use your favorite DWH, such as BigQuery. See Feast documentation
# for more info.
transaction_source = FileSource(
    name="transaction_stats_source",
    path="data/final_data.parquet",
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
        Field(name="distance_from_home", dtype=Float32),
        Field(name="distance_from_last_transaction", dtype=Float32),
        Field(name="ratio_to_median_purchase_price", dtype=Float32),
        Field(name="repeat_retailer", dtype=Bool),
        Field(name="used_chip", dtype=Bool),
        Field(name="used_pin_number", dtype=Bool),
        Field(name="online_order", dtype=Bool),
        Field(name="fraud", dtype=Bool),

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
        Field(name="val_to_add", dtype=Int64),
        Field(name="val_to_add_2", dtype=Int64),
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

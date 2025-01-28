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
from pandas import Series

DATA_DIR = "data"


def calculate_point_in_time_features(label_dataset, transactions_df) -> pd.DataFrame:
    label_dataset["created"] = pd.to_datetime(label_dataset["created"])
    transactions_df["transaction_timestamp"] = pd.to_datetime(
        transactions_df["date_of_transaction"]
    )

    # Get all transactions before the created time
    transactions_before = pd.merge(
        label_dataset[["user_id", "created"]], transactions_df, on="user_id"
    )
    transactions_before = transactions_before[
        transactions_before["transaction_timestamp"] < transactions_before["created_x"]
        ]
    transactions_before["days_between_transactions"] = (
            transactions_before["transaction_timestamp"] - transactions_before["created_x"]
    ).dt.days

    # Group by user_id and created to calculate features
    features_df: pd.DataFrame = (
        transactions_before.groupby(["user_id", "created_x"])
        .agg(
            num_prev_transactions=("transaction_amount", "count"),
            avg_prev_transaction_amount=("transaction_amount", "mean"),
            max_prev_transaction_amount=("transaction_amount", "max"),
            stdv_prev_transaction_amount=("transaction_amount", "std"),
            days_since_last_transaction=("days_between_transactions", "min"),
            days_since_first_transaction=("days_between_transactions", "max"),
        )
        .reset_index()
        .fillna(0)
    )

    final_df = (
        pd.merge(
            label_dataset,
            features_df,
            left_on=["user_id", "created"],
            right_on=["user_id", "created_x"],
            how="left",
        )
        .reset_index(drop=True)
        .drop("created_x", axis=1)
    )

    return final_df


def float_to_bool(column: Series):
    return column.map({0.0: False, 1.0: True})


def get_features() -> pd.DataFrame:
    print("loading data...")

    train_set = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_set = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    validate_set = pd.read_csv(os.path.join(DATA_DIR, "validate.csv"))
    train_set["set"] = "train"
    test_set["set"] = "test"
    validate_set["set"] = "valid"

    all_sets = pd.concat([train_set, test_set, validate_set], axis=0).reset_index(drop=True)

    all_sets['fraud'] = float_to_bool(all_sets['fraud'])
    all_sets['repeat_retailer'] = float_to_bool(all_sets['repeat_retailer'])
    all_sets['used_chip'] = float_to_bool(all_sets['used_chip'])
    all_sets['used_pin_number'] = float_to_bool(all_sets['used_pin_number'])
    all_sets['online_order'] = float_to_bool(all_sets['online_order'])

    all_sets["user_id"] = [f"user_{i}" for i in range(all_sets.shape[0])]
    all_sets["transaction_id"] = [f"txn_{i}" for i in range(all_sets.shape[0])]

    for date_col in ["created", "updated"]:
        all_sets[date_col] = pd.Timestamp.now()

    label_dataset = pd.DataFrame(
        all_sets[
            [
                "user_id",
                "fraud",
                "created",
                "updated",
                "set",
                "distance_from_home",
                "distance_from_last_transaction",
                "ratio_to_median_purchase_price",
            ]
        ]
    )

    user_purchase_history = pd.read_csv(os.path.join(DATA_DIR, "raw_transaction_datasource.csv"))

    features_df = calculate_point_in_time_features(label_dataset, user_purchase_history)

    features_df = features_df.merge(
        all_sets[["user_id", "created", "used_chip", "used_pin_number", "online_order"]],
        on=["user_id", "created"],
    )

    return features_df


features_file_name = os.path.join(DATA_DIR, "features.csv")
entity_file_name = os.path.join(DATA_DIR, "entity.csv")

features = get_features()
features.to_csv(features_file_name)

entity_df = features[["created", "updated", "user_id"]]
entity_df.rename(columns={'created': 'created_timestamp', 'updated': 'event_timestamp'}, inplace=True)
entity_df.to_csv(entity_file_name)

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

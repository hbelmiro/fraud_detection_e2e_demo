import os
from datetime import timedelta

import feast.types
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
from pyspark.sql import SparkSession, Window
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import (
    max as spark_max,
    min as spark_min,
    col, lit, row_number, concat, current_timestamp, to_timestamp, count, avg, stddev, to_utc_timestamp, datediff
)

MINIO_BUCKET = os.getenv("MINIO_BUCKET")

LOCAL_FEATURE_REPO_DIR = "."
LOCAL_DATA_DIR = os.path.join(LOCAL_FEATURE_REPO_DIR, "data")
LOCAL_INPUT_DIR = os.path.join(LOCAL_DATA_DIR, "input")
LOCAL_OUTPUT_DIR = os.path.join(LOCAL_DATA_DIR, "output")


def get_spark():
    return SparkSession.builder.appName("FeatureEngineeringSpark").getOrCreate()


def calculate_point_in_time_features(label_dataset: DataFrame, transactions_df: DataFrame) -> DataFrame:
    label_dataset = label_dataset.withColumn(
        "created",
        to_timestamp("created", "yyyy-MM-dd HH:mm:ss.SSSSSS")
    )
    transactions_df = transactions_df.withColumn(
        "transaction_timestamp",
        to_timestamp(col("date_of_transaction"))
    )

    # Get all transactions before the created time
    label_subset = label_dataset.select("user_id", "created")
    transactions_before = label_subset.join(
        transactions_df,
        on="user_id",
        how="inner"
    )
    transactions_before = transactions_before.filter(
        col("transaction_timestamp") < col("created_x")
    )
    transactions_before = transactions_before.withColumn(
        "days_between_transactions",
        datediff(col("created_x"), col("transaction_timestamp"))
    )

    features_df = (
        transactions_before
        .groupBy("user_id", "created")
        .agg(
            count("transaction_amount").alias("num_prev_transactions"),
            avg("transaction_amount").alias("avg_prev_transaction_amount"),
            spark_max("transaction_amount").alias("max_prev_transaction_amount"),
            stddev("transaction_amount").alias("stdv_prev_transaction_amount"),
            spark_min("days_between_transactions").alias("days_since_last_transaction"),
            spark_max("days_between_transactions").alias("days_since_first_transaction")
        )
        .na.fill(0)
    )

    final_df = label_dataset.join(
        features_df,
        on=[
            label_dataset.user_id == features_df.user_id,
            label_dataset.created == features_df.created_x
        ],
        how="left"
    )
    final_df = final_df.drop("created_x")

    return final_df


def get_features() -> DataFrame:
    print("loading data...")

    spark = get_spark()

    train_set = spark.read.csv("s3a://" + MINIO_BUCKET + "/artifacts/feature_repo/data/input/train.csv", header=True, inferSchema=True)
    test_set = spark.read.csv("s3a://" + MINIO_BUCKET + "/artifacts/feature_repo/data/input/test.csv", header=True, inferSchema=True)
    validate_set = spark.read.csv("s3a://" + MINIO_BUCKET + "/artifacts/feature_repo/data/input/validate.csv", header=True, inferSchema=True)
    train_set = train_set.withColumn("set", lit("train"))
    test_set = test_set.withColumn("set", lit("test"))
    validate_set = validate_set.withColumn("set", lit("valid"))

    all_sets = train_set.unionByName(test_set).unionByName(validate_set)

    all_sets = all_sets.withColumn("fraud", col("fraud") == 1.0)
    all_sets = all_sets.withColumn("repeat_retailer", col("repeat_retailer") == 1.0)
    all_sets = all_sets.withColumn("used_chip", col("used_chip") == 1.0)
    all_sets = all_sets.withColumn("used_pin_number", col("used_pin_number") == 1.0)
    all_sets = all_sets.withColumn("online_order", col("online_order") == 1.0)

    w = Window.orderBy(lit(1))

    all_sets = (
        all_sets
        .withColumn("idx", row_number().over(w))
        .withColumn("user_id", concat(lit("user_"), col("idx") - lit(1)))
        .withColumn("transaction_id", concat(lit("txn_"), col("idx") - lit(1)))
        .drop("idx")
    )

    for date_col in ["created", "updated"]:
        all_sets = all_sets.withColumn(date_col, current_timestamp())

    columns = [
        "user_id",
        "fraud",
        "created",
        "updated",
        "set",
        "distance_from_home",
        "distance_from_last_transaction",
        "ratio_to_median_purchase_price",
    ]

    label_dataset = all_sets.select(*columns)

    user_purchase_history = spark.read.csv(
        f"s3a://{MINIO_BUCKET}/artifacts/feature_repo/data/input/raw_transaction_datasource.csv", header=True,
        inferSchema=True)

    features_df = calculate_point_in_time_features(label_dataset, user_purchase_history)

    selected_columns = all_sets.select(
        "user_id", "created", "used_chip", "used_pin_number", "online_order"
    )

    features_df = features_df.join(
        selected_columns,
        on=["user_id", "created"],
        how="inner"
    )

    return features_df


features_file_name = os.path.join(LOCAL_OUTPUT_DIR, "features.csv")
entity_file_name = os.path.join(LOCAL_OUTPUT_DIR, "entity.csv")

features = get_features()

entity_df: DataFrame = features.select("created", "updated", "user_id")
entity_df = (
    entity_df
      .withColumnRenamed("created", "created_timestamp")
      .withColumnRenamed("updated", "event_timestamp")
)
entity_df.write.option("header", True).mode("overwrite").csv(entity_file_name)

# Define a project for the feature repo
project = Project(name="fraud_detection_e2e_demo", description="A project for driver statistics")
# Define an entity for the driver. You can think of an entity as a primary key used to
# fetch features.
transaction = Entity(name="transaction", join_keys=["user_id"])
df = features
df = (
    df.withColumn("created", to_timestamp(col("created"), "yyyy-MM-dd HH:mm:ss.SSSSSS"))
    .withColumn("created", to_utc_timestamp(col("created"), "UTC"))
)
df = (
    df.withColumn("updated", to_timestamp(col("updated"), "yyyy-MM-dd HH:mm:ss.SSSSSS"))
    .withColumn("updated", to_utc_timestamp(col("updated"), "UTC"))
)

parquet_file_name = features_file_name.replace(".csv", ".parquet")

df.write.parquet(parquet_file_name)
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

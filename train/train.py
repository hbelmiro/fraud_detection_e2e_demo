import argparse
import logging
import os
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import tf2onnx
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Input
from keras.models import Sequential
from minio import Minio, S3Error
from pyspark import Row
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from sklearn.utils import class_weight

logging.basicConfig(
    level=logging.INFO,
)

SPARK = SparkSession.builder.appName("FraudDetection").getOrCreate()

MINIO_ENDPOINT = "http://minio-service.fraud-detection.svc.cluster.local:9000"
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minio123"
MINIO_BUCKET = "mlpipeline"

REMOTE_FEATURE_REPO_DIR = "artifacts/feature_repo/"
REMOTE_DATA_DIR = REMOTE_FEATURE_REPO_DIR + "data/"
REMOTE_OUTPUT_DIR = REMOTE_DATA_DIR + "output/"


def bool_to_float(df, column_names: list):
    for column_name in column_names:
        df = df.withColumn(column_name, when(col(column_name) == True, 1.0).otherwise(0.0))

    return df


def download_artifact(artifact_path, dest):
    client = Minio(
        MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_ENDPOINT.startswith("https://")
    )

    os.makedirs(dest, exist_ok=True)

    final_file_path = os.path.join(dest, "training_data.csv")
    object_path = artifact_path.replace(f"s3://{MINIO_BUCKET}/", "")

    print(f"Downloading: {artifact_path} -> {final_file_path}")
    client.fget_object(MINIO_BUCKET, object_path, final_file_path)
    print("Download complete.")


def get_features(features_url: str) -> pd.DataFrame:
    download_artifact(features_url, "./features")
    features = SPARK.read.csv("./features/training_data.csv", header=True, inferSchema=True)
    features = bool_to_float(features, ["fraud", "used_chip", "used_pin_number", "online_order"])

    # Removes the titles
    features = features.rdd.zipWithIndex() \
        .filter(lambda row_index: row_index[1] > 0) \
        .map(lambda row_index: row_index[0]) \
        .toDF(features.columns)

    return features


def build_model(feature_indexes: list[int]) -> Sequential:
    model = Sequential()
    model.add(Input(shape=(len(feature_indexes),)))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(x_train, x_val, y_train, y_val, class_weights, model):
    import time
    start = time.time()
    epochs = 2
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=True,
        class_weight=class_weights
    )
    end = time.time()
    print(f"Training of model is complete. Took {end - start} seconds")


def save_model(x_train, model, model_path):
    import tensorflow as tf
    # Normally we use tf2.onnx.convert.from_keras.
    # workaround for tf2onnx bug https://github.com/onnx/tensorflow-onnx/issues/2348
    # Wrap the model in a `tf.function`
    @tf.function(input_signature=[tf.TensorSpec([None, x_train.shape[1]], tf.float32, name='dense_input')])
    def model_fn(x):
        return model(x)

    # Convert the Keras model to ONNX
    model_proto, _ = tf2onnx.convert.from_function(
        model_fn,
        input_signature=[tf.TensorSpec([None, x_train.shape[1]], tf.float32, name='dense_input')]
    )

    # Save the model as ONNX for easy use of ModelMesh
    onnx.save(model_proto, model_path)


def upload_model(model_local_path: str, run_id: str):
    client = Minio(
        MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_ENDPOINT.startswith("https://")
    )

    object_path = os.path.join(os.path.join(REMOTE_OUTPUT_DIR, run_id, "model.onnx"))

    try:
        print(f"Uploading: {model_local_path} -> {object_path}")
        client.fput_object(MINIO_BUCKET, object_path, model_local_path)
    except S3Error as e:
        print(f"Failed to upload {model_local_path}: {e}")


def main():
    if not all([MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET]):
        raise ValueError("Missing required environment variables!")

    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--training-data-url", type=str, required=True, help="URL to the training data CSV file.")
    parser.add_argument('--model', type=str, required=True, help="The path where the model will be saved.")
    args = parser.parse_args()

    features_url = args.training_data_url
    model_path = args.model

    logging.info("Training data URL: {}".format(features_url))
    logging.info("Model path: {}".format(model_path))

    features = get_features(features_url)
    # Set the input (X) and output (Y) data.
    # The only output data is whether it's fraudulent. All other fields are inputs to the model.
    feature_indexes: list[int] = [
        4,  # distance_from_last_transaction
        5,  # ratio_to_median_purchase_price
        7,  # used_chip
        8,  # used_pin_number
        9,  # online_order
    ]
    label_indexes = [
        6  # fraud
    ]
    # Split the data into train, validation, and test datasets
    train_features = features.filter(features["set"] == "train")
    test_features = features.filter(features["set"] == "test")
    validate_features = features.filter(features["set"] == "valid")

    # Select features
    x_train = train_features.select([col(features.columns[i]) for i in feature_indexes]).rdd.map(
        lambda x: Vectors.dense(x)).collect()
    y_train = train_features.select(col(features.columns[label_indexes[0]])).rdd.map(lambda x: x[0]).collect()
    x_val = validate_features.select([col(features.columns[i]) for i in feature_indexes]).rdd.map(
        lambda x: Vectors.dense(x)).collect()
    y_val = validate_features.select(col(features.columns[label_indexes[0]])).rdd.map(lambda x: x[0]).collect()
    x_test = test_features.select([col(features.columns[i]) for i in feature_indexes]).rdd.map(
        lambda x: Vectors.dense(x)).collect()
    y_test = test_features.select(col(features.columns[label_indexes[0]])).rdd.map(lambda x: x[0]).collect()

    # Convert to numpy arrays for use with Keras
    x_train = np.array(x_train)
    x_val = np.array(x_val)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    # Scale the data to remove mean and have unit variance. The data will be between -1 and 1, which makes it a lot easier for the model to learn than random (and potentially large) values.
    # It is important to only fit the scaler to the training data, otherwise you are leaking information about the global distribution of variables (which is influenced by the test set) into the training set.
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    train_df = SPARK.createDataFrame([(Vectors.dense(x_train[i]),) for i in range(len(x_train))], ["features"])
    test_df = SPARK.createDataFrame([(Vectors.dense(x_test[i]),) for i in range(len(x_test))], ["features"])
    val_df = SPARK.createDataFrame([(Vectors.dense(x_val[i]),) for i in range(len(x_val))], ["features"])

    scaler_model = scaler.fit(train_df)
    train_df = scaler_model.transform(train_df)
    test_df = scaler_model.transform(test_df)
    val_df = scaler_model.transform(val_df)

    # Extract scaled data
    x_train = np.array([row.scaled_features.toArray() for row in train_df.collect()])
    x_val = np.array([row.scaled_features.toArray() for row in val_df.collect()])
    x_test = np.array([row.scaled_features.toArray() for row in test_df.collect()])

    test_data = [Row(features=Vectors.dense(x_test[i]), label=float(y_test[i])) for i in range(len(x_test))]
    test_df = SPARK.createDataFrame(test_data)

    Path("artifact").mkdir(parents=True, exist_ok=True)
    test_df.write.parquet("artifact/test_data.parquet")
    scaler_model.save("artifact/scaler_model")

    # Since the dataset is unbalanced (it has many more non-fraud transactions than fraudulent ones), set a class weight to weight the few fraudulent transactions higher than the many non-fraud transactions.
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.ravel())
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    model: Sequential = build_model(feature_indexes)

    train_model(x_train, x_val, y_train, y_val, class_weights, model)

    save_model(x_train, model, model_path)


if __name__ == "__main__":
    main()

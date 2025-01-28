import os
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import tf2onnx
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Input
from keras.models import Sequential
from pyspark import Row
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from sklearn.utils import class_weight

spark = SparkSession.builder.appName("FraudDetection").getOrCreate()


def bool_to_float(df, column_names: list):
    for column_name in column_names:
        df = df.withColumn(column_name, when(col(column_name) == True, 1.0).otherwise(0.0))

    return df


def get_features() -> pd.DataFrame:
    features = spark.read.csv("../feature_engineering/feature_repo/data/features.csv", header=True,
                              inferSchema=True)

    features = bool_to_float(features, ["fraud", "used_chip", "used_pin_number", "online_order"])

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


def save_model(x_train, model):
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
    os.makedirs("models/fraud/1", exist_ok=True)
    onnx.save(model_proto, "models/fraud/1/model.onnx")


def main():
    features = get_features()
    # features.to_csv(os.path.join(data_dir, "final_data.csv"))
    # Set the input (X) and output (Y) data.
    # The only output data is whether it's fraudulent. All other fields are inputs to the model.
    feature_indexes: list[int] = [
        7,  # distance_from_last_transaction
        8,  # ratio_to_median_purchase_price
        15,  # used_chip
        16,  # used_pin_number
        17,  # online_order
    ]
    label_indexes = [
        2  # fraud
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
    train_df = spark.createDataFrame([(Vectors.dense(x_train[i]),) for i in range(len(x_train))], ["features"])
    test_df = spark.createDataFrame([(Vectors.dense(x_test[i]),) for i in range(len(x_test))], ["features"])
    val_df = spark.createDataFrame([(Vectors.dense(x_val[i]),) for i in range(len(x_val))], ["features"])

    scaler_model = scaler.fit(train_df)
    train_df = scaler_model.transform(train_df)
    test_df = scaler_model.transform(test_df)
    val_df = scaler_model.transform(val_df)

    # Extract scaled data
    x_train = np.array([row.scaled_features.toArray() for row in train_df.collect()])
    x_val = np.array([row.scaled_features.toArray() for row in val_df.collect()])
    x_test = np.array([row.scaled_features.toArray() for row in test_df.collect()])

    test_data = [Row(features=Vectors.dense(x_test[i]), label=float(y_test[i])) for i in range(len(x_test))]
    test_df = spark.createDataFrame(test_data)

    Path("artifact").mkdir(parents=True, exist_ok=True)
    test_df.write.parquet("artifact/test_data.parquet")
    scaler_model.save("artifact/scaler_model")

    # Since the dataset is unbalanced (it has many more non-fraud transactions than fraudulent ones), set a class weight to weight the few fraudulent transactions higher than the many non-fraud transactions.
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.ravel())
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    model: Sequential = build_model(feature_indexes)

    train_model(x_train, x_val, y_train, y_val, class_weights, model)

    save_model(x_train, model)


if __name__ == "__main__":
    main()

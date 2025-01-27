import os
import pickle
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import tf2onnx
from keras.api._v2.keras.layers import Dense, Dropout, BatchNormalization, Activation, Input
from keras.api._v2.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

data_dir = "feature_repo/data"


def get_features() -> pd.DataFrame:
    return pd.read_csv("../feature_engineering/feature_repo/data/features.csv")


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


def train_model(X_train, X_val, y_train, y_val, class_weights, model):
    import time
    start = time.time()
    epochs = 2
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=True,
        class_weight=class_weights
    )
    end = time.time()
    print(f"Training of model is complete. Took {end - start} seconds")


def save_model(X_train, model):
    import tensorflow as tf
    # Normally we use tf2.onnx.convert.from_keras.
    # workaround for tf2onnx bug https://github.com/onnx/tensorflow-onnx/issues/2348
    # Wrap the model in a `tf.function`
    @tf.function(input_signature=[tf.TensorSpec([None, X_train.shape[1]], tf.float32, name='dense_input')])
    def model_fn(x):
        return model(x)

    # Convert the Keras model to ONNX
    model_proto, _ = tf2onnx.convert.from_function(
        model_fn,
        input_signature=[tf.TensorSpec([None, X_train.shape[1]], tf.float32, name='dense_input')]
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
    train_features = features[features["set"] == "train"]
    test_features = features[features["set"] == "test"]
    validate_features = features[features["set"] == "valid"]
    X_train = train_features.iloc[:, feature_indexes].values
    y_train = train_features.iloc[:, label_indexes].values
    X_val = validate_features.iloc[:, feature_indexes].values
    y_val = validate_features.iloc[:, label_indexes].values
    X_test = test_features.iloc[:, feature_indexes].values
    y_test = test_features.iloc[:, label_indexes].values
    # Scale the data to remove mean and have unit variance. The data will be between -1 and 1, which makes it a lot easier for the model to learn than random (and potentially large) values.
    # It is important to only fit the scaler to the training data, otherwise you are leaking information about the global distribution of variables (which is influenced by the test set) into the training set.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    Path("artifact").mkdir(parents=True, exist_ok=True)
    with open("artifact/test_data.pkl", "wb") as handle:
        pickle.dump((X_test, y_test), handle)
    with open("artifact/scaler.pkl", "wb") as handle:
        pickle.dump(scaler, handle)
    # Since the dataset is unbalanced (it has many more non-fraud transactions than fraudulent ones), set a class weight to weight the few fraudulent transactions higher than the many non-fraud transactions.
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.ravel())
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    model: Sequential = build_model(feature_indexes)

    train_model(X_train, X_val, y_train, y_val, class_weights, model)

    save_model(X_train, model)


main()

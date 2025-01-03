import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import tensorflow as tf
from feast import FeatureStore
from keras.api.layers import Dense, Dropout, BatchNormalization, Activation
from keras.api.models import Sequential
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight


def fetch_historical_features_entity_df(store: FeatureStore, for_batch_scoring: bool) -> DataFrame:
    # Note: see https://docs.feast.dev/getting-started/concepts/feature-retrieval for more details on how to retrieve
    # for all entities in the offline store instead
    entity_df = pd.DataFrame.from_dict(
        {
            # entity's join key -> entity values
            "transaction_id": [1, 2, 3],
            # "event_timestamp" (reserved key) -> timestamps
            "event_timestamp": [
                datetime(2024, 12, 1, 4, 17, 47, 688222),
                datetime(2024, 12, 5, 7, 22, 19, 688222),
                datetime(2024, 12, 10, 11, 27, 18, 688222),
            ],
        }
    )

    # For batch scoring, we want the latest timestamps
    if for_batch_scoring:
        entity_df["event_timestamp"] = pd.to_datetime("now", utc=True)

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "transactions:distance_from_last_transaction",
            "transactions:ratio_to_median_purchase_price",
            "transactions:used_chip",
            "transactions:used_pin_number",
            "transactions:online_order",
            "transactions:fraud",
        ],
    ).to_df()

    return training_df


store = FeatureStore(repo_path="../feature_repo")
df = fetch_historical_features_entity_df(store, for_batch_scoring=False)

# Set the input (X) and output (Y) data.
# The only output data is whether it's fraudulent. All other fields are inputs to the model.

feature_indexes = [
    1,  # distance_from_last_transaction
    2,  # ratio_to_median_purchase_price
    4,  # used_chip
    5,  # used_pin_number
    6,  # online_order
]

label_indexes = [
    7  # fraud
]

df = pd.read_csv('data/train.csv')
X_train = df.iloc[:, feature_indexes].values
y_train = df.iloc[:, label_indexes].values

df = pd.read_csv('data/validate.csv')
X_val = df.iloc[:, feature_indexes].values
y_val = df.iloc[:, label_indexes].values

df = pd.read_csv('data/test.csv')
X_test = df.iloc[:, feature_indexes].values
y_test = df.iloc[:, label_indexes].values

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

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=len(feature_indexes)))
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

model.summary()

import os
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

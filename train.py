import pickle
from pathlib import Path
import os

import numpy as np
import pandas as pd
from keras.api.layers import Dense, Dropout, BatchNormalization, Activation
from keras.api.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

data_dir = "feature_repo/data"

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
    features = (
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
            features,
            left_on=["user_id", "created"],
            right_on=["user_id", "created_x"],
            how="left",
        )
        .reset_index(drop=True)
        .drop("created_x", axis=1)
    )

    return final_df


def get_features() -> pd.DataFrame:
    print("loading data...")

    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    valid = pd.read_csv(os.path.join(data_dir, "validate.csv"))
    train["set"] = "train"
    test["set"] = "test"
    valid["set"] = "valid"

    df = pd.concat([train, test, valid], axis=0).reset_index(drop=True)

    df["user_id"] = [f"user_{i}" for i in range(df.shape[0])]
    df["transaction_id"] = [f"txn_{i}" for i in range(df.shape[0])]

    for date_col in ["created", "updated"]:
        df[date_col] = pd.Timestamp.now()

    label_dataset = pd.DataFrame(
        df[
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

    user_purchase_history = pd.read_csv(os.path.join(data_dir, "raw_transactions.csv"))

    features = calculate_point_in_time_features(label_dataset, user_purchase_history)

    features = features.merge(
        df[["user_id", "created", "used_chip", "used_pin_number", "online_order"]],
        on=["user_id", "created"],
    )

    return features


features = get_features()
# features.to_csv(os.path.join(data_dir, "final_data.csv"))

# Set the input (X) and output (Y) data.
# The only output data is whether it's fraudulent. All other fields are inputs to the model.

feature_indexes = [
    6,  # distance_from_last_transaction
    7,  # ratio_to_median_purchase_price
    14,  # used_chip
    15,  # used_pin_number
    16,  # online_order
]

label_indexes = [
    1  # fraud
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


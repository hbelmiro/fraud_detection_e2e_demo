import os

import pandas as pd

data_dir = "data"


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

    train_set = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_set = pd.read_csv(os.path.join(data_dir, "test.csv"))
    validate_set = pd.read_csv(os.path.join(data_dir, "validate.csv"))
    train_set["set"] = "train"
    test_set["set"] = "test"
    validate_set["set"] = "valid"

    df = pd.concat([train_set, test_set, validate_set], axis=0).reset_index(drop=True)

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

    user_purchase_history = pd.read_csv(os.path.join(data_dir, "raw_transaction_datasource.csv"))

    features = calculate_point_in_time_features(label_dataset, user_purchase_history)

    features = features.merge(
        df[["user_id", "created", "used_chip", "used_pin_number", "online_order"]],
        on=["user_id", "created"],
    )

    return features


def main():
    features = get_features()
    features.to_csv(os.path.join(data_dir, "features.csv"))


# if __name__ == "__main__":
#     main()

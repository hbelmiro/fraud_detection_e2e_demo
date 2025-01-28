import os

import pandas as pd
from feast import FeatureStore


def run_demo():
    store = FeatureStore(repo_path="feature_repo/")

    print("\n--- Historical features for training ---")
    fetch_historical_features_entity_df(store)


def fetch_historical_features_entity_df(store: FeatureStore):
    entity_df = pd.read_csv(os.path.join("feature_repo/data/entity.csv"))
    entity_df["created_timestamp"] = pd.to_datetime("now", utc=True)
    entity_df["event_timestamp"] = pd.to_datetime("now", utc=True)

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "transactions:distance_from_home",
            "transactions:distance_from_last_transaction",
            "transactions:fraud",
        ],
    ).to_df()
    print(training_df.head())


def fetch_online_features(store, source: str = ""):
    entity_rows = [
        {
            "driver_id": 1001,
            "val_to_add": 1000,
            "val_to_add_2": 2000,
        },
        {
            "driver_id": 1002,
            "val_to_add": 1001,
            "val_to_add_2": 2002,
        },
    ]
    if source == "feature_service":
        features_to_fetch = store.get_feature_service("driver_activity_v1")
    elif source == "push":
        features_to_fetch = store.get_feature_service("driver_activity_v3")
    else:
        features_to_fetch = [
            "driver_hourly_stats:acc_rate",
            "transformed_conv_rate:conv_rate_plus_val1",
            "transformed_conv_rate:conv_rate_plus_val2",
        ]
    returned_features = store.get_online_features(
        features=features_to_fetch,
        entity_rows=entity_rows,
    ).to_dict()
    for key, value in sorted(returned_features.items()):
        print(key, " : ", value)


if __name__ == "__main__":
    run_demo()

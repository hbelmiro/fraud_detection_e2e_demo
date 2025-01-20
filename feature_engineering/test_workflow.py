from datetime import datetime

import pandas as pd
from feast import FeatureStore


def run_demo():
    store = FeatureStore(repo_path="..")
    print("\n--- Run feast apply ---")
    # subprocess.run(["feast", "apply"])

    print("\n--- Historical features for training ---")
    fetch_historical_features_entity_df(store, for_batch_scoring=False)

    # print("\n--- Historical features for batch scoring ---")
    # fetch_historical_features_entity_df(store, for_batch_scoring=True)
    #
    # print("\n--- Load features into online store ---")
    # store.materialize_incremental(end_date=datetime.now())
    #
    # print("\n--- Online features ---")
    # fetch_online_features(store)
    #
    # print("\n--- Online features retrieved (instead) through a feature service---")
    # fetch_online_features(store, source="feature_service")
    #
    # print(
    #     "\n--- Online features retrieved (using feature service v3, which uses a feature view with a push source---"
    # )
    # fetch_online_features(store, source="push")
    #
    # print("\n--- Simulate a stream event ingestion of the hourly stats df ---")
    # event_df = pd.DataFrame.from_dict(
    #     {
    #         "driver_id": [1001],
    #         "event_timestamp": [
    #             datetime.now(),
    #         ],
    #         "created": [
    #             datetime.now(),
    #         ],
    #         "conv_rate": [1.0],
    #         "acc_rate": [1.0],
    #         "avg_daily_trips": [1000],
    #     }
    # )
    # print(event_df)
    # store.push("driver_stats_push_source", event_df, to=PushMode.ONLINE_AND_OFFLINE)
    #
    # print("\n--- Online features again with updated values from a stream push---")
    # fetch_online_features(store, source="push")
    #
    # print("\n--- Run feast teardown ---")
    # subprocess.run(["feast", "teardown"])


def fetch_historical_features_entity_df(store: FeatureStore, for_batch_scoring: bool):
    # Note: see https://docs.feast.dev/getting-started/concepts/feature-retrieval for more details on how to retrieve
    # for all entities in the offline store instead
    entity_df = pd.DataFrame.from_dict(
        {
            # entity's join key -> entity values
            "user_id": ["user_1", "user_2", "user_3"],
            # "event_timestamp" (reserved key) -> timestamps
            "created": [
                datetime(2025, 1, 7, 15, 9, 26, 562557),
                datetime(2025, 1, 7, 15, 9, 26, 562557),
                datetime(2025, 1, 7, 15, 9, 26, 562557),
            ],
        }
    )

    # For batch scoring, we want the latest timestamps
    if for_batch_scoring:
        entity_df["created"] = pd.to_datetime("now", utc=True)

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
        # {join_key: entity_value}
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

import os

import pandas as pd
from feast import FeatureStore


def main():
    store = FeatureStore(repo_path="feature_repo/")
    fetch_historical_features_entity_df(store)


def fetch_historical_features_entity_df(store: FeatureStore):
    entity_df = pd.read_csv(os.path.join("feature_repo/data/output/entity.csv"))
    entity_df["created_timestamp"] = pd.to_datetime("now", utc=True)
    entity_df["event_timestamp"] = pd.to_datetime("now", utc=True)

    features = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "transactions:distance_from_home",
            "transactions:distance_from_last_transaction",
            "transactions:ratio_to_median_purchase_price",
            "transactions:fraud",
            "transactions:used_chip",
            "transactions:used_pin_number",
            "transactions:online_order",
            "transactions:set"
        ],
    ).to_df()

    features.to_csv(os.path.join("fetched_features/training_data.csv"))


if __name__ == "__main__":
    main()

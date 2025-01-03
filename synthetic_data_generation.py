import os
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_random_transactions(
        users_df: pd.DataFrame, max_transactions: int = 11, max_days_back=365
) -> pd.DataFrame:
    # Predefined lists of categories and locations
    transaction_categories = [
        "Groceries",
        "Utilities",
        "Entertainment",
        "Dining",
        "Travel",
        "Health",
        "Education",
        "Shopping",
        "Automotive",
        "Rent",
    ]
    cities_and_states = [
        ("New York", "NY"),
        ("Los Angeles", "CA"),
        ("Chicago", "IL"),
        ("Houston", "TX"),
        ("Phoenix", "AZ"),
        ("Philadelphia", "PA"),
        ("San Antonio", "TX"),
        ("San Diego", "CA"),
        ("Dallas", "TX"),
        ("San Jose", "CA"),
    ]
    transactions_list = []
    total_users = users_df.shape[0]
    batch = total_users // 10

    i: int
    for i, row in users_df.iterrows():
        num_transactions = np.random.randint(1, max_transactions)
        for j in range(num_transactions):
            # Random date within the last 10-max_days_back (default 365) days
            random_days = np.random.randint(10, max_days_back)
            date_of_transaction = datetime.now() - timedelta(days=random_days)
            city, state = random.choice(cities_and_states)
            if j == (num_transactions - 1):
                date_of_transaction = row["created"]

            transactions_list.append(
                {
                    "user_id": row["user_id"],
                    "created": date_of_transaction,
                    "updated": date_of_transaction,
                    "date_of_transaction": date_of_transaction,
                    "transaction_amount": round(np.random.uniform(10, 1000), 2),
                    "transaction_category": random.choice(transaction_categories),
                    "card_token": str(uuid.uuid4()),
                    "city": city,
                    "state": state,
                }
            )
        if (i % batch) == 0:
            formatted_i = f"{i:,}"
            percent_complete = i / total_users * 100
            print(
                f"{formatted_i:>{len(f'{total_users:,}')}} of {total_users:,} "
                f"({percent_complete:.0f}%) complete"
            )

    return pd.DataFrame(transactions_list)


print("loading data...")
train = pd.read_csv(os.path.join("feature_repo", "data", "train.csv"))
test = pd.read_csv(os.path.join("feature_repo", "data", "test.csv"))
valid = pd.read_csv(os.path.join("feature_repo", "data", "validate.csv"))

train["set"] = "train"
test["set"] = "test"
valid["set"] = "valid"

df = pd.concat([train, test, valid], axis=0).reset_index(drop=True)

df["user_id"] = [f"user_{i}" for i in range(df.shape[0])]
df["transaction_id"] = [f"txn_{i}" for i in range(df.shape[0])]

for date_col in ["created", "updated"]:
    df[date_col] = pd.Timestamp.now()

print("generating transaction level data...")
user_purchase_history = generate_random_transactions(
    users_df=df[df["repeat_retailer"] == 1].reset_index(drop=True),
    max_transactions=5,
    max_days_back=365,
)

user_purchase_history.to_csv(
    os.path.join("feature_repo/data", "raw_transaction_datasource.csv")
)

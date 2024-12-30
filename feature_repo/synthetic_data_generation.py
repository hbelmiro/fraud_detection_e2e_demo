from datetime import datetime, timedelta

import numpy as np
import pandas as pd

df = pd.read_csv("data/train.csv")

# Generate synthetic event timestamps (last 30 days)
start_date = datetime.now() - timedelta(days=30)
end_date = datetime.now()

def random_date(start, end):
    """Generate a random datetime between `start` and `end`."""
    return start + timedelta(seconds=np.random.randint(0, int((end - start).total_seconds())))

df['transaction_id'] = df.index + 1
df = df[['transaction_id'] + [col for col in df.columns if col != 'transaction_id']]

# Add event_timestamp
df['event_timestamp'] = [random_date(start_date, end_date) for _ in range(len(df))]

# Ensure created_timestamp is always later than event_timestamp (within a few seconds or minutes)
df['created_timestamp'] = df['event_timestamp'] + pd.to_timedelta(
    np.random.randint(1, 300, size=len(df)), unit='s'  # Adding 1 to 300 seconds
)

df.to_csv('data/synthetic_train.csv')

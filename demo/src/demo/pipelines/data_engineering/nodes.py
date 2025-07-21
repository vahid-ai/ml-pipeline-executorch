from feast import FeatureStore
from sklearn.model_selection import train_test_split
import pandas as pd
from .schema import EventSchema
from demo.config import settings
from kedro.io import DataCatalog

def pull_features() -> pd.DataFrame:
    store = FeatureStore(settings.feast_repo)
    df = store.get_historical_features(
        entity_df="SELECT * FROM cicmaldroid_source",
        features=["basic_stats:cpu_usage", "basic_stats:net_bytes", "basic_stats:num_syscalls"],
    ).to_df()

    EventSchema.validate(df)
    return df

def split_data(df: pd.DataFrame, params: dict):
    train, val = train_test_split(
        df, test_size=params["test_size"], random_state=params["seed"]
    )
    return train, val

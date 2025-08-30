import os
from pathlib import Path
from feast import FeatureStore, RepoConfig
from feast.infra.offline_stores.duckdb import DuckDBOfflineStoreConfig
from feast.infra.online_stores.sqlite import SqliteOnlineStoreConfig
from typing import Optional


class FeastManager:
    """Manager class for Feast feature store operations"""
    
    def __init__(self, config: dict):
        self.config = config
        self.fs: Optional[FeatureStore] = None
        self._setup_feast_repo()
    
    def _setup_feast_repo(self):
        """Setup Feast repository configuration"""
        repo_config = RepoConfig(
            project=self.config["project"],
            provider=self.config["provider"],
            registry=self.config["registry"],
            offline_store=DuckDBOfflineStoreConfig(),
            online_store=SqliteOnlineStoreConfig(
                path=self.config["online_store_path"]
            ),
            entity_key_serialization_version=2,
        )
        
        # Create feature store instance
        self.fs = FeatureStore(config=repo_config)
    
    def apply_features(self, features_config: dict):
        """Apply feature definitions to the feature store"""
        objects = [
            features_config["entity"],
            features_config["feature_view"]
        ]
        self.fs.apply(objects)
    
    def materialize_features(self, start_date, end_date):
        """Materialize features to online store"""
        self.fs.materialize(start_date, end_date)
    
    def get_historical_features(self, entity_df, feature_refs):
        """Get historical features for training"""
        return self.fs.get_historical_features(
            entity_df=entity_df,
            features=feature_refs
        ).to_df()
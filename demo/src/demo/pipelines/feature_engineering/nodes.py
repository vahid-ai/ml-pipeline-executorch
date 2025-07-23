import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta
from ...feature_store.feast_config import FeastManager
from ...feature_store.feature_definitions import create_feature_definitions
import logging


logger = logging.getLogger(__name__)


def prepare_features_for_feast(
    data_splits: Dict[str, pd.DataFrame],
    dataset_config: Dict[str, Any],
    feature_store_config: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """Prepare features for Feast feature store"""
    
    # Combine all splits for feature store
    all_data = pd.concat([
        data_splits["train"],
        data_splits["validation"],
        data_splits["test"]
    ])
    
    # Ensure we have required columns
    if dataset_config["timestamp_column"] not in all_data.columns:
        all_data["event_timestamp"] = pd.Timestamp.now()
    else:
        all_data["event_timestamp"] = all_data[dataset_config["timestamp_column"]]
    
    # Ensure entity column exists
    if feature_store_config["entity_column"] not in all_data.columns:
        if dataset_config["id_column"] in all_data.columns:
            all_data[feature_store_config["entity_column"]] = all_data[dataset_config["id_column"]]
        else:
            all_data[feature_store_config["entity_column"]] = range(len(all_data))
    
    # Select feature columns
    feature_columns = [col for col in all_data.columns 
                      if col not in [feature_store_config["entity_column"], 
                                   "event_timestamp",
                                   dataset_config.get("target_column")]]
    
    # Create feature dataframe
    feature_df = all_data[[feature_store_config["entity_column"], "event_timestamp"] + feature_columns]
    
    return {
        "feature_df": feature_df,
        "feature_columns": feature_columns
    }


def setup_feast_feature_store(
    feature_data: Dict[str, Any],
    feature_store_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Setup Feast feature store with features"""
    
    feature_df = feature_data["feature_df"]
    feature_columns = feature_data["feature_columns"]
    
    # Save features to parquet for Feast
    feature_path = "data/features.parquet"
    feature_df.to_parquet(feature_path)
    
    # Create feature definitions
    features_config = create_feature_definitions(
        feature_names=feature_columns,
        entity_name=feature_store_config["entity_name"],
        entity_column=feature_store_config["entity_column"],
        source_path=feature_path
    )
    
    # Initialize Feast manager
    feast_manager = FeastManager(feature_store_config)
    
    # Apply features to feature store
    feast_manager.apply_features(features_config)
    
    # Materialize features
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    feast_manager.materialize_features(start_date, end_date)
    
    logger.info(f"Feature store setup complete with {len(feature_columns)} features")
    
    return {
        "feast_manager": feast_manager,
        "features_config": features_config,
        "feature_columns": feature_columns
    }


def create_training_features(
    data_splits: Dict[str, pd.DataFrame],
    feast_artifacts: Dict[str, Any],
    dataset_config: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """Create feature arrays for training"""
    
    feature_columns = feast_artifacts["feature_columns"]
    target_column = dataset_config.get("target_column")
    
    training_features = {}
    
    for split_name, df in data_splits.items():
        # Extract features
        X = df[feature_columns].values
        
        # Extract labels if available
        if target_column and target_column in df.columns:
            y = df[target_column].values
        else:
            y = None
        
        training_features[f"X_{split_name}"] = X
        if y is not None:
            training_features[f"y_{split_name}"] = y
    
    # Store feature dimensions
    training_features["input_dim"] = len(feature_columns)
    training_features["feature_names"] = feature_columns
    
    return training_features
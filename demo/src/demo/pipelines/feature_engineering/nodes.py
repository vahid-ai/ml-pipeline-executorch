import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta
from ...feature_store.feast_config import FeastManager
from ...feature_store.feature_definitions import create_feature_definitions
import logging
import os
from pathlib import Path


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
    ], ignore_index=True)
    
    # Ensure we have required columns with proper timestamps
    if dataset_config["timestamp_column"] not in all_data.columns:
        # Create incremental timestamps for each row
        base_time = pd.Timestamp.now()
        all_data["event_timestamp"] = [
            base_time - pd.Timedelta(minutes=i) for i in range(len(all_data))
        ]
    else:
        all_data["event_timestamp"] = pd.to_datetime(all_data[dataset_config["timestamp_column"]])
    
    # Ensure entity column exists and is string type
    if feature_store_config["entity_column"] not in all_data.columns:
        if dataset_config["id_column"] in all_data.columns:
            all_data[feature_store_config["entity_column"]] = all_data[dataset_config["id_column"]].astype(str)
        else:
            all_data[feature_store_config["entity_column"]] = [f"sample_{i}" for i in range(len(all_data))]
    else:
        all_data[feature_store_config["entity_column"]] = all_data[feature_store_config["entity_column"]].astype(str)

    # Select feature columns
    feature_columns = [col for col in all_data.columns 
                      if col not in [feature_store_config["entity_column"], 
                                   "event_timestamp",
                                   dataset_config.get("target_column")]]
    
    # Ensure all feature columns are numeric
    for col in feature_columns:
        all_data[col] = pd.to_numeric(all_data[col], errors='coerce').fillna(0.0)
    
    # Create feature dataframe with proper column order
    feature_df = all_data[[feature_store_config["entity_column"], "event_timestamp"] + feature_columns].copy()
    
    # Ensure event_timestamp is properly formatted
    feature_df["event_timestamp"] = pd.to_datetime(feature_df["event_timestamp"])
    
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
    
    # Save features to parquet for Feast - use absolute path
    current_dir = Path.cwd()
    feature_path = current_dir / "data" / "features.parquet"
    
    # Ensure the directory exists
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with proper parquet format
    try:
        # Reset index to ensure clean parquet file
        feature_df_clean = feature_df.reset_index(drop=True)
        feature_df_clean.to_parquet(str(feature_path), index=False)
        logger.info(f"Saved features to: {feature_path}")
        
        # Verify the file was created and is readable
        if not feature_path.exists():
            raise FileNotFoundError(f"Failed to create parquet file at {feature_path}")
            
        # Test reading the file
        test_df = pd.read_parquet(str(feature_path))
        logger.info(f"Verified parquet file: {test_df.shape} rows x {test_df.shape[1]} columns")
        
    except Exception as e:
        logger.error(f"Error saving parquet file: {e}")
        raise
    
    # Initialize feast_manager and setup features
    feast_setup_success = False
    feast_error_message = None
    
    try:
        # Create feature definitions with absolute path
        features_config = create_feature_definitions(
            feature_names=feature_columns,
            entity_name=feature_store_config["entity_name"],
            entity_column=feature_store_config["entity_column"],
            source_path=str(feature_path)
        )
        
        # Initialize Feast manager
        feast_manager = FeastManager(feature_store_config)
        
        # Apply features to feature store
        feast_manager.apply_features(features_config)
        
        # Materialize features with a shorter time range to avoid data issues
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=1)  # Use 1 hour instead of 1 day
        feast_manager.materialize_features(start_date, end_date)
        
        feast_setup_success = True
        logger.info(f"Feature store setup complete with {len(feature_columns)} features")
        
    except Exception as e:
        feast_error_message = str(e)
        logger.error(f"Error setting up Feast feature store: {e}")
        logger.warning("Continuing pipeline without Feast materialization")
    
    # Return only serializable data (no FeastManager object)
    return {
        "feature_columns": feature_columns,
        "feature_path": str(feature_path),
        "feast_setup_success": feast_setup_success,
        "feast_error_message": feast_error_message,
        "feature_store_config": feature_store_config,  # Keep config for downstream use
        "num_features": len(feature_columns),
        "data_shape": feature_df.shape
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
        # Ensure all feature columns are properly numeric
        df_clean = df.copy()
        
        # Convert feature columns to numeric, handling any remaining issues
        for col in feature_columns:
            if col in df_clean.columns:
                # Convert to numeric, coercing errors to NaN
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                # Fill any remaining NaN values with 0
                df_clean[col] = df_clean[col].fillna(0.0)
                # Handle any remaining infinite values
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], 0.0)
                # Ensure the column is float64
                df_clean[col] = df_clean[col].astype(np.float64)
        
        # Extract features with proper data type handling
        try:
            X = df_clean[feature_columns].values.astype(np.float64)
        except Exception as e:
            logger.error(f"Error converting features to numeric for split {split_name}: {e}")
            logger.info(f"Feature columns data types: {df_clean[feature_columns].dtypes}")
            # Debug: check for problematic columns
            for col in feature_columns:
                if col in df_clean.columns:
                    unique_types = df_clean[col].apply(type).value_counts()
                    logger.info(f"Column {col} types: {unique_types}")
            raise
        
        # Extract labels if available
        if target_column and target_column in df_clean.columns:
            try:
                y = df_clean[target_column].astype(np.int64).values
            except Exception as e:
                logger.warning(f"Could not convert target column to integer: {e}")
                y = None
        else:
            y = None
        
        # Store arrays
        training_features[f"X_{split_name}"] = X
        if y is not None:
            training_features[f"y_{split_name}"] = y
    
    # Add metadata
    training_features["input_dim"] = len(feature_columns)
    training_features["feature_names"] = feature_columns
    
    # Log shapes for debugging
    for split_name in data_splits.keys():
        X_shape = training_features[f"X_{split_name}"].shape
        logger.info(f"{split_name} X shape: {X_shape}, dtype: {training_features[f'X_{split_name}'].dtype}")
        if f"y_{split_name}" in training_features:
            y_shape = training_features[f"y_{split_name}"].shape
            logger.info(f"{split_name} y shape: {y_shape}, dtype: {training_features[f'y_{split_name}'].dtype}")
    
    return training_features
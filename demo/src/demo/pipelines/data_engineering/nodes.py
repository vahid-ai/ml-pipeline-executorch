import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
from ...validation.schemas import get_malware_data_schema, DatasetConfig
from ...validation.expectations import DataQualityValidator


logger = logging.getLogger(__name__)


def load_cicmaldroid_dataset(filepath: str, config: Dict[str, Any]) -> pd.DataFrame:
    """Load CICMalDroid 2020 dataset"""
    logger.info(f"Loading dataset from {filepath}")
    
    # Load data based on file extension
    file_path = Path(filepath)
    
    if file_path.suffix == '.csv':
        df = pd.read_csv(filepath)
    elif file_path.suffix == '.parquet':
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
        # Add timestamp if not present
    timestamp_column = config.get("timestamp_column")
    if timestamp_column and timestamp_column not in df.columns:
        df[timestamp_column] = pd.Timestamp.now()

    # Validate schema
    try:
        schema = get_malware_data_schema()
        schema.validate(df)
        logger.info("Data validation passed")
    except Exception as e:
        logger.warning(f"Data validation failed: {e}")
    
    return df


def validate_data_quality(df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, Dict]:
    """Validate data quality using Great Expectations"""
    validator = DataQualityValidator()
    validation_results = validator.validate_dataset(df, dataset_name)
    
    if not validation_results["success"]:
        logger.warning("Data quality validation failed")
        logger.warning(f"Results: {validation_results}")
    
    return df, validation_results


def preprocess_features(
    df: pd.DataFrame,
    config: Dict[str, Any],
    scaler_type: str = "standard"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Preprocess features for training"""
    
    # Select feature columns
    feature_columns = config.get("feature_columns")
    if feature_columns:
        feature_cols = feature_columns
    else:
        # Auto-detect numeric columns
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove id and target columns
        exclude_cols = [config.get("id_column", "id")]
        target_column = config.get("target_column")
        if target_column:
            exclude_cols.append(target_column)
        
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
    
    # Handle missing values
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # Handle infinite values - ADD THIS SECTION
    # Replace positive and negative infinity with finite values
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # Alternatively, you could replace inf with max/min finite values:
    # for col in feature_cols:
    #     finite_values = df[col][np.isfinite(df[col])]
    #     if len(finite_values) > 0:
    #         max_finite = finite_values.max()
    #         min_finite = finite_values.min()
    #         df[col] = df[col].replace(np.inf, max_finite)
    #         df[col] = df[col].replace(-np.inf, min_finite)
    
    # Scale features
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    preprocessing_artifacts = {
        "scaler": scaler,
        "feature_columns": feature_cols,
        "original_shape": df.shape,
        "processed_shape": df_scaled.shape
    }
    
    return df_scaled, preprocessing_artifacts


def split_data(
    df: pd.DataFrame,
    config: Dict[str, Any],
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """Split data into train/validation/test sets"""
    
    # Separate normal and anomaly data if labels are available
    target_column = config.get("target_column")
    if target_column and target_column in df.columns:
        normal_data = df[df[target_column] == 0]
        anomaly_data = df[df[target_column] == 1]
        
        # Use only normal data for training (semi-supervised approach)
        train_data, temp_data = train_test_split(
            normal_data,
            test_size=test_size * 2,
            random_state=random_state
        )
        
        val_data, test_normal = train_test_split(
            temp_data,
            test_size=0.5,
            random_state=random_state
        )
        
        # Add anomalies to validation and test sets
        if len(anomaly_data) > 0:
            anomaly_val, anomaly_test = train_test_split(
                anomaly_data,
                test_size=0.5,
                random_state=random_state
            )
            
            val_data = pd.concat([val_data, anomaly_val])
            test_data = pd.concat([test_normal, anomaly_test])
        else:
            test_data = test_normal
    else:
        # No labels - standard split
        train_data, temp_data = train_test_split(
            df,
            test_size=test_size * 2,
            random_state=random_state
        )
        
        val_data, test_data = train_test_split(
            temp_data,
            test_size=0.5,
            random_state=random_state
        )
    
    logger.info(f"Train size: {len(train_data)}")
    logger.info(f"Validation size: {len(val_data)}")
    logger.info(f"Test size: {len(test_data)}")
    
    return {
        "train": train_data,
        "validation": val_data,
        "test": test_data
    }


def check_data_quality(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """Check for data quality issues"""
    for col in feature_cols:
        if np.isinf(df[col]).any():
            logger.warning(f"Infinite values detected in column: {col}")
        if df[col].isna().any():
            logger.warning(f"Missing values detected in column: {col}")
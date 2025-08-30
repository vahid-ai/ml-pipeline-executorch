from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
import pandera as pa
import pandas as pd
import numpy as np


class DatasetConfig(BaseModel):
    """Configuration for dataset parameters"""
    name: str = Field(..., description="Dataset name")
    path: str = Field(..., description="Path to dataset")
    target_column: Optional[str] = Field(None, description="Target column for supervised tasks")
    id_column: str = Field("id", description="ID column name")
    timestamp_column: Optional[str] = Field(None, description="Timestamp column")
    feature_columns: Optional[List[str]] = Field(None, description="Feature columns to use")
    
    class Config:
        extra = "forbid"


class ModelConfig(BaseModel):
    """Configuration for model parameters"""
    input_dim: int = Field(..., gt=0, description="Input dimension")
    hidden_dims: List[int] = Field(..., description="Hidden layer dimensions")
    latent_dim: int = Field(..., gt=0, description="Latent space dimension")
    activation: str = Field("relu", description="Activation function")
    dropout_rate: float = Field(0.1, ge=0, le=1, description="Dropout rate")
    
    @validator("hidden_dims")
    def validate_hidden_dims(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Hidden dimensions must not be empty")
        if any(dim <= 0 for dim in v):
            raise ValueError("All hidden dimensions must be positive")
        return v


class TrainingConfig(BaseModel):
    """Configuration for training parameters"""
    batch_size: int = Field(32, gt=0, description="Batch size")
    learning_rate: float = Field(1e-3, gt=0, description="Learning rate")
    epochs: int = Field(100, gt=0, description="Number of epochs")
    early_stopping_patience: int = Field(10, gt=0, description="Early stopping patience")
    validation_split: float = Field(0.2, gt=0, lt=1, description="Validation split ratio")
    anomaly_threshold_percentile: float = Field(95, ge=0, le=100, description="Anomaly threshold percentile")
    
    class Config:
        extra = "forbid"


class FeatureStoreConfig(BaseModel):
    """Configuration for Feast feature store"""
    project: str = Field("anomaly_detection", description="Feast project name")
    provider: str = Field("local", description="Feast provider")
    registry: str = Field("data/feature_store/registry.db", description="Registry path")
    online_store_path: str = Field("data/feature_store/online_store.db", description="Online store path")
    entity_name: str = Field("sample", description="Entity name")
    entity_column: str = Field("sample_id", description="Entity column name")


# Pandera schemas for data validation
def get_malware_data_schema():
    """Create schema for Android malware dataset"""
    return pa.DataFrameSchema({
        "sample_id": pa.Column(str, nullable=False, unique=True),
        "timestamp": pa.Column(pd.Timestamp, nullable=True),
        "label": pa.Column(int, nullable=True, checks=pa.Check.isin([0, 1]))
    }, strict=True, coerce=True)

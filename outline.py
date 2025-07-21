# Project Structure
"""
project/
├── conf/
│   ├── base/
│   │   ├── catalog.yml
│   │   ├── parameters.yml
│   │   └── logging.yml
│   └── local/
│       └── credentials.yml
├── src/
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── pipelines/
│   │   │   ├── __init__.py
│   │   │   ├── data_engineering/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── nodes.py
│   │   │   │   └── pipeline.py
│   │   │   ├── feature_engineering/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── nodes.py
│   │   │   │   └── pipeline.py
│   │   │   └── modeling/
│   │   │       ├── __init__.py
│   │   │       ├── nodes.py
│   │   │       └── pipeline.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── autoencoder.py
│   │   │   └── lightning_module.py
│   │   ├── feature_store/
│   │   │   ├── __init__.py
│   │   │   ├── feast_config.py
│   │   │   └── feature_definitions.py
│   │   ├── validation/
│   │   │   ├── __init__.py
│   │   │   ├── schemas.py
│   │   │   └── expectations.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── metrics.py
│   └── tests/
├── feature_repo/
│   ├── feature_store.yaml
│   └── features.py
├── notebooks/
├── mlruns/
└── requirements.txt
"""

# requirements.txt
"""
kedro>=0.18.0
kedro-mlflow>=0.11.0
pytorch-lightning>=2.0.0
torch>=2.0.0
feast[duckdb]>=0.30.0
mlflow>=2.0.0
pydantic>=2.0.0
pandera>=0.13.0
great-expectations>=0.16.0
scikit-learn>=1.0.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
"""

# src/project_name/validation/schemas.py
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
import pandera as pa
from pandera.typing import DataFrame, Series
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
class MalwareDataSchema(pa.SchemaModel):
    """Schema for Android malware dataset"""
    
    sample_id: Series[str] = pa.Field(nullable=False, unique=True)
    timestamp: Series[pd.Timestamp] = pa.Field(nullable=True)
    label: Series[int] = pa.Field(nullable=True, isin=[0, 1])
    
    class Config:
        strict = True
        coerce = True
        
    @pa.check("label")
    def check_label_distribution(cls, series: Series[int]) -> bool:
        """Ensure we have both normal and anomaly samples"""
        if series.notna().any():
            return series.value_counts().shape[0] >= 2
        return True


# src/project_name/feature_store/feature_definitions.py
from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from feast.types import Float32, Int64, String
from pydantic import BaseModel
from typing import List, Dict, Any


class FeatureDefinition(BaseModel):
    """Feature definition for Feast"""
    name: str
    dtype: str
    description: str = ""


def create_feature_definitions(
    feature_names: List[str],
    entity_name: str = "sample",
    entity_column: str = "sample_id",
    source_path: str = "data/features.parquet"
) -> Dict[str, Any]:
    """Create Feast feature definitions dynamically"""
    
    # Define entity
    entity = Entity(
        name=entity_name,
        value_type=ValueType.STRING,
        description="Sample entity for anomaly detection",
    )
    
    # Define features
    features = []
    for feature_name in feature_names:
        features.append(
            Feature(
                name=feature_name,
                dtype=ValueType.FLOAT,
                description=f"Feature {feature_name}"
            )
        )
    
    # Define source
    source = FileSource(
        path=source_path,
        event_timestamp_column="event_timestamp",
    )
    
    # Define feature view
    feature_view = FeatureView(
        name="malware_features",
        entities=[entity_name],
        ttl=timedelta(days=365),
        features=features,
        online=True,
        batch_source=source,
        tags={"team": "anomaly_detection"},
    )
    
    return {
        "entity": entity,
        "feature_view": feature_view,
        "features": features
    }


# src/project_name/feature_store/feast_config.py
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


# src/project_name/validation/expectations.py
import great_expectations as ge
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.core.expectation_configuration import ExpectationConfiguration
from typing import Dict, Any, List
import pandas as pd


class DataQualityValidator:
    """Data quality validation using Great Expectations"""
    
    def __init__(self):
        self.context = ge.get_context()
        self.expectations = self._define_expectations()
    
    def _define_expectations(self) -> List[ExpectationConfiguration]:
        """Define data quality expectations"""
        return [
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={"min_value": 100}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "sample_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_unique",
                kwargs={"column": "sample_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "label",
                    "min_value": 0,
                    "max_value": 1,
                    "mostly": 0.99
                }
            ),
        ]
    
    def validate_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Validate dataset against expectations"""
        
        # Create a batch from the dataframe
        batch_request = RuntimeBatchRequest(
            datasource_name="pandas_datasource",
            data_connector_name="runtime_data_connector",
            data_asset_name=dataset_name,
            runtime_parameters={"batch_data": df},
            batch_identifiers={"default_identifier_name": "default_identifier"},
        )
        
        # Add or update expectation suite
        suite_name = f"{dataset_name}_expectations"
        suite = self.context.create_expectation_suite(
            expectation_suite_name=suite_name,
            overwrite_existing=True
        )
        
        # Add expectations to suite
        for expectation in self.expectations:
            suite.add_expectation(expectation_configuration=expectation)
        
        # Run validation
        validator = self.context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=suite_name
        )
        
        results = validator.validate()
        
        return {
            "success": results.success,
            "statistics": results.statistics,
            "results": results.results
        }


# src/project_name/models/autoencoder.py
import torch
import torch.nn as nn
from typing import List, Tuple
from pydantic import BaseModel, validator


class AutoencoderArchitecture(BaseModel):
    """Autoencoder architecture configuration"""
    input_dim: int
    encoder_dims: List[int]
    latent_dim: int
    decoder_dims: List[int]
    activation: str = "relu"
    dropout_rate: float = 0.1
    
    @validator("decoder_dims", always=True)
    def set_decoder_dims(cls, v, values):
        if v is None or len(v) == 0:
            # Mirror encoder architecture
            return list(reversed(values.get("encoder_dims", [])))
        return v


class Autoencoder(nn.Module):
    """Flexible autoencoder for anomaly detection"""
    
    def __init__(self, config: AutoencoderArchitecture):
        super().__init__()
        self.config = config
        
        # Get activation function
        self.activation = self._get_activation(config.activation)
        
        # Build encoder
        encoder_layers = []
        dims = [config.input_dim] + config.encoder_dims + [config.latent_dim]
        
        for i in range(len(dims) - 1):
            encoder_layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                self.activation,
                nn.BatchNorm1d(dims[i + 1]),
                nn.Dropout(config.dropout_rate)
            ])
        
        self.encoder = nn.Sequential(*encoder_layers[:-1])  # Remove last dropout
        
        # Build decoder
        decoder_layers = []
        dims = [config.latent_dim] + config.decoder_dims + [config.input_dim]
        
        for i in range(len(dims) - 1):
            decoder_layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                self.activation if i < len(dims) - 2 else nn.Identity(),
                nn.BatchNorm1d(dims[i + 1]) if i < len(dims) - 2 else nn.Identity(),
                nn.Dropout(config.dropout_rate) if i < len(dims) - 2 else nn.Identity()
            ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both reconstruction and latent representation"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


# src/project_name/models/lightning_module.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from .autoencoder import Autoencoder, AutoencoderArchitecture


class AutoencoderLightningModule(pl.LightningModule):
    """PyTorch Lightning module for autoencoder training"""
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        learning_rate: float = 1e-3,
        anomaly_threshold_percentile: float = 95
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Create model architecture config
        arch_config = AutoencoderArchitecture(
            input_dim=model_config["input_dim"],
            encoder_dims=model_config["hidden_dims"],
            latent_dim=model_config["latent_dim"],
            decoder_dims=list(reversed(model_config["hidden_dims"])),
            activation=model_config["activation"],
            dropout_rate=model_config["dropout_rate"]
        )
        
        # Initialize model
        self.model = Autoencoder(arch_config)
        self.learning_rate = learning_rate
        self.anomaly_threshold_percentile = anomaly_threshold_percentile
        self.anomaly_threshold = None
        
        # Metrics storage
        self.validation_losses = []
        self.training_losses = []
    
    def forward(self, x):
        return self.model(x)
    
    def reconstruction_loss(self, x, x_recon):
        """Calculate reconstruction loss"""
        return F.mse_loss(x_recon, x, reduction='none').mean(dim=1)
    
    def training_step(self, batch, batch_idx):
        x, _ = batch if isinstance(batch, tuple) else (batch, None)
        x_recon, z = self(x)
        
        loss = self.reconstruction_loss(x, x_recon).mean()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.training_losses.append(loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, labels = batch if isinstance(batch, tuple) else (batch, None)
        x_recon, z = self(x)
        
        loss = self.reconstruction_loss(x, x_recon)
        avg_loss = loss.mean()
        
        self.log('val_loss', avg_loss, on_epoch=True, prog_bar=True)
        
        # Calculate anomaly scores and metrics if labels are available
        if labels is not None:
            anomaly_scores = loss.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            
            # Calculate AUC if we have both classes
            if len(np.unique(labels_np)) > 1:
                auc = roc_auc_score(labels_np, anomaly_scores)
                self.log('val_auc', auc, on_epoch=True, prog_bar=True)
        
        return {'val_loss': avg_loss, 'anomaly_scores': loss}
    
    def validation_epoch_end(self, outputs):
        """Calculate anomaly threshold at the end of validation"""
        all_scores = torch.cat([x['anomaly_scores'] for x in outputs])
        self.anomaly_threshold = torch.quantile(
            all_scores, 
            self.anomaly_threshold_percentile / 100
        ).item()
        
        self.log('anomaly_threshold', self.anomaly_threshold)
    
    def predict_step(self, batch, batch_idx):
        """Predict anomalies"""
        x = batch[0] if isinstance(batch, tuple) else batch
        x_recon, z = self(x)
        
        anomaly_scores = self.reconstruction_loss(x, x_recon)
        
        if self.anomaly_threshold is not None:
            anomalies = anomaly_scores > self.anomaly_threshold
        else:
            anomalies = torch.zeros_like(anomaly_scores, dtype=torch.bool)
        
        return {
            'anomaly_scores': anomaly_scores,
            'anomalies': anomalies,
            'reconstructions': x_recon,
            'latent': z
        }
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


# src/project_name/pipelines/data_engineering/nodes.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
from ...validation.schemas import MalwareDataSchema, DatasetConfig
from ...validation.expectations import DataQualityValidator


logger = logging.getLogger(__name__)


def load_cicmaldroid_dataset(filepath: str, config: DatasetConfig) -> pd.DataFrame:
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
    if config.timestamp_column and config.timestamp_column not in df.columns:
        df[config.timestamp_column] = pd.Timestamp.now()
    
    # Validate schema
    try:
        MalwareDataSchema.validate(df)
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
    config: DatasetConfig,
    scaler_type: str = "standard"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Preprocess features for training"""
    
    # Select feature columns
    if config.feature_columns:
        feature_cols = config.feature_columns
    else:
        # Auto-detect numeric columns
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove id and target columns
        exclude_cols = [config.id_column]
        if config.target_column:
            exclude_cols.append(config.target_column)
        
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
    
    # Handle missing values
    df[feature_cols] = df[feature_cols].fillna(0)
    
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
    config: DatasetConfig,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """Split data into train/validation/test sets"""
    
    # Separate normal and anomaly data if labels are available
    if config.target_column and config.target_column in df.columns:
        normal_data = df[df[config.target_column] == 0]
        anomaly_data = df[df[config.target_column] == 1]
        
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


# src/project_name/pipelines/data_engineering/pipeline.py
from kedro.pipeline import Pipeline, node
from .nodes import (
    load_cicmaldroid_dataset,
    validate_data_quality,
    preprocess_features,
    split_data
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_cicmaldroid_dataset,
                inputs=["params:dataset.path", "params:dataset_config"],
                outputs="raw_data",
                name="load_dataset",
            ),
            node(
                func=validate_data_quality,
                inputs=["raw_data", "params:dataset.name"],
                outputs=["validated_data", "validation_report"],
                name="validate_data",
            ),
            node(
                func=preprocess_features,
                inputs=["validated_data", "params:dataset_config", "params:preprocessing.scaler_type"],
                outputs=["preprocessed_data", "preprocessing_artifacts"],
                name="preprocess_features",
            ),
            node(
                func=split_data,
                inputs=[
                    "preprocessed_data",
                    "params:dataset_config",
                    "params:training.validation_split",
                    "params:random_state"
                ],
                outputs="data_splits",
                name="split_data",
            ),
        ]
    )


# src/project_name/pipelines/feature_engineering/nodes.py
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


# src/project_name/pipelines/feature_engineering/pipeline.py
from kedro.pipeline import Pipeline, node
from .nodes import (
    prepare_features_for_feast,
    setup_feast_feature_store,
    create_training_features
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=prepare_features_for_feast,
                inputs=[
                    "data_splits",
                    "params:dataset_config",
                    "params:feature_store"
                ],
                outputs="feast_feature_data",
                name="prepare_feast_features",
            ),
            node(
                func=setup_feast_feature_store,
                inputs=[
                    "feast_feature_data",
                    "params:feature_store"
                ],
                outputs="feast_artifacts",
                name="setup_feast_store",
            ),
            node(
                func=create_training_features,
                inputs=[
                    "data_splits",
                    "feast_artifacts",
                    "params:dataset_config"
                ],
                outputs="training_features",
                name="create_training_features",
            ),
        ]
    )


# src/project_name/pipelines/modeling/nodes.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
from typing import Dict, Any, Tuple
from ...models.lightning_module import AutoencoderLightningModule
from ...utils.metrics import calculate_anomaly_metrics
import logging


logger = logging.getLogger(__name__)


def prepare_dataloaders(
    training_features: Dict[str, np.ndarray],
    training_config: Dict[str, Any]
) -> Dict[str, DataLoader]:
    """Prepare PyTorch dataloaders"""
    
    batch_size = training_config["batch_size"]
    
    # Create datasets
    datasets = {}
    
    for split in ["train", "validation", "test"]:
        X = torch.FloatTensor(training_features[f"X_{split}"])
        
        if f"y_{split}" in training_features:
            y = torch.LongTensor(training_features[f"y_{split}"])
            dataset = TensorDataset(X, y)
        else:
            dataset = TensorDataset(X)
        
        datasets[split] = dataset
    
    # Create dataloaders
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    }
    
    return dataloaders


def train_autoencoder(
    dataloaders: Dict[str, DataLoader],
    training_features: Dict[str, Any],
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    mlflow_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Train autoencoder using PyTorch Lightning"""
    
    # Update model config with input dimension
    model_config["input_dim"] = training_features["input_dim"]
    
    # Initialize model
    model = AutoencoderLightningModule(
        model_config=model_config,
        learning_rate=training_config["learning_rate"],
        anomaly_threshold_percentile=training_config["anomaly_threshold_percentile"]
    )
    
    # Setup MLflow
    mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(mlflow_config.get("experiment_name", "anomaly_detection"))
    
    # Initialize MLflow logger
    mlflow_logger = MLFlowLogger(
        experiment_name=mlflow_config.get("experiment_name", "anomaly_detection"),
        tracking_uri=mlflow_config.get("tracking_uri", "mlruns"),
        tags=mlflow_config.get("tags", {})
    )
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=training_config["early_stopping_patience"],
            mode="min",
            verbose=True
        ),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath="models/checkpoints",
            filename="autoencoder-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            mode="min"
        )
    ]
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=training_config["epochs"],
        callbacks=callbacks,
        logger=mlflow_logger,
        accelerator="auto",
        devices=1,
        precision=16,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_progress_bar=True
    )
    
    # Train model
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            "model_type": "autoencoder",
            "input_dim": model_config["input_dim"],
            "hidden_dims": str(model_config["hidden_dims"]),
            "latent_dim": model_config["latent_dim"],
            "batch_size": training_config["batch_size"],
            "learning_rate": training_config["learning_rate"],
            "epochs": training_config["epochs"]
        })
        
        # Train
        trainer.fit(
            model,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["validation"]
        )
        
        # Get best model path
        best_model_path = trainer.checkpoint_callback.best_model_path
        
        # Log model
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="anomaly_autoencoder"
        )
        
        training_artifacts = {
            "model": model,
            "trainer": trainer,
            "best_model_path": best_model_path,
            "run_id": run.info.run_id,
            "anomaly_threshold": model.anomaly_threshold
        }
    
    return training_artifacts


def evaluate_model(
    training_artifacts: Dict[str, Any],
    dataloaders: Dict[str, DataLoader],
    training_features: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate trained model"""
    
    model = training_artifacts["model"]
    trainer = training_artifacts["trainer"]
    
    # Run predictions on test set
    predictions = trainer.predict(model, dataloaders["test"])
    
    # Aggregate predictions
    all_scores = []
    all_anomalies = []
    all_reconstructions = []
    all_latents = []
    
    for batch_pred in predictions:
        all_scores.append(batch_pred["anomaly_scores"])
        all_anomalies.append(batch_pred["anomalies"])
        all_reconstructions.append(batch_pred["reconstructions"])
        all_latents.append(batch_pred["latent"])
    
    anomaly_scores = torch.cat(all_scores).cpu().numpy()
    anomalies = torch.cat(all_anomalies).cpu().numpy()
    reconstructions = torch.cat(all_reconstructions).cpu().numpy()
    latent_representations = torch.cat(all_latents).cpu().numpy()
    
    # Calculate metrics if labels are available
    metrics = {}
    if "y_test" in training_features:
        y_true = training_features["y_test"]
        metrics = calculate_anomaly_metrics(y_true, anomalies, anomaly_scores)
        
        # Log metrics to MLflow
        with mlflow.start_run(run_id=training_artifacts["run_id"]):
            mlflow.log_metrics({
                "test_auc": metrics["auc"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1_score"]
            })
    
    evaluation_results = {
        "anomaly_scores": anomaly_scores,
        "anomalies": anomalies,
        "reconstructions": reconstructions,
        "latent_representations": latent_representations,
        "metrics": metrics,
        "anomaly_threshold": training_artifacts["anomaly_threshold"]
    }
    
    return evaluation_results


def save_model_artifacts(
    training_artifacts: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    preprocessing_artifacts: Dict[str, Any]
) -> Dict[str, str]:
    """Save model and related artifacts"""
    
    model = training_artifacts["model"]
    
    # Save model state
    model_path = "models/autoencoder_final.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model.hparams,
        "anomaly_threshold": evaluation_results["anomaly_threshold"],
        "feature_columns": preprocessing_artifacts["feature_columns"],
        "scaler": preprocessing_artifacts["scaler"]
    }, model_path)
    
    # Save evaluation results
    np.savez(
        "models/evaluation_results.npz",
        anomaly_scores=evaluation_results["anomaly_scores"],
        anomalies=evaluation_results["anomalies"],
        metrics=evaluation_results["metrics"]
    )
    
    logger.info(f"Model artifacts saved to {model_path}")
    
    return {
        "model_path": model_path,
        "evaluation_path": "models/evaluation_results.npz"
    }


# src/project_name/pipelines/modeling/pipeline.py
from kedro.pipeline import Pipeline, node
from .nodes import (
    prepare_dataloaders,
    train_autoencoder,
    evaluate_model,
    save_model_artifacts
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=prepare_dataloaders,
                inputs=[
                    "training_features",
                    "params:training"
                ],
                outputs="dataloaders",
                name="prepare_dataloaders",
            ),
            node(
                func=train_autoencoder,
                inputs=[
                    "dataloaders",
                    "training_features",
                    "params:model",
                    "params:training",
                    "params:mlflow"
                ],
                outputs="training_artifacts",
                name="train_model",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "training_artifacts",
                    "dataloaders",
                    "training_features"
                ],
                outputs="evaluation_results",
                name="evaluate_model",
            ),
            node(
                func=save_model_artifacts,
                inputs=[
                    "training_artifacts",
                    "evaluation_results",
                    "preprocessing_artifacts"
                ],
                outputs="saved_model_paths",
                name="save_artifacts",
            ),
        ]
    )


# src/project_name/utils/metrics.py
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
    average_precision_score
)
from typing import Dict, Any


def calculate_anomaly_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray
) -> Dict[str, Any]:
    """Calculate comprehensive metrics for anomaly detection"""
    
    # Basic metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1
    )
    
    # AUC scores
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": roc_auc,
        "pr_auc": pr_auc,
        "specificity": specificity,
        "fpr": fpr,
        "confusion_matrix": cm,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn
    }
    
    return metrics


# conf/base/parameters.yml
# Dataset configuration
dataset:
  name: "CICMalDroid2020"
  path: "data/01_raw/cicmaldroid2020.csv"

dataset_config:
  name: "CICMalDroid2020"
  path: "data/01_raw/cicmaldroid2020.csv"
  target_column: "label"
  id_column: "sample_id"
  timestamp_column: "timestamp"
  feature_columns: null  # Auto-detect

# Model configuration
model:
  hidden_dims: [256, 128, 64]
  latent_dim: 32
  activation: "relu"
  dropout_rate: 0.1

# Training configuration
training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10
  validation_split: 0.2
  anomaly_threshold_percentile: 95

# Preprocessing configuration
preprocessing:
  scaler_type: "standard"  # or "minmax"

# Feature store configuration
feature_store:
  project: "anomaly_detection"
  provider: "local"
  registry: "data/feature_store/registry.db"
  online_store_path: "data/feature_store/online_store.db"
  entity_name: "sample"
  entity_column: "sample_id"

# MLflow configuration
mlflow:
  tracking_uri: "mlruns"
  experiment_name: "cicmaldroid_anomaly_detection"
  tags:
    model_type: "autoencoder"
    dataset: "CICMalDroid2020"
    task: "anomaly_detection"

# General settings
random_state: 42


# conf/base/catalog.yml
# Data Catalog
raw_data:
  type: pandas.CSVDataSet
  filepath: data/01_raw/cicmaldroid2020.csv

validated_data:
  type: pandas.ParquetDataSet
  filepath: data/02_intermediate/validated_data.parquet

preprocessed_data:
  type: pandas.ParquetDataSet
  filepath: data/02_intermediate/preprocessed_data.parquet

data_splits:
  type: pickle.PickleDataSet
  filepath: data/03_primary/data_splits.pkl

training_features:
  type: pickle.PickleDataSet
  filepath: data/03_primary/training_features.pkl

preprocessing_artifacts:
  type: pickle.PickleDataSet
  filepath: data/03_primary/preprocessing_artifacts.pkl

feast_artifacts:
  type: pickle.PickleDataSet
  filepath: data/03_primary/feast_artifacts.pkl

training_artifacts:
  type: pickle.PickleDataSet
  filepath: data/04_model/training_artifacts.pkl

evaluation_results:
  type: pickle.PickleDataSet
  filepath: data/04_model/evaluation_results.pkl

validation_report:
  type: json.JSONDataSet
  filepath: data/08_reporting/validation_report.json


# Usage example script
"""
# Initialize and run the Kedro pipeline
from kedro.runner import SequentialRunner
from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import Pipeline
from src.project_name.pipelines import data_engineering, feature_engineering, modeling

# Create the master pipeline
def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    data_engineering_pipeline = data_engineering.create_pipeline()
    feature_engineering_pipeline = feature_engineering.create_pipeline()
    modeling_pipeline = modeling.create_pipeline()
    
    return {
        "__default__": (
            data_engineering_pipeline +
            feature_engineering_pipeline +
            modeling_pipeline
        ),
        "de": data_engineering_pipeline,
        "fe": feature_engineering_pipeline,
        "ml": modeling_pipeline,
    }

# Run the pipeline
# kedro run
# kedro run --pipeline=de  # Run only data engineering
# kedro run --pipeline=ml  # Run only modeling
"""
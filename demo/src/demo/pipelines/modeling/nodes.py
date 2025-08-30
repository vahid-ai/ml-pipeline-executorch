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
        else:
            # Create dummy labels (all zeros) to maintain consistent batch structure
            y = torch.zeros(X.shape[0], dtype=torch.long)
        
        # Always create dataset with both X and y
        dataset = TensorDataset(X, y)
        datasets[split] = dataset
    
    # Create dataloaders
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
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
    # Check if GPU is available
    import torch
    if torch.cuda.is_available():
        accelerator = "gpu"
        precision = "16-mixed"  # Use mixed precision for GPU
    else:
        accelerator = "cpu"
        precision = "32"  # Use full precision for CPU to avoid BFloat16 issues
    
    trainer = pl.Trainer(
        max_epochs=training_config["epochs"],
        callbacks=callbacks,
        logger=mlflow_logger,
        accelerator=accelerator,
        devices=1,
        precision=precision,
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
    
    # Convert to float32 first to avoid BFloat16 conversion error
    anomaly_scores = torch.cat(all_scores).float().cpu().numpy()
    anomalies = torch.cat(all_anomalies).cpu().numpy()
    reconstructions = torch.cat(all_reconstructions).float().cpu().numpy()
    latent_representations = torch.cat(all_latents).float().cpu().numpy()
    
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
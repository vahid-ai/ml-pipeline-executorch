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
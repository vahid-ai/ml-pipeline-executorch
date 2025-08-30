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
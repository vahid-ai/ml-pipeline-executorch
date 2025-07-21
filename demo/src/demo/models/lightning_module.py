import pytorch_lightning as pl
import torch
from .autoencoder import AutoEncoder
import torchmetrics

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims, lr, dropout):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoEncoder(input_dim, hidden_dims, dropout)
        self.criterion = torch.nn.MSELoss()
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()

    def forward(self, x):
        return self.model(x)[0]

    def step(self, batch, stage):
        x, _ = batch
        x_hat, _ = self.model(x)
        loss = self.criterion(x_hat, x)
        mae_metric = self.train_mae if stage == "train" else self.val_mae
        mae_metric(x_hat, x)
        self.log(f"{stage}_loss", loss)
        self.log(f"{stage}_mae", mae_metric)
        return loss

    def training_step(self, batch, _):
        return self.step(batch, "train")

    def validation_step(self, batch, _):
        self.step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

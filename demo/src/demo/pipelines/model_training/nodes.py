from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from demo.models.lightning_module import LitAutoEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch

def train_model(train_df, val_df, params, mlflow_config):
    logger = MLFlowLogger(
        experiment_name=mlflow_config["experiment"],
        tracking_uri=mlflow_config.get("tracking_uri", "mlruns"),
    )

    model = LitAutoEncoder(
        input_dim=train_df.shape[1] - 1,  # minus timestamp column
        hidden_dims=params["hidden_dims"],
        lr=params["lr"],
        dropout=params["dropout"],
    )

    def to_tensor(df):
        x = torch.tensor(df.drop(columns=["event_timestamp"]).values, dtype=torch.float32)
        return TensorDataset(x, x)

    train_loader = DataLoader(to_tensor(train_df), batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(to_tensor(val_df), batch_size=params["batch_size"])

    trainer = Trainer(
        max_epochs=params["max_epochs"],
        logger=logger,
        callbacks=[],
    )
    trainer.fit(model, train_loader, val_loader)

    # MLflow autolog logs the model; return run_id for downstream use
    return logger.experiment.get_run(logger.run_id).info.run_id

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
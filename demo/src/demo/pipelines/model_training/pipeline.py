# src/android_malware_ae/pipelines/model_training/pipeline.py
from kedro.pipeline import Pipeline, node
from .nodes import train_model

def create_pipeline(**_):
    return Pipeline(
        [
            node(
                func=train_model,
                inputs=[
                    "training_features",
                    "validation_features",
                    "params:model",
                    "params:mlflow",
                ],
                outputs="mlflow_run_id",
            )
        ]
    ) 
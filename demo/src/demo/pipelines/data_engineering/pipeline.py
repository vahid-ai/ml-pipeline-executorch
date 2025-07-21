# src/android_malware_ae/pipelines/data_engineering/pipeline.py
from kedro.pipeline import Pipeline, node
from .nodes import pull_features, split_data

def create_pipeline(**_):
    return Pipeline(
        [
            node(pull_features, None, "raw_features"),
            node(split_data, ["raw_features", "params:data"], ["training_features", "validation_features"]),
        ]
    )

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

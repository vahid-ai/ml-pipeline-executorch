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
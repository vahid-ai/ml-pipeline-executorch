"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline
from demo.pipelines.data_engineering import pipeline as de
from demo.pipelines.model_training import pipeline as mt

def register_pipelines():
    return {
        "de": de.create_pipeline(),
        "training": mt.create_pipeline(),
        "__default__": de.create_pipeline() + mt.create_pipeline(),
    }

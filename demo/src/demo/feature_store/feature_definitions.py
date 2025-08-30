from datetime import timedelta
from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float32, Int64, String
from pydantic import BaseModel
from typing import List, Dict, Any


class FeatureDefinition(BaseModel):
    """Feature definition for Feast"""
    name: str
    dtype: str
    description: str = ""


def create_feature_definitions(
    feature_names: List[str],
    entity_name: str = "sample",
    entity_column: str = "sample_id",
    source_path: str = "data/features.parquet"
) -> Dict[str, Any]:
    """Create Feast feature definitions dynamically"""
    
    # Define entity
    entity = Entity(
        name=entity_name,
        join_keys=[entity_column],
        description="Sample entity for anomaly detection",
    )
    
    # Define features using Field instead of Feature
    features = []
    for feature_name in feature_names:
        features.append(
            Field(
                name=feature_name,
                dtype=Float32,
                description=f"Feature {feature_name}"
            )
        )
    
    # Define source
    source = FileSource(
        path=source_path,
        timestamp_field="event_timestamp",
    )
    
    # Define feature view with updated API
    feature_view = FeatureView(
        name="malware_features",
        entities=[entity],
        ttl=timedelta(days=365),
        schema=features,  # Changed from 'features' to 'schema'
        online=True,
        source=source,  # Changed from 'batch_source' to 'source'
        tags={"team": "anomaly_detection"},
    )
    
    return {
        "entity": entity,
        "feature_view": feature_view,
        "features": features
    }
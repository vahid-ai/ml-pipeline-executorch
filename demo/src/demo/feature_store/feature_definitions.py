from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType
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
        value_type=ValueType.STRING,
        description="Sample entity for anomaly detection",
    )
    
    # Define features
    features = []
    for feature_name in feature_names:
        features.append(
            Feature(
                name=feature_name,
                dtype=ValueType.FLOAT,
                description=f"Feature {feature_name}"
            )
        )
    
    # Define source
    source = FileSource(
        path=source_path,
        event_timestamp_column="event_timestamp",
    )
    
    # Define feature view
    feature_view = FeatureView(
        name="malware_features",
        entities=[entity_name],
        ttl=timedelta(days=365),
        features=features,
        online=True,
        batch_source=source,
        tags={"team": "anomaly_detection"},
    )
    
    return {
        "entity": entity,
        "feature_view": feature_view,
        "features": features
    }
from datetime import timedelta
from feast import Entity, Feature, FeatureView, Field, FileSource
from feast.types import Float32
from feast.infra.offline_stores.contrib.duckdb_offline_store.duckdb_source import DuckDBSource

apk_events = Entity(name="apk_sha", join_keys=["apk_sha"])

source = DuckDBSource(
    name="cicmaldroid_source",
    path="../../data/cicmaldroid.parquet",
    timestamp_field="event_timestamp",
)

basic_stats_view = FeatureView(
    name="basic_stats",
    entities=[apk_events],
    ttl=timedelta(days=365),
    schema=[
        Field(name="cpu_usage", dtype=Float32),
        Field(name="net_bytes", dtype=Float32),
        Field(name="num_syscalls", dtype=Float32),
    ],
    online=True,
    source=source,
)

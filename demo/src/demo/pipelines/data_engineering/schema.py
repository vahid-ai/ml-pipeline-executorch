# src/android_malware_ae/pipelines/data_engineering/schema.py
import pandera as pa

# Use DataFrameSchema rather than SchemaModel to support older Pandera versions
# that may not expose `SchemaModel` at the top-level API.
#
# The resulting `EventSchema` object still exposes a `.validate(df)` method,
# so existing downstream code (e.g. in `nodes.py`) continues to work without
# any changes.

EventSchema = pa.DataFrameSchema(
    {
        "event_timestamp": pa.Column(pa.DateTime),
        "cpu_usage": pa.Column(float, checks=pa.Check.ge(0)),
        "net_bytes": pa.Column(float, checks=pa.Check.ge(0)),
        "num_syscalls": pa.Column(float, checks=pa.Check.ge(0)),
    },
    coerce=True,
    strict=True,
)

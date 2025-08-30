# kedro-mlflow-feast

This project demonstrates the integration of Kedro, MLflow, and Feast for building production-ready ML pipelines.

## Privacy and Telemetry

This repository has telemetry disabled by default:
- No `kedro-telemetry` package is installed
- Telemetry configuration has been removed from `pyproject.toml`
- `.telemetry` files are ignored in `.gitignore`

To ensure telemetry remains disabled:
1. Do not install `kedro-telemetry` package
2. Set environment variable (optional): `export KEDRO_DISABLE_TELEMETRY=true`
3. Keep `.telemetry` in your `.gitignore` file

 WARNING  There are 2 nodes that have not run.                                                                         runner.py:344
                             You can resume the pipeline run from the nearest nodes with persisted inputs by adding the following
                             argument to your previous command:
                               --from-nodes "prepare_dataloaders"

graph TD
    A[Telemetry Disabled] --> B[Removed kedro_telemetry config]
    A --> C[Added .telemetry to .gitignore]
    A --> D[No kedro-telemetry package installed]
    B --> E[pyproject.toml updated]
    C --> F[Both root and demo .gitignore updated]
    D --> G[Not in dependencies]
    
    H[ML/DS .gitignore patterns] --> I[Data files]
    H --> J[Model artifacts]
    H --> K[ML frameworks]
    H --> L[Credentials]
    
    I --> M[*.csv, *.parquet, *.h5, etc.]
    J --> N[*.pth, *.ckpt, checkpoints/]
    K --> O[mlruns/, wandb/, tensorboard/]
    L --> P[*credentials*, *.pem, .env]                              
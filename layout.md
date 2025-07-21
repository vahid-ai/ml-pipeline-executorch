android_malware_ae/
├── conf/
│   ├── base/
│   │   ├── catalog.yml          # Kedro DataSets, incl. Feast registry
│   │   ├── logging.yml
│   │   ├── parameters.yml       # hyper-params, path cfg
│   │   └── credentials.yml      # (mlflow, s3, etc.)
│   └── local/…
├── data/                        # raw CICMalDroid files sit here
├── src/
│   ├── android_malware_ae/
│   │   ├── __init__.py
│   │   ├── config.py            # Pydantic project settings
│   │   ├── models/
│   │   │   ├── autoencoder.py
│   │   │   └── lightning_module.py
│   │   ├── pipelines/
│   │   │   ├── data_engineering/
│   │   │   │   ├── nodes.py
│   │   │   │   └── pipeline.py
│   │   │   └── model_training/
│   │   │       ├── nodes.py
│   │   │       └── pipeline.py
│   │   └── feature_repo/
│   │       ├── feature_repo.yaml
│   │       └── features.py
├── tests/
└── README.md

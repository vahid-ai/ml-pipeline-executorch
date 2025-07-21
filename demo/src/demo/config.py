from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Path to the local Feast repository (relative to the project root)
    feast_repo: str = "demo/feature_repo"
    feast_project: str = "cicmaldroid_offline"
    mlflow_tracking_uri: str = "mlruns"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AE_")


# Create a singleton-like instance that can be imported elsewhere
settings = Settings()

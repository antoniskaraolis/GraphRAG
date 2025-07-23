from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    GRAPH_PATH: str       # path to data/processed/graph.graphml (or GraphML)
    OPENAI_API_KEY: str   # set via env or .env
    CACHE_TTL: int = 300  # seconds for any caching

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
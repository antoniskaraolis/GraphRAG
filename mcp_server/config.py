# mcp_server/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENV: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    GRAPH_PATH: str = "data/processed/graph.graphml"
    OPENAI_API_KEY: str = ""
    CACHE_TTL: int = 300
    
    class Config:
        env_file = ".env"

settings = Settings()

"""create config variables from .env file"""

from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    DATA_DIR: str = Field(default="./data", alias="DATA_DIR")
    SUPPORTED_EXTENSIONS: list[str] = Field(default =["html", "pdf"], alias="SUPPORTED_EXTENSIONS")

    # Chunking
    EMBED_MODEL_ID: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBED_MODEL_ID")
    CHUNK_SIZE: int = Field(default=1000, alias="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=200, alias="CHUNK_OVERLAP")


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

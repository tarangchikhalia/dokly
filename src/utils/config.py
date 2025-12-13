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

    # Vector Store
    VECTOR_STORE_PATH: str = Field(default="./vector_store", alias="VECTOR_STORE_PATH")
    VECTOR_STORE_COLLECTION: str = Field(default="documents", alias="VECTOR_STORE_COLLECTION")

    LLM_PROVIDER: str = Field(default="Ollama", alias="LLM_PROVIDER")
    LLM_MODEL: str = Field(default="llama2", alias="LLM_MODEL")
    LLM_URL: str = Field(default="", alias="LLM_URL")
    LLM_API_KEY: str = Field(default="", alias="LLM_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

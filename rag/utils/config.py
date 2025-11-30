"""create config variables from .env file"""

from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    
    DATA_DIR: str = Field(default="./data", alias="DATA_DIR")
    SUPPORTED_EXTENSIONS: list[str] = Field(default =["html", "pdf"], alias="SUPPORTED_EXTENSIONS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

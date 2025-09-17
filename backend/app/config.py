import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # API Configuration
    app_name: str = Field(default="Mini Mako Backend", description="Application name")
    version: str = Field(default="0.1.0", description="API version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    
    # Gemini Configuration
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    gemini_model: str = Field(default="gemini-1.5-flash", description="Default Gemini model")
    
    # OpenAI Configuration (for compatibility)
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_base_url: Optional[str] = Field(default=None, description="OpenAI base URL for custom endpoints")
    
    # Hugging Face Configuration
    huggingface_api_key: Optional[str] = Field(default=None, description="Hugging Face API key")
    huggingface_model: str = Field(default="microsoft/DialoGPT-medium", description="Default Hugging Face model")
    device: str = Field(default="auto", description="Device for model inference (auto, cpu, cuda)")
    
    # Model Configuration
    max_tokens: int = Field(default=8000, description="Default maximum tokens")
    temperature: float = Field(default=0.7, description="Default temperature")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

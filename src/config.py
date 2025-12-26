"""
Configuration management for AI Assistant.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    nvidia_api_key: Optional[str] = Field(default=None, alias="NVIDIA_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    
    # Model Configuration
    default_model: str = "nvidia/nemotron-3-nano-30b"
    nvidia_api_base: str = "https://integrate.api.nvidia.com/v1"
    
    # Application Settings
    app_name: str = "AI Assistant"
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = Field(default=None)
    knowledge_base_dir: Path = Field(default=None)
    
    # Agent Settings
    max_conversation_history: int = 50
    default_temperature: float = 0.7
    max_tokens: int = 1024
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    class Config:
        env_file = ".env"
        extra = "ignore"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set derived paths
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.knowledge_base_dir is None:
            self.knowledge_base_dir = self.data_dir / "knowledge_base"
    
    @property
    def has_nvidia_api(self) -> bool:
        """Check if NVIDIA API is configured."""
        return self.nvidia_api_key is not None and len(self.nvidia_api_key) > 0
    
    @property
    def has_openai_api(self) -> bool:
        """Check if OpenAI API is configured."""
        return self.openai_api_key is not None and len(self.openai_api_key) > 0


# Global settings instance
settings = Settings()

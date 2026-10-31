"""Configuration management for the AI Video Pipeline."""
import os
from pathlib import Path
from typing import Any, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class HiggsfieldConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.higgsfield.ai/v1"
    model: str = "higgsfield-video-v1"


class InstagramConfig(BaseModel):
    app_id: str = ""
    app_secret: str = ""
    access_token: str = ""
    business_account_id: str = ""
    refresh_token: str = ""


class PipelineConfig(BaseModel):
    video_output_dir: str = "./output/videos"
    temp_dir: str = "./temp"
    max_video_duration_seconds: int = 60
    default_aspect_ratio: str = "9:16"
    default_fps: int = 30


class InteractionConfig(BaseModel):
    auto_reply_enabled: bool = True
    auto_reply_templates: list[str] = Field(default_factory=lambda: [
        "Thanks for watching! 🙏",
        "Appreciate the support! ❤️",
        "Glad you liked it! 😊"
    ])
    engagement_keywords: list[str] = Field(default_factory=lambda: [
        "love", "amazing", "great", "awesome", "cool", "fire", "🔥", "❤️"
    ])
    engagement_cooldown_minutes: int = 30
    analytics_retention_days: int = 90


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "./logs/pipeline.log"
    max_size_mb: int = 10
    backup_count: int = 5


class Config(BaseModel):
    higgsfield: HiggsfieldConfig = Field(default_factory=HiggsfieldConfig)
    instagram: InstagramConfig = Field(default_factory=InstagramConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    interactions: InteractionConfig = Field(default_factory=InteractionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file."""
    config = Config()
    
    # Default config file paths to check
    search_paths = [
        config_path,
        "config.local.yaml",
        "config.yaml",
        os.path.expanduser("~/.config/insta_ai/config.yaml"),
    ]
    
    for path in search_paths:
        if path and Path(path).exists():
            with open(path, "r") as f:
                data = yaml.safe_load(f)
                if data:
                    config = Config(**data)
                    break
    
    # Override with environment variables
    _apply_env_overrides(config)
    
    return config


def _apply_env_overrides(config: Config) -> None:
    """Apply environment variable overrides."""
    env_mappings = {
        "HIGGSFIELD_API_KEY": ("higgsfield", "api_key"),
        "HIGGSFIELD_BASE_URL": ("higgsfield", "base_url"),
        "HIGGSFIELD_MODEL": ("higgsfield", "model"),
        "INSTAGRAM_APP_ID": ("instagram", "app_id"),
        "INSTAGRAM_APP_SECRET": ("instagram", "app_secret"),
        "INSTAGRAM_ACCESS_TOKEN": ("instagram", "access_token"),
        "INSTAGRAM_BUSINESS_ACCOUNT_ID": ("instagram", "business_account_id"),
        "INSTAGRAM_REFRESH_TOKEN": ("instagram", "refresh_token"),
    }
    
    for env_var, (section, key) in env_mappings.items():
        value = os.getenv(env_var)
        if value:
            section_obj = getattr(config, section)
            setattr(section_obj, key, value)


def save_config(config: Config, path: str = "config.local.yaml") -> None:
    """Save configuration to YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
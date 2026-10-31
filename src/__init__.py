"""AI Video Pipeline for Instagram - Generate, Approve, Post, Engage."""
from .config import load_config, save_config, Config
from .higgsfield_client import HiggsfieldClient, VideoGenerationRequest, VideoGenerationResponse
from .local_video_client import LocalVideoClient, VideoGenerationRequest as LocalVideoRequest, VideoGenerationResponse as LocalVideoResponse, list_available_models
from .instagram_client import InstagramClient, MediaContainer, PostInsights
from .interaction_manager import InteractionManager, manage_post_interactions
from .pipeline import VideoPipeline, PipelineResult, PipelineStage, ConsoleApprovalCallback

__version__ = "0.1.0"
__all__ = [
    "load_config",
    "save_config",
    "Config",
    "HiggsfieldClient",
    "LocalVideoClient",
    "VideoGenerationRequest",
    "VideoGenerationResponse",
    "LocalVideoRequest",
    "LocalVideoResponse",
    "list_available_models",
    "InstagramClient",
    "MediaContainer",
    "PostInsights",
    "InteractionManager",
    "manage_post_interactions",
    "VideoPipeline",
    "PipelineResult",
    "PipelineStage",
    "ConsoleApprovalCallback",
]
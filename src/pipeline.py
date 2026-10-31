"""Main pipeline orchestrator for AI video generation and Instagram posting."""
import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Union
import logging

from .config import load_config, PipelineConfig, Config
from .higgsfield_client import HiggsfieldClient, VideoGenerationRequest as HiggsfieldRequest, VideoGenerationResponse as HiggsfieldResponse, VideoStatus
from .local_video_client import LocalVideoClient, VideoGenerationRequest as LocalRequest, VideoGenerationResponse as LocalResponse, list_available_models
from .instagram_client import InstagramClient, MediaContainer
from .interaction_manager import InteractionManager, manage_post_interactions

logger = logging.getLogger(__name__)


# Unified video request/response types
@dataclass
class VideoGenerationRequest:
    """Unified video generation request."""
    prompt: str
    negative_prompt: str = ""
    duration_seconds: int = 10
    aspect_ratio: str = "9:16"
    fps: int = 8
    seed: Optional[int] = None
    model: Optional[str] = None
    num_frames: Optional[int] = None
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    init_image: Optional[str] = None
    motion_scale: float = 1.0


@dataclass
class VideoGenerationResponse:
    """Unified video generation response."""
    video_id: str
    status: VideoStatus
    local_path: Optional[str] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class VideoBackend(str, Enum):
    """Video generation backend."""
    HIGGSFIELD = "higgsfield"
    LOCAL = "local"


def _to_higgsfield_request(req: VideoGenerationRequest) -> HiggsfieldRequest:
    return HiggsfieldRequest(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        duration_seconds=req.duration_seconds,
        aspect_ratio=req.aspect_ratio,
        fps=req.fps,
        seed=req.seed,
        model=req.model,
    )


def _to_local_request(req: VideoGenerationRequest) -> LocalRequest:
    return LocalRequest(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        duration_seconds=req.duration_seconds,
        aspect_ratio=req.aspect_ratio,
        fps=req.fps,
        seed=req.seed,
        model=req.model,
        num_frames=req.num_frames,
        guidance_scale=req.guidance_scale,
        num_inference_steps=req.num_inference_steps,
        init_image=req.init_image,
        motion_scale=req.motion_scale,
    )


def _normalize_response(resp: Union[HiggsfieldResponse, LocalResponse]) -> VideoGenerationResponse:
    if isinstance(resp, HiggsfieldResponse):
        return VideoGenerationResponse(
            video_id=resp.video_id,
            status=resp.status,
            local_path=resp.metadata.get("local_path") if resp.metadata else None,
            video_url=resp.video_url,
            error=resp.error,
            metadata=resp.metadata,
        )
    else:
        return VideoGenerationResponse(
            video_id=resp.video_id,
            status=resp.status,
            local_path=resp.local_path,
            error=resp.error,
            metadata=resp.metadata,
        )


class PipelineStage(str, Enum):
    INIT = "init"
    GENERATING_VIDEO = "generating_video"
    VIDEO_READY = "video_ready"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    MANAGING_INTERACTIONS = "managing_interactions"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineState:
    pipeline_id: str
    stage: PipelineStage = PipelineStage.INIT
    topic: str = ""
    script: str = ""
    video_request: Optional[VideoGenerationRequest] = None
    video_response: Optional[VideoGenerationResponse] = None
    video_path: Optional[Path] = None
    media_container: Optional[MediaContainer] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_stage(self, stage: PipelineStage):
        self.stage = stage
        self.updated_at = datetime.now()
        logger.info(f"Pipeline {self.pipeline_id}: {stage.value}")


@dataclass
class PipelineResult:
    success: bool
    pipeline_id: str
    video_path: Optional[Path] = None
    instagram_url: Optional[str] = None
    media_id: Optional[str] = None
    error: Optional[str] = None
    interaction_results: Optional[Dict] = None


class ApprovalCallback:
    """Base class for approval callbacks."""
    
    async def request_approval(
        self, 
        pipeline_id: str, 
        video_path: Path, 
        topic: str,
        script: str
    ) -> bool:
        """Request user approval. Return True to approve, False to reject."""
        raise NotImplementedError


class ConsoleApprovalCallback(ApprovalCallback):
    """Console-based approval (for CLI usage)."""
    
    async def request_approval(
        self, 
        pipeline_id: str, 
        video_path: Path, 
        topic: str,
        script: str
    ) -> bool:
        print(f"\n{'='*60}")
        print(f"VIDEO READY FOR APPROVAL")
        print(f"{'='*60}")
        print(f"Pipeline ID: {pipeline_id}")
        print(f"Topic: {topic}")
        print(f"Script: {script[:200]}{'...' if len(script) > 200 else ''}")
        print(f"Video: {video_path}")
        print(f"{'='*60}")
        
        while True:
            choice = input("Approve and post to Instagram? (y/n): ").strip().lower()
            if choice in ('y', 'yes'):
                return True
            elif choice in ('n', 'no'):
                return False
            else:
                print("Please enter 'y' or 'n'")


class WebApprovalCallback(ApprovalCallback):
    """Web-based approval (for future web UI integration)."""
    
    def __init__(self, callback_url: str, auth_token: Optional[str] = None):
        self.callback_url = callback_url
        self.auth_token = auth_token
        self._pending_approvals: Dict[str, asyncio.Future] = {}
    
    async def request_approval(
        self, 
        pipeline_id: str, 
        video_path: Path, 
        topic: str,
        script: str
    ) -> bool:
        future = asyncio.Future()
        self._pending_approvals[pipeline_id] = future
        
        logger.info(f"Approval requested for {pipeline_id} - waiting for web UI")
        
        try:
            return await asyncio.wait_for(future, timeout=3600)
        except asyncio.TimeoutError:
            logger.warning(f"Approval timeout for {pipeline_id}")
            return False
        finally:
            self._pending_approvals.pop(pipeline_id, None)
    
    def resolve_approval(self, pipeline_id: str, approved: bool):
        future = self._pending_approvals.get(pipeline_id)
        if future and not future.done():
            future.set_result(approved)


class VideoPipeline:
    """Main pipeline orchestrator supporting multiple video backends."""
    
    def __init__(
        self,
        config: Optional[Config] = None,
        approval_callback: Optional[ApprovalCallback] = None,
        progress_callback: Optional[Callable[[PipelineState], None]] = None,
        video_backend: VideoBackend = VideoBackend.LOCAL,
        local_model_preset: str = "animatediff-sd15",
    ):
        self.config = config or load_config()
        self.approval_callback = approval_callback or ConsoleApprovalCallback()
        self.progress_callback = progress_callback
        self.video_backend = video_backend
        self.local_model_preset = local_model_preset
        
        # Initialize clients
        self.higgsfield = HiggsfieldClient(self.config.higgsfield)
        self.local_video: Optional[LocalVideoClient] = None
        self.instagram = InstagramClient(self.config.instagram)
        self.interactions = InteractionManager(
            self.instagram, 
            self.config.interactions
        )
        
        # Ensure output directories exist
        Path(self.config.pipeline.video_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.pipeline.temp_dir).mkdir(parents=True, exist_ok=True)
    
    def _get_local_client(self) -> LocalVideoClient:
        """Lazy initialize local video client."""
        if self.local_video is None:
            self.local_video = LocalVideoClient(
                model_preset=self.local_model_preset,
                output_dir=self.config.pipeline.video_output_dir,
            )
        return self.local_video
    
    def _notify_progress(self, state: PipelineState):
        if self.progress_callback:
            try:
                self.progress_callback(state)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    async def run(
        self,
        topic: str,
        script: str,
        duration_seconds: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        auto_post: bool = False,
        manage_interactions: bool = True,
        **generation_kwargs
    ) -> PipelineResult:
        """Run the complete pipeline."""
        pipeline_id = str(uuid.uuid4())[:8]
        state = PipelineState(
            pipeline_id=pipeline_id,
            topic=topic,
            script=script
        )
        
        self._notify_progress(state)
        
        try:
            # Stage 1: Generate video
            state.update_stage(PipelineStage.GENERATING_VIDEO)
            self._notify_progress(state)
            
            video_response = await self._generate_video(state, duration_seconds, aspect_ratio, **generation_kwargs)
            state.video_response = video_response
            
            if video_response.status != VideoStatus.COMPLETED:
                state.update_stage(PipelineStage.FAILED)
                state.error = video_response.error or "Video generation failed"
                return self._create_result(state, success=False)
            
            state.video_path = Path(video_response.local_path or "")
            state.update_stage(PipelineStage.VIDEO_READY)
            self._notify_progress(state)
            
            # Stage 2: Approval
            if not auto_post:
                state.update_stage(PipelineStage.AWAITING_APPROVAL)
                self._notify_progress(state)
                
                approved = await self.approval_callback.request_approval(
                    pipeline_id, state.video_path, topic, script
                )
                
                if not approved:
                    state.update_stage(PipelineStage.REJECTED)
                    state.error = "User rejected the video"
                    return self._create_result(state, success=False)
                
                state.update_stage(PipelineStage.APPROVED)
                self._notify_progress(state)
            
            # Stage 3: Upload to Instagram
            state.update_stage(PipelineStage.UPLOADING)
            self._notify_progress(state)
            
            media_container = await self._upload_to_instagram(state)
            state.media_container = media_container
            
            if media_container.status != "FINISHED":
                state.update_stage(PipelineStage.FAILED)
                state.error = media_container.error or "Media processing failed"
                return self._create_result(state, success=False)
            
            state.update_stage(PipelineStage.PUBLISHING)
            self._notify_progress(state)
            
            published = await self.instagram.publish_media(media_container.id)
            state.media_container = published
            
            if published.status != "FINISHED" and not published.permalink:
                state.update_stage(PipelineStage.FAILED)
                state.error = published.error or "Publishing failed"
                return self._create_result(state, success=False)
            
            state.update_stage(PipelineStage.PUBLISHED)
            self._notify_progress(state)
            
            # Stage 4: Manage interactions
            if manage_interactions and published.id:
                state.update_stage(PipelineStage.MANAGING_INTERACTIONS)
                self._notify_progress(state)
                
                interaction_results = await manage_post_interactions(
                    published.id, self.instagram
                )
                state.metadata["interactions"] = interaction_results
            
            state.update_stage(PipelineStage.COMPLETED)
            self._notify_progress(state)
            
            return self._create_result(state, success=True)
            
        except Exception as e:
            logger.exception(f"Pipeline {pipeline_id} failed")
            state.update_stage(PipelineStage.FAILED)
            state.error = str(e)
            self._notify_progress(state)
            return self._create_result(state, success=False)
        
        finally:
            await self._cleanup()
    
    async def _generate_video(
        self,
        state: PipelineState,
        duration_seconds: Optional[int],
        aspect_ratio: Optional[str],
        **kwargs
    ) -> VideoGenerationResponse:
        """Generate video using selected backend."""
        request = VideoGenerationRequest(
            prompt=state.script or state.topic,
            duration_seconds=duration_seconds or self.config.pipeline.max_video_duration_seconds,
            aspect_ratio=aspect_ratio or self.config.pipeline.default_aspect_ratio,
            fps=self.config.pipeline.default_fps,
            **kwargs
        )
        state.video_request = request
        
        # Generate unique output path
        output_filename = f"{state.pipeline_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = Path(self.config.pipeline.video_output_dir) / output_filename
        
        if self.video_backend == VideoBackend.HIGGSFIELD:
            hf_request = _to_higgsfield_request(request)
            response = await self.higgsfield.generate_and_download(hf_request, output_path)
        else:
            local_client = self._get_local_client()
            local_request = _to_local_request(request)
            response = await local_client.generate_and_download(local_request, output_path)
        
        return _normalize_response(response)
    
    async def _upload_to_instagram(self, state: PipelineState) -> MediaContainer:
        """Upload video to Instagram as Reel."""
        caption = self._generate_caption(state.topic, state.script)
        
        return await self.instagram.post_reel(
            video_path=state.video_path,
            caption=caption
        )
    
    def _generate_caption(self, topic: str, script: str) -> str:
        hashtags = "#AI #AIVideo #GeneratedContent #Tech #Innovation"
        return f"{topic}\n\n{script[:500]}\n\n{hashtags}"
    
    def _create_result(self, state: PipelineState, success: bool) -> PipelineResult:
        return PipelineResult(
            success=success,
            pipeline_id=state.pipeline_id,
            video_path=state.video_path,
            instagram_url=state.media_container.permalink if state.media_container else None,
            media_id=state.media_container.id if state.media_container else None,
            error=state.error,
            interaction_results=state.metadata.get("interactions")
        )
    
    async def _cleanup(self):
        """Cleanup resources."""
        await self.higgsfield.close()
        if self.local_video:
            await self.local_video.close()
        await self.instagram.close()
        await self.interactions.stop_monitoring()
    
    # Convenience methods
    
    async def generate_video_only(
        self,
        topic: str,
        script: str,
        **kwargs
    ) -> VideoGenerationResponse:
        """Generate video without posting."""
        pipeline_id = str(uuid.uuid4())[:8]
        state = PipelineState(pipeline_id=pipeline_id, topic=topic, script=script)
        return await self._generate_video(state, None, None, **kwargs)
    
    async def post_existing_video(
        self,
        video_path: Path,
        caption: str = "",
        manage_interactions: bool = True
    ) -> PipelineResult:
        """Post an existing video to Instagram."""
        pipeline_id = str(uuid.uuid4())[:8]
        state = PipelineState(
            pipeline_id=pipeline_id,
            video_path=video_path,
            topic="Existing Video"
        )
        
        try:
            media_container = await self.instagram.post_reel(video_path, caption)
            state.media_container = media_container
            
            if media_container.status != "FINISHED":
                return PipelineResult(
                    success=False,
                    pipeline_id=pipeline_id,
                    error=media_container.error
                )
            
            published = await self.instagram.publish_media(media_container.id)
            
            interaction_results = None
            if manage_interactions and published.id:
                interaction_results = await manage_post_interactions(
                    published.id, self.instagram
                )
            
            return PipelineResult(
                success=True,
                pipeline_id=pipeline_id,
                video_path=video_path,
                instagram_url=published.permalink,
                media_id=published.id,
                interaction_results=interaction_results
            )
        except Exception as e:
            return PipelineResult(
                success=False,
                pipeline_id=pipeline_id,
                error=str(e)
            )
        finally:
            await self._cleanup()


# Convenience function
async def run_pipeline(
    topic: str,
    script: str,
    auto_post: bool = False,
    video_backend: VideoBackend = VideoBackend.LOCAL,
    local_model_preset: str = "animatediff-sd15",
    **kwargs
) -> PipelineResult:
    """Simple one-liner to run the full pipeline."""
    pipeline = VideoPipeline(
        video_backend=video_backend,
        local_model_preset=local_model_preset
    )
    return await pipeline.run(topic, script, auto_post=auto_post, **kwargs)
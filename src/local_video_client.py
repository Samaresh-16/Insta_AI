"""
Local AI Video Generation Client using Diffusers (Hugging Face).
Optimized for 6GB VRAM GPUs like RTX 4050.
Drop-in replacement for HiggsfieldClient.
"""
import asyncio
import os
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import tempfile

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class VideoStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class VideoGenerationRequest:
    prompt: str
    negative_prompt: str = ""
    duration_seconds: int = 10
    aspect_ratio: str = "9:16"
    fps: int = 8  # Lower FPS for local generation
    seed: Optional[int] = None
    model: Optional[str] = None
    num_frames: Optional[int] = None
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    # For image-to-video
    init_image: Optional[str] = None  # Path to initial image
    motion_scale: float = 1.0  # For AnimateDiff


@dataclass
class VideoGenerationResponse:
    video_id: str
    status: VideoStatus
    local_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class LocalVideoClient:
    """
    Local video generation using Hugging Face Diffusers.
    Supports multiple models optimized for 6GB VRAM.
    """
    
    # Pre-configured model presets for 6GB VRAM
    MODEL_PRESETS = {
        "animatediff-sd15": {
            "name": "AnimateDiff + SD 1.5",
            "type": "animatediff",
            "base_model": "runwayml/stable-diffusion-v1-5",
            "motion_module": "guoyww/animatediff-motion-adapter-v1-5-2",
            "lora_path": None,
            "default_frames": 16,
            "default_fps": 8,
            "resolution": (512, 512),
            "vram_gb": 4.5,
        },
        "animatediff-sdxl": {
            "name": "AnimateDiff + SDXL (tight fit)",
            "type": "animatediff",
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "motion_module": "guoyww/animatediff-motion-adapter-v1-5-2",
            "lora_path": None,
            "default_frames": 16,
            "default_fps": 8,
            "resolution": (512, 512),  # Keep low for VRAM
            "vram_gb": 5.5,
        },
        "modelscope-t2v": {
            "name": "ModelScope Text-to-Video",
            "type": "modelscope",
            "model_id": "damo-vilab/text-to-video-ms-1.7b",
            "default_frames": 16,
            "default_fps": 8,
            "resolution": (320, 240),  # ModelScope native resolution
            "vram_gb": 5.5,
        },
        "zeroscope-v2": {
            "name": "ZeroScope v2",
            "type": "zeroscope",
            "model_id": "cerspense/zeroscope_v2_576w",
            "default_frames": 24,
            "default_fps": 8,
            "resolution": (576, 320),
            "vram_gb": 5.8,
        },
        "svd-base": {
            "name": "Stable Video Diffusion (img2vid)",
            "type": "svd",
            "model_id": "stabilityai/stable-video-diffusion-img2vid",
            "default_frames": 14,
            "default_fps": 7,
            "resolution": (576, 1024),
            "vram_gb": 5.5,
            "requires_image": True,
        },
        "svd-xt": {
            "name": "Stable Video Diffusion XT (img2vid, 25 frames)",
            "type": "svd",
            "model_id": "stabilityai/stable-video-diffusion-img2vid-xt",
            "default_frames": 25,
            "default_fps": 7,
            "resolution": (576, 1024),
            "vram_gb": 6.2,  # Might need CPU offload
            "requires_image": True,
        },
    }
    
    def __init__(
        self,
        model_preset: str = "animatediff-sd15",
        device: str = "cuda",
        output_dir: str = "./output/videos",
        enable_optimizations: bool = True,
    ):
        """
        Initialize local video client.
        
        Args:
            model_preset: One of MODEL_PRESETS keys
            device: "cuda" or "cpu"
            output_dir: Directory to save generated videos
            enable_optimizations: Apply VRAM optimizations
        """
        self.model_preset = model_preset
        self.device = device if torch.cuda.is_available() else "cpu"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_optimizations = enable_optimizations
        
        self._pipeline = None
        self._model_config = self.MODEL_PRESETS.get(model_preset, self.MODEL_PRESETS["animatediff-sd15"])
        
        # Validate VRAM
        if self.device == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU VRAM: {vram_gb:.1f} GB")
            if vram_gb < self._model_config["vram_gb"] - 0.5:
                logger.warning(f"Model may need {self._model_config['vram_gb']}GB VRAM, you have {vram_gb:.1f}GB. Enabling aggressive offload.")
                self.enable_optimizations = True
    
    def _load_pipeline(self):
        """Lazy load the pipeline with optimizations."""
        if self._pipeline is not None:
            return self._pipeline
        
        config = self._model_config
        model_type = config["type"]
        
        logger.info(f"Loading {config['name']}...")
        
        if model_type == "animatediff":
            self._pipeline = self._load_animatediff(config)
        elif model_type == "modelscope":
            self._pipeline = self._load_modelscope(config)
        elif model_type == "zeroscope":
            self._pipeline = self._load_zeroscope(config)
        elif model_type == "svd":
            self._pipeline = self._load_svd(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Apply optimizations
        if self.enable_optimizations and self.device == "cuda":
            self._apply_optimizations()
        
        logger.info(f"Pipeline loaded on {self.device}")
        return self._pipeline
    
    def _load_animatediff(self, config: Dict) -> Any:
        """Load AnimateDiff pipeline."""
        from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
        from diffusers.utils import export_to_video
        
        # Load motion adapter
        adapter = MotionAdapter.from_pretrained(
            config["motion_module"],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Load base pipeline
        pipe = AnimateDiffPipeline.from_pretrained(
            config["base_model"],
            motion_adapter=adapter,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            variant="fp16" if self.device == "cuda" else None,
        )
        
        # Use Euler scheduler for better quality
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        
        pipe.to(self.device)
        return pipe
    
    def _load_modelscope(self, config: Dict) -> Any:
        """Load ModelScope T2V pipeline."""
        from diffusers import TextToVideoSDPipeline
        
        pipe = TextToVideoSDPipeline.from_pretrained(
            config["model_id"],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            variant="fp16" if self.device == "cuda" else None,
        )
        pipe.to(self.device)
        return pipe
    
    def _load_zeroscope(self, config: Dict) -> Any:
        """Load ZeroScope pipeline."""
        from diffusers import TextToVideoSDPipeline
        
        pipe = TextToVideoSDPipeline.from_pretrained(
            config["model_id"],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            variant="fp16" if self.device == "cuda" else None,
        )
        pipe.to(self.device)
        return pipe
    
    def _load_svd(self, config: Dict) -> Any:
        """Load Stable Video Diffusion pipeline."""
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video
        
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            config["model_id"],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            variant="fp16" if self.device == "cuda" else None,
        )
        pipe.to(self.device)
        return pipe
    
    def _apply_optimizations(self):
        """Apply VRAM optimizations for 6GB GPUs."""
        pipe = self._pipeline
        
        # Enable xformers if available
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
        except Exception as e:
            logger.warning(f"xformers not available: {e}")
        
        # Enable VAE slicing
        try:
            pipe.enable_vae_slicing()
            logger.info("Enabled VAE slicing")
        except Exception:
            pass
        
        # Enable model CPU offload (most aggressive)
        try:
            pipe.enable_model_cpu_offload()
            logger.info("Enabled model CPU offload")
        except Exception:
            pass
        
        # Enable sequential CPU offload (even more aggressive)
        try:
            pipe.enable_sequential_cpu_offload()
            logger.info("Enabled sequential CPU offload")
        except Exception:
            pass
        
        # Enable attention slicing
        try:
            pipe.enable_attention_slicing("auto")
            logger.info("Enabled attention slicing")
        except Exception:
            pass
    
    def _calculate_frames(self, duration_seconds: int, fps: int) -> int:
        """Calculate number of frames from duration and FPS."""
        return min(duration_seconds * fps, 64)  # Cap at 64 frames
    
    def _export_video(self, frames: List[Image.Image], output_path: Path, fps: int):
        """Export frames to video file."""
        try:
            # Try using diffusers export utility first
            from diffusers.utils import export_to_video
            export_to_video(frames, str(output_path), fps=fps)
        except Exception:
            # Fallback: use imageio
            import imageio
            imageio.mimsave(str(output_path), frames, fps=fps, codec="libx264", quality=8)
    
    async def generate_video(self, request: VideoGenerationRequest) -> VideoGenerationResponse:
        """Generate video from text prompt (or image for SVD)."""
        video_id = str(uuid.uuid4())[:8]
        config = self._model_config
        
        logger.info(f"Generating video {video_id} with {config['name']}")
        
        try:
            pipe = self._load_pipeline()
            
            # Prepare generation parameters
            num_frames = request.num_frames or self._calculate_frames(
                request.duration_seconds, 
                request.fps or config["default_fps"]
            )
            
            width, height = config["resolution"]
            
            # Adjust for aspect ratio
            if request.aspect_ratio == "9:16":  # Portrait (Reels)
                width, height = height, width
            elif request.aspect_ratio == "16:9":  # Landscape
                pass  # Keep as is
            elif request.aspect_ratio == "1:1":  # Square
                width = height = min(width, height)
            
            # Set seed
            generator = None
            if request.seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(request.seed)
            
            # Generate based on model type
            model_type = config["type"]
            
            if model_type == "animatediff":
                frames = await self._generate_animatediff(
                    pipe, request, num_frames, width, height, generator
                )
            elif model_type in ["modelscope", "zeroscope"]:
                frames = await self._generate_text_to_video(
                    pipe, request, num_frames, width, height, generator
                )
            elif model_type == "svd":
                frames = await self._generate_svd(
                    pipe, request, num_frames, generator
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Save video
            output_filename = f"local_{video_id}_{request.prompt[:30].replace(' ', '_')}.mp4"
            output_path = self.output_dir / output_filename
            
            fps = request.fps or config["default_fps"]
            self._export_video(frames, output_path, fps)
            
            logger.info(f"Video saved to {output_path}")
            
            return VideoGenerationResponse(
                video_id=video_id,
                status=VideoStatus.COMPLETED,
                local_path=str(output_path),
                metadata={
                    "model": config["name"],
                    "frames": len(frames),
                    "fps": fps,
                    "resolution": f"{width}x{height}",
                    "duration_seconds": len(frames) / fps,
                }
            )
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM: {e}")
            # Try to clear cache and retry with more aggressive settings
            torch.cuda.empty_cache()
            return VideoGenerationResponse(
                video_id=video_id,
                status=VideoStatus.FAILED,
                error="CUDA out of memory. Try reducing frames/resolution or enable more optimizations."
            )
        except Exception as e:
            logger.exception(f"Generation failed: {e}")
            return VideoGenerationResponse(
                video_id=video_id,
                status=VideoStatus.FAILED,
                error=str(e)
            )
    
    async def _generate_animatediff(
        self, pipe, request: VideoGenerationRequest,
        num_frames: int, width: int, height: int,
        generator
    ) -> List[Image.Image]:
        """Generate using AnimateDiff."""
        # Run in thread pool to not block event loop
        def _generate():
            with torch.inference_mode(), torch.autocast("cuda" if self.device == "cuda" else "cpu"):
                result = pipe(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    num_frames=num_frames,
                    width=width,
                    height=height,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    generator=generator,
                    motion_scale=request.motion_scale,
                )
            return result.frames[0]
        
        return await asyncio.to_thread(_generate)
    
    async def _generate_text_to_video(
        self, pipe, request: VideoGenerationRequest,
        num_frames: int, width: int, height: int,
        generator
    ) -> List[Image.Image]:
        """Generate using ModelScope/ZeroScope (text-to-video)."""
        def _generate():
            with torch.inference_mode(), torch.autocast("cuda" if self.device == "cuda" else "cpu"):
                result = pipe(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    num_frames=num_frames,
                    width=width,
                    height=height,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    generator=generator,
                )
            return result.frames[0]
        
        return await asyncio.to_thread(_generate)
    
    async def _generate_svd(
        self, pipe, request: VideoGenerationRequest,
        num_frames: int, generator
    ) -> List[Image.Image]:
        """Generate using Stable Video Diffusion (image-to-video)."""
        if not request.init_image:
            raise ValueError("SVD requires init_image parameter (path to starting image)")
        
        # Load and resize initial image
        init_image = Image.open(request.init_image).convert("RGB")
        config = self._model_config
        init_image = init_image.resize(config["resolution"])
        
        def _generate():
            with torch.inference_mode(), torch.autocast("cuda" if self.device == "cuda" else "cpu"):
                result = pipe(
                    image=init_image,
                    num_frames=num_frames,
                    num_inference_steps=request.num_inference_steps,
                    motion_bucket_id=127,  # Default motion
                    noise_aug_strength=0.02,
                    generator=generator,
                )
            return result.frames[0]
        
        return await asyncio.to_thread(_generate)
    
    async def generate_and_download(
        self, 
        request: VideoGenerationRequest,
        output_path: Path
    ) -> VideoGenerationResponse:
        """Generate video and save to specific path."""
        response = await self.generate_video(request)
        
        if response.status == VideoStatus.COMPLETED and response.local_path:
            # Copy to requested path
            import shutil
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(response.local_path, output_path)
            response.metadata["local_path"] = str(output_path)
            response.local_path = str(output_path)
        
        return response
    
    async def check_status(self, video_id: str) -> VideoGenerationResponse:
        """Local generation is synchronous, so status is always completed/failed."""
        return VideoGenerationResponse(
            video_id=video_id,
            status=VideoStatus.COMPLETED,
            error="Local generation is synchronous"
        )
    
    async def wait_for_completion(
        self, video_id: str, poll_interval: int = 1, max_wait: int = 300
    ) -> VideoGenerationResponse:
        return await self.check_status(video_id)
    
    async def close(self):
        """Clean up GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience function
async def generate_video(
    prompt: str,
    output_path: Path,
    model_preset: str = "animatediff-sd15",
    duration_seconds: int = 10,
    aspect_ratio: str = "9:16",
    **kwargs
) -> VideoGenerationResponse:
    """Simple one-liner to generate video locally."""
    client = LocalVideoClient(model_preset=model_preset)
    request = VideoGenerationRequest(
        prompt=prompt,
        duration_seconds=duration_seconds,
        aspect_ratio=aspect_ratio,
        **kwargs
    )
    try:
        return await client.generate_and_download(request, output_path)
    finally:
        await client.close()


# Model selection helper
def list_available_models() -> Dict[str, Dict]:
    """List all available model presets with VRAM requirements."""
    return LocalVideoClient.MODEL_PRESETS


def estimate_vram_usage(model_preset: str) -> float:
    """Estimate VRAM usage for a model preset."""
    return LocalVideoClient.MODEL_PRESETS.get(model_preset, {}).get("vram_gb", 0)
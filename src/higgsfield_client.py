"""Higgsfield API client for AI video generation."""
import asyncio
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import httpx
import logging
from dataclasses import dataclass
from enum import Enum

from .config import load_config, HiggsfieldConfig

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
    fps: int = 30
    seed: Optional[int] = None
    model: Optional[str] = None


@dataclass
class VideoGenerationResponse:
    video_id: str
    status: VideoStatus
    video_url: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class HiggsfieldClient:
    """Client for interacting with Higgsfield AI video generation API."""
    
    def __init__(self, config: Optional[HiggsfieldConfig] = None):
        self.config = config or load_config().higgsfield
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=300.0,  # 5 minute timeout for video generation
        )
    
    async def generate_video(self, request: VideoGenerationRequest) -> VideoGenerationResponse:
        """Generate a video from a text prompt."""
        payload = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "duration": request.duration_seconds,
            "aspect_ratio": request.aspect_ratio,
            "fps": request.fps,
            "model": request.model or self.config.model,
        }
        
        if request.seed is not None:
            payload["seed"] = request.seed
        
        logger.info(f"Generating video with prompt: {request.prompt[:100]}...")
        
        try:
            response = await self.client.post("/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            
            video_id = data.get("id") or data.get("video_id")
            if not video_id:
                raise ValueError("No video ID returned from API")
            
            return VideoGenerationResponse(
                video_id=video_id,
                status=VideoStatus.PENDING,
                metadata=data
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Higgsfield API error: {e.response.status_code} - {e.response.text}")
            return VideoGenerationResponse(
                video_id="",
                status=VideoStatus.FAILED,
                error=f"API error: {e.response.status_code}"
            )
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            return VideoGenerationResponse(
                video_id="",
                status=VideoStatus.FAILED,
                error=str(e)
            )
    
    async def check_status(self, video_id: str) -> VideoGenerationResponse:
        """Check the status of a video generation job."""
        try:
            response = await self.client.get(f"/generate/{video_id}")
            response.raise_for_status()
            data = response.json()
            
            status_map = {
                "pending": VideoStatus.PENDING,
                "processing": VideoStatus.PROCESSING,
                "completed": VideoStatus.COMPLETED,
                "failed": VideoStatus.FAILED,
            }
            
            return VideoGenerationResponse(
                video_id=video_id,
                status=status_map.get(data.get("status", "pending"), VideoStatus.PENDING),
                video_url=data.get("video_url") or data.get("url"),
                error=data.get("error"),
                metadata=data
            )
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return VideoGenerationResponse(
                video_id=video_id,
                status=VideoStatus.FAILED,
                error=str(e)
            )
    
    async def wait_for_completion(
        self, 
        video_id: str, 
        poll_interval: int = 10,
        max_wait: int = 600
    ) -> VideoGenerationResponse:
        """Wait for video generation to complete."""
        elapsed = 0
        while elapsed < max_wait:
            status = await self.check_status(video_id)
            
            if status.status == VideoStatus.COMPLETED:
                logger.info(f"Video {video_id} completed successfully")
                return status
            elif status.status == VideoStatus.FAILED:
                logger.error(f"Video {video_id} failed: {status.error}")
                return status
            
            logger.info(f"Video {video_id} status: {status.status.value} (elapsed: {elapsed}s)")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        
        return VideoGenerationResponse(
            video_id=video_id,
            status=VideoStatus.FAILED,
            error="Timeout waiting for video generation"
        )
    
    async def download_video(self, video_url: str, output_path: Path) -> bool:
        """Download generated video to local file."""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.get(video_url)
                response.raise_for_status()
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                logger.info(f"Video downloaded to {output_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to download video: {e}")
            return False
    
    async def generate_and_download(
        self, 
        request: VideoGenerationRequest,
        output_path: Path
    ) -> VideoGenerationResponse:
        """Generate video and download it in one call."""
        # Start generation
        response = await self.generate_video(request)
        if response.status == VideoStatus.FAILED:
            return response
        
        # Wait for completion
        response = await self.wait_for_completion(response.video_id)
        if response.status != VideoStatus.COMPLETED or not response.video_url:
            return response
        
        # Download video
        success = await self.download_video(response.video_url, output_path)
        if not success:
            return VideoGenerationResponse(
                video_id=response.video_id,
                status=VideoStatus.FAILED,
                error="Failed to download video"
            )
        
        response.metadata["local_path"] = str(output_path)
        return response
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience function for simple usage
async def generate_video(
    prompt: str,
    output_path: Path,
    duration_seconds: int = 10,
    aspect_ratio: str = "9:16",
    **kwargs
) -> VideoGenerationResponse:
    """Simple one-liner to generate and download a video."""
    config = load_config()
    request = VideoGenerationRequest(
        prompt=prompt,
        duration_seconds=duration_seconds,
        aspect_ratio=aspect_ratio,
        **kwargs
    )
    
    async with HiggsfieldClient(config.higgsfield) as client:
        return await client.generate_and_download(request, output_path)
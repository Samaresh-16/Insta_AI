"""Instagram Meta Graph API client for posting and managing content."""
import asyncio
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import httpx
import logging

from .config import load_config, InstagramConfig

logger = logging.getLogger(__name__)

GRAPH_API_VERSION = "v20.0"
GRAPH_API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"


@dataclass
class MediaContainer:
    id: str
    status: str = "pending"
    media_type: str = "REELS"
    permalink: Optional[str] = None
    error: Optional[str] = None


@dataclass
class PostInsights:
    media_id: str
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    reach: int = 0
    engagement_rate: float = 0.0
    fetched_at: datetime = field(default_factory=datetime.now)


@dataclass
class Comment:
    id: str
    text: str
    username: str
    user_id: str
    timestamp: datetime
    like_count: int = 0
    replied: bool = False


class InstagramClient:
    """Client for Instagram Meta Graph API operations."""
    
    def __init__(self, config: Optional[InstagramConfig] = None):
        self.config = config or load_config().instagram
        self.client = httpx.AsyncClient(
            base_url=GRAPH_API_BASE,
            timeout=60.0,
        )
        self._access_token = self.config.access_token
    
    @property
    def access_token(self) -> str:
        return self._access_token
    
    @access_token.setter
    def access_token(self, token: str):
        self._access_token = token
        # Update client headers
        self.client.headers["Authorization"] = f"Bearer {token}"
    
    async def _request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        files: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make an authenticated request to the Graph API."""
        params = params or {}
        params["access_token"] = self._access_token
        
        if files:
            # For multipart uploads, don't set Content-Type
            headers = {"Authorization": f"Bearer {self._access_token}"}
        else:
            headers = {"Content-Type": "application/json"}
        
        try:
            response = await self.client.request(
                method, endpoint, params=params, json=data, files=files, headers=headers
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json() if e.response.content else str(e)
            logger.error(f"Instagram API error ({e.response.status_code}): {error_detail}")
            raise InstagramAPIError(e.response.status_code, error_detail)
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    async def validate_token(self) -> bool:
        """Validate the current access token."""
        try:
            await self._request("GET", "/me", params={"fields": "id,name"})
            return True
        except Exception:
            return False
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get Instagram business account info."""
        return await self._request(
            "GET", 
            f"/{self.config.business_account_id}",
            params={"fields": "id,username,name,profile_picture_url,media_count,followers_count"}
        )
    
    # --- Video/Reel Upload ---
    
    async def create_media_container(
        self,
        video_path: Path,
        caption: str = "",
        cover_url: Optional[str] = None,
        location_id: Optional[str] = None,
        share_to_feed: bool = True,
    ) -> MediaContainer:
        """Create a media container for a Reel (video)."""
        # Step 1: Upload video to get a video ID
        video_id = await self._upload_video(video_path)
        
        # Step 2: Create media container
        data = {
            "media_type": "REELS",
            "video_id": video_id,
            "caption": caption,
            "share_to_feed": share_to_feed,
        }
        
        if cover_url:
            data["cover_url"] = cover_url
        if location_id:
            data["location_id"] = location_id
        
        result = await self._request(
            "POST", 
            f"/{self.config.business_account_id}/media", 
            data=data
        )
        
        return MediaContainer(
            id=result["id"],
            status="pending",
            media_type="REELS"
        )
    
    async def _upload_video(self, video_path: Path) -> str:
        """Upload video file and return video ID."""
        # For large videos, use resumable upload
        file_size = video_path.stat().st_size
        
        if file_size > 100 * 1024 * 1024:  # > 100MB
            return await self._resumable_upload(video_path)
        else:
            return await self._simple_upload(video_path)
    
    async def _simple_upload(self, video_path: Path) -> str:
        """Simple video upload for smaller files."""
        with open(video_path, "rb") as f:
            files = {
                "video_file": (video_path.name, f, "video/mp4"),
            }
            data = {"media_type": "REELS"}
            
            result = await self._request(
                "POST",
                f"/{self.config.business_account_id}/media",
                data=data,
                files=files
            )
            return result["id"]
    
    async def _resumable_upload(self, video_path: Path) -> str:
        """Resumable upload for large videos."""
        file_size = video_path.stat().st_size
        chunk_size = 4 * 1024 * 1024  # 4MB chunks
        
        # Start upload session
        start_result = await self._request(
            "POST",
            f"/{self.config.business_account_id}/media",
            data={
                "media_type": "REELS",
                "upload_type": "resumable",
                "file_size": file_size,
            }
        )
        
        upload_session_id = start_result["upload_session_id"]
        video_id = start_result["video_id"]
        
        # Upload chunks
        with open(video_path, "rb") as f:
            offset = 0
            while offset < file_size:
                chunk = f.read(chunk_size)
                chunk_end = offset + len(chunk)
                
                headers = {
                    "Authorization": f"Bearer {self._access_token}",
                    "Content-Range": f"bytes {offset}-{chunk_end - 1}/{file_size}",
                }
                
                await self.client.put(
                    f"{GRAPH_API_BASE}/{upload_session_id}",
                    content=chunk,
                    headers=headers
                )
                
                offset = chunk_end
                logger.info(f"Uploaded {offset}/{file_size} bytes")
        
        # Finalize
        await self._request(
            "POST",
            f"/{upload_session_id}/finish",
            data={"video_id": video_id}
        )
        
        return video_id
    
    async def check_media_status(self, container_id: str) -> MediaContainer:
        """Check the status of a media container."""
        result = await self._request(
            "GET",
            f"/{container_id}",
            params={"fields": "status_code,media_type,permalink,error_message"}
        )
        
        return MediaContainer(
            id=container_id,
            status=result.get("status_code", "unknown"),
            media_type=result.get("media_type", "REELS"),
            permalink=result.get("permalink"),
            error=result.get("error_message")
        )
    
    async def publish_media(self, container_id: str) -> MediaContainer:
        """Publish a media container."""
        result = await self._request(
            "POST",
            f"/{self.config.business_account_id}/media_publish",
            data={"creation_id": container_id}
        )
        
        # Get the published media ID
        media_id = result.get("id")
        if media_id:
            container = await self.check_media_status(media_id)
            container.id = media_id
            return container
        
        return MediaContainer(id=container_id, status="published")
    
    async def wait_for_processing(
        self, 
        container_id: str, 
        poll_interval: int = 10,
        max_wait: int = 300
    ) -> MediaContainer:
        """Wait for media container to finish processing."""
        elapsed = 0
        while elapsed < max_wait:
            container = await self.check_media_status(container_id)
            
            if container.status == "FINISHED":
                logger.info(f"Media {container_id} ready to publish")
                return container
            elif container.status in ["ERROR", "EXPIRED"]:
                logger.error(f"Media {container_id} failed: {container.error}")
                return container
            
            logger.info(f"Media {container_id} status: {container.status} (elapsed: {elapsed}s)")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        
        return MediaContainer(
            id=container_id,
            status="TIMEOUT",
            error="Timeout waiting for media processing"
        )
    
    async def post_reel(
        self,
        video_path: Path,
        caption: str = "",
        cover_url: Optional[str] = None,
        location_id: Optional[str] = None,
    ) -> MediaContainer:
        """Complete workflow: upload, process, and publish a Reel."""
        logger.info(f"Starting Reel upload: {video_path.name}")
        
        # Create container
        container = await self.create_media_container(
            video_path, caption, cover_url, location_id
        )
        
        # Wait for processing
        container = await self.wait_for_processing(container.id)
        
        if container.status != "FINISHED":
            return container
        
        # Publish
        logger.info(f"Publishing Reel {container.id}")
        return await self.publish_media(container.id)
    
    # --- Insights & Analytics ---
    
    async def get_media_insights(self, media_id: str) -> PostInsights:
        """Get insights for a published media."""
        metrics = [
            "views", "likes", "comments", "shares", 
            "saved", "reach", "engagement"
        ]
        
        result = await self._request(
            "GET",
            f"/{media_id}/insights",
            params={"metric": ",".join(metrics), "period": "lifetime"}
        )
        
        insights_data = {item["name"]: item["values"][0]["value"] for item in result.get("data", [])}
        
        views = insights_data.get("views", 0)
        likes = insights_data.get("likes", 0)
        comments = insights_data.get("comments", 0)
        shares = insights_data.get("shares", 0)
        saves = insights_data.get("saved", 0)
        reach = insights_data.get("reach", 0)
        engagement = insights_data.get("engagement", 0)
        
        engagement_rate = (engagement / reach * 100) if reach > 0 else 0
        
        return PostInsights(
            media_id=media_id,
            views=views,
            likes=likes,
            comments=comments,
            shares=shares,
            saves=saves,
            reach=reach,
            engagement_rate=round(engagement_rate, 2)
        )
    
    async def get_recent_media(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Get recent media from the account."""
        result = await self._request(
            "GET",
            f"/{self.config.business_account_id}/media",
            params={
                "fields": "id,media_type,media_url,permalink,caption,timestamp,like_count,comments_count",
                "limit": limit
            }
        )
        return result.get("data", [])
    
    # --- Comments Management ---
    
    async def get_media_comments(self, media_id: str, limit: int = 50) -> List[Comment]:
        """Get comments on a media."""
        result = await self._request(
            "GET",
            f"/{media_id}/comments",
            params={
                "fields": "id,text,username,user_id,timestamp,like_count",
                "limit": limit,
                "order": "chronological"
            }
        )
        
        comments = []
        for item in result.get("data", []):
            comments.append(Comment(
                id=item["id"],
                text=item["text"],
                username=item["username"],
                user_id=item["user_id"],
                timestamp=datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")),
                like_count=item.get("like_count", 0)
            ))
        
        return comments
    
    async def reply_to_comment(self, comment_id: str, message: str) -> Dict[str, Any]:
        """Reply to a comment."""
        return await self._request(
            "POST",
            f"/{comment_id}/replies",
            data={"message": message}
        )
    
    async def like_comment(self, comment_id: str) -> Dict[str, Any]:
        """Like a comment."""
        return await self._request("POST", f"/{comment_id}/likes")
    
    async def hide_comment(self, comment_id: str) -> Dict[str, Any]:
        """Hide a comment."""
        return await self._request("POST", f"/{comment_id}", data={"hidden": True})
    
    async def delete_comment(self, comment_id: str) -> Dict[str, Any]:
        """Delete a comment."""
        return await self._request("DELETE", f"/{comment_id}")
    
    # --- Engagement ---
    
    async def like_media(self, media_id: str) -> Dict[str, Any]:
        """Like a media post."""
        return await self._request("POST", f"/{media_id}/likes")
    
    async def comment_on_media(self, media_id: str, message: str) -> Dict[str, Any]:
        """Comment on a media post."""
        return await self._request(
            "POST",
            f"/{media_id}/comments",
            data={"message": message}
        )
    
    # --- Token Management ---
    
    async def refresh_access_token(self) -> str:
        """Refresh long-lived access token."""
        if not self.config.refresh_token:
            raise ValueError("No refresh token configured")
        
        result = await self._request(
            "GET",
            "/oauth/access_token",
            params={
                "grant_type": "fb_exchange_token",
                "client_id": self.config.app_id,
                "client_secret": self.config.app_secret,
                "fb_exchange_token": self.config.refresh_token,
            }
        )
        
        new_token = result.get("access_token")
        if new_token:
            self.access_token = new_token
            logger.info("Access token refreshed successfully")
        
        return new_token
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class InstagramAPIError(Exception):
    """Instagram API error."""
    def __init__(self, status_code: int, detail: Any):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Instagram API error {status_code}: {detail}")
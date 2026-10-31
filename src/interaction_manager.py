"""Interaction management for Instagram: comments, engagement, analytics."""
import asyncio
import json
import random
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import logging

from .config import load_config, InteractionConfig
from .instagram_client import InstagramClient, Comment

logger = logging.getLogger(__name__)


@dataclass
class InteractionLog:
    id: Optional[int] = None
    timestamp: datetime = None
    action_type: str = ""  # reply, like, comment, hide, delete
    target_id: str = ""    # comment_id or media_id
    target_type: str = ""  # comment, media
    content: str = ""      # reply text, etc.
    status: str = "success"  # success, failed, skipped
    error: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class EngagementRule:
    keywords: List[str]
    response_templates: List[str]
    probability: float = 0.3  # 30% chance to engage
    cooldown_minutes: int = 30
    max_per_user_per_day: int = 2


class InteractionDatabase:
    """SQLite database for tracking interactions and analytics."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    content TEXT,
                    status TEXT NOT NULL,
                    error TEXT
                );
                
                CREATE TABLE IF NOT EXISTS media_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    media_id TEXT NOT NULL UNIQUE,
                    views INTEGER DEFAULT 0,
                    likes INTEGER DEFAULT 0,
                    comments INTEGER DEFAULT 0,
                    shares INTEGER DEFAULT 0,
                    saves INTEGER DEFAULT 0,
                    reach INTEGER DEFAULT 0,
                    engagement_rate REAL DEFAULT 0,
                    fetched_at TEXT NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS user_engagement (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    last_interaction TEXT NOT NULL,
                    interaction_count INTEGER DEFAULT 1,
                    date TEXT NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_interactions_target 
                    ON interactions(target_id, target_type);
                CREATE INDEX IF NOT EXISTS idx_media_analytics_media 
                    ON media_analytics(media_id);
                CREATE INDEX IF NOT EXISTS idx_user_engagement_user 
                    ON user_engagement(user_id, date);
            """)
    
    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def log_interaction(self, interaction: InteractionLog) -> int:
        with self._get_conn() as conn:
            cursor = conn.execute("""
                INSERT INTO interactions (timestamp, action_type, target_id, target_type, content, status, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                interaction.timestamp.isoformat(),
                interaction.action_type,
                interaction.target_id,
                interaction.target_type,
                interaction.content,
                interaction.status,
                interaction.error
            ))
            return cursor.lastrowid
    
    def get_recent_interactions(
        self, 
        target_id: str, 
        target_type: str, 
        hours: int = 24
    ) -> List[InteractionLog]:
        cutoff = datetime.now() - timedelta(hours=hours)
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM interactions 
                WHERE target_id = ? AND target_type = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (target_id, target_type, cutoff.isoformat())).fetchall()
            return [self._row_to_interaction(row) for row in rows]
    
    def save_media_analytics(self, analytics) -> None:
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO media_analytics 
                (media_id, views, likes, comments, shares, saves, reach, engagement_rate, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analytics.media_id,
                analytics.views,
                analytics.likes,
                analytics.comments,
                analytics.shares,
                analytics.saves,
                analytics.reach,
                analytics.engagement_rate,
                analytics.fetched_at.isoformat()
            ))
    
    def get_media_analytics(self, media_id: str) -> Optional[Dict]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM media_analytics WHERE media_id = ?", (media_id,)
            ).fetchone()
            return dict(row) if row else None
    
    def get_all_media_analytics(self, days: int = 30) -> List[Dict]:
        cutoff = datetime.now() - timedelta(days=days)
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM media_analytics 
                WHERE fetched_at > ?
                ORDER BY fetched_at DESC
            """, (cutoff.isoformat(),)).fetchall()
            return [dict(row) for row in rows]
    
    def can_engage_with_user(self, user_id: str, username: str, max_per_day: int = 2) -> bool:
        today = datetime.now().date().isoformat()
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT interaction_count FROM user_engagement 
                WHERE user_id = ? AND date = ?
            """, (user_id, today)).fetchone()
            
            if row and row["interaction_count"] >= max_per_day:
                return False
            return True
    
    def record_user_engagement(self, user_id: str, username: str) -> None:
        today = datetime.now().date().isoformat()
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO user_engagement (user_id, username, last_interaction, interaction_count, date)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(user_id, date) DO UPDATE SET
                    interaction_count = interaction_count + 1,
                    last_interaction = ?
            """, (user_id, username, datetime.now().isoformat(), today, datetime.now().isoformat()))
    
    def _row_to_interaction(self, row: sqlite3.Row) -> InteractionLog:
        return InteractionLog(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            action_type=row["action_type"],
            target_id=row["target_id"],
            target_type=row["target_type"],
            content=row["content"] or "",
            status=row["status"],
            error=row["error"] or ""
        )


class InteractionManager:
    """Manages Instagram interactions: auto-replies, engagement, analytics."""
    
    def __init__(
        self, 
        instagram_client: InstagramClient,
        config: Optional[InteractionConfig] = None,
        db_path: Optional[Path] = None
    ):
        self.client = instagram_client
        self.config = config or load_config().interactions
        self.db = InteractionDatabase(db_path or Path("./data/interactions.db"))
        self.engagement_rules: List[EngagementRule] = []
        self._setup_default_rules()
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    def _setup_default_rules(self):
        """Setup default engagement rules from config."""
        self.engagement_rules.append(EngagementRule(
            keywords=self.config.engagement_keywords,
            response_templates=self.config.auto_reply_templates,
            probability=0.3,
            cooldown_minutes=self.config.engagement_cooldown_minutes,
            max_per_user_per_day=2
        ))
    
    def add_engagement_rule(self, rule: EngagementRule):
        """Add a custom engagement rule."""
        self.engagement_rules.append(rule)
    
    async def auto_reply_to_comments(
        self, 
        media_id: str,
        custom_templates: Optional[List[str]] = None
    ) -> List[InteractionLog]:
        """Auto-reply to comments on a media."""
        logs = []
        templates = custom_templates or self.config.auto_reply_templates
        
        try:
            comments = await self.client.get_media_comments(media_id)
            
            for comment in comments:
                # Skip if already replied
                if comment.replied:
                    continue
                
                # Check cooldown
                recent = self.db.get_recent_interactions(
                    comment.id, "comment", 
                    hours=self.config.engagement_cooldown_minutes / 60
                )
                if recent:
                    continue
                
                # Check daily limit per user
                if not self.db.can_engage_with_user(comment.user_id, comment.username):
                    logger.info(f"Daily limit reached for user {comment.username}")
                    continue
                
                # Select random template
                reply_text = random.choice(templates)
                
                # Send reply
                try:
                    await self.client.reply_to_comment(comment.id, reply_text)
                    
                    log = InteractionLog(
                        action_type="reply",
                        target_id=comment.id,
                        target_type="comment",
                        content=reply_text,
                        status="success"
                    )
                    self.db.log_interaction(log)
                    self.db.record_user_engagement(comment.user_id, comment.username)
                    logs.append(log)
                    
                    logger.info(f"Replied to comment by {comment.username}: {reply_text}")
                    
                except Exception as e:
                    log = InteractionLog(
                        action_type="reply",
                        target_id=comment.id,
                        target_type="comment",
                        content=reply_text,
                        status="failed",
                        error=str(e)
                    )
                    self.db.log_interaction(log)
                    logs.append(log)
                    logger.error(f"Failed to reply to comment {comment.id}: {e}")
                
                # Small delay to avoid rate limits
                await asyncio.sleep(2)
        
        except Exception as e:
            logger.error(f"Auto-reply failed for media {media_id}: {e}")
        
        return logs
    
    async def engage_with_comments(
        self, 
        media_id: str,
        like_probability: float = 0.5
    ) -> List[InteractionLog]:
        """Like comments that match engagement keywords."""
        logs = []
        
        try:
            comments = await self.client.get_media_comments(media_id)
            
            for comment in comments:
                # Check if comment matches keywords
                text_lower = comment.text.lower()
                matched = any(kw.lower() in text_lower for kw in self.config.engagement_keywords)
                
                if not matched:
                    continue
                
                # Check if already liked
                recent = self.db.get_recent_interactions(comment.id, "comment", hours=24)
                if any(r.action_type == "like" for r in recent):
                    continue
                
                # Random probability
                if random.random() > like_probability:
                    continue
                
                try:
                    await self.client.like_comment(comment.id)
                    
                    log = InteractionLog(
                        action_type="like",
                        target_id=comment.id,
                        target_type="comment",
                        content="Liked comment",
                        status="success"
                    )
                    self.db.log_interaction(log)
                    logs.append(log)
                    
                    logger.info(f"Liked comment by {comment.username}")
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    log = InteractionLog(
                        action_type="like",
                        target_id=comment.id,
                        target_type="comment",
                        status="failed",
                        error=str(e)
                    )
                    self.db.log_interaction(log)
                    logs.append(log)
        
        except Exception as e:
            logger.error(f"Comment engagement failed for media {media_id}: {e}")
        
        return logs
    
    async def fetch_and_store_analytics(self, media_id: str) -> Optional[Dict]:
        """Fetch insights for a media and store in database."""
        try:
            insights = await self.client.get_media_insights(media_id)
            self.db.save_media_analytics(insights)
            return asdict(insights)
        except Exception as e:
            logger.error(f"Failed to fetch analytics for {media_id}: {e}")
            return None
    
    async def fetch_all_recent_analytics(self, limit: int = 25) -> List[Dict]:
        """Fetch analytics for all recent media."""
        media_list = await self.client.get_recent_media(limit)
        results = []
        
        for media in media_list:
            analytics = await self.fetch_and_store_analytics(media["id"])
            if analytics:
                analytics["media_info"] = media
                results.append(analytics)
            await asyncio.sleep(1)  # Rate limiting
        
        return results
    
    async def start_monitoring(
        self, 
        media_id: str, 
        interval_minutes: int = 15,
        auto_reply: bool = True,
        engage_comments: bool = True
    ):
        """Start background monitoring for a media."""
        if self._running:
            logger.warning("Monitoring already running")
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(media_id, interval_minutes, auto_reply, engage_comments)
        )
        logger.info(f"Started monitoring media {media_id}")
    
    async def stop_monitoring(self):
        """Stop background monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped monitoring")
    
    async def _monitor_loop(
        self, 
        media_id: str, 
        interval_minutes: int,
        auto_reply: bool,
        engage_comments: bool
    ):
        """Background monitoring loop."""
        while self._running:
            try:
                if auto_reply:
                    await self.auto_reply_to_comments(media_id)
                
                if engage_comments:
                    await self.engage_with_comments(media_id)
                
                # Fetch analytics periodically
                await self.fetch_and_store_analytics(media_id)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            await asyncio.sleep(interval_minutes * 60)
    
    def get_interaction_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get interaction statistics."""
        cutoff = datetime.now() - timedelta(days=days)
        
        with self.db._get_conn() as conn:
            # Total interactions by type
            type_stats = conn.execute("""
                SELECT action_type, status, COUNT(*) as count
                FROM interactions
                WHERE timestamp > ?
                GROUP BY action_type, status
            """, (cutoff.isoformat(),)).fetchall()
            
            # Daily breakdown
            daily_stats = conn.execute("""
                SELECT date(timestamp) as day, action_type, COUNT(*) as count
                FROM interactions
                WHERE timestamp > ?
                GROUP BY date(timestamp), action_type
                ORDER BY day DESC
            """, (cutoff.isoformat(),)).fetchall()
            
            return {
                "by_type": [dict(row) for row in type_stats],
                "daily": [dict(row) for row in daily_stats],
                "period_days": days
            }
    
    def export_analytics_report(self, output_path: Path, days: int = 30) -> None:
        """Export analytics report as JSON."""
        analytics = self.db.get_all_media_analytics(days)
        stats = self.get_interaction_stats(days)
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "period_days": days,
            "media_analytics": analytics,
            "interaction_stats": stats,
            "summary": {
                "total_media": len(analytics),
                "total_views": sum(a["views"] for a in analytics),
                "total_likes": sum(a["likes"] for a in analytics),
                "total_comments": sum(a["comments"] for a in analytics),
                "avg_engagement_rate": (
                    sum(a["engagement_rate"] for a in analytics) / len(analytics) 
                    if analytics else 0
                )
            }
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Analytics report exported to {output_path}")


# Convenience function
async def manage_post_interactions(
    media_id: str,
    instagram_client: InstagramClient,
    auto_reply: bool = True,
    engage_comments: bool = True,
    fetch_analytics: bool = True
) -> Dict[str, Any]:
    """One-shot interaction management for a post."""
    config = load_config()
    manager = InteractionManager(instagram_client, config.interactions)
    
    results = {}
    
    if auto_reply:
        results["replies"] = await manager.auto_reply_to_comments(media_id)
    
    if engage_comments:
        results["engagements"] = await manager.engage_with_comments(media_id)
    
    if fetch_analytics:
        results["analytics"] = await manager.fetch_and_store_analytics(media_id)
    
    return results
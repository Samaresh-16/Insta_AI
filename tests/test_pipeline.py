"""Tests for the AI Video Pipeline."""
import pytest
from pathlib import Path
import tempfile
import yaml

from src.config import load_config, save_config, Config, HiggsfieldConfig, InstagramConfig


class TestConfig:
    """Test configuration loading and saving."""
    
    def test_default_config(self):
        config = Config()
        assert config.higgsfield.base_url == "https://api.higgsfield.ai/v1"
        assert config.pipeline.default_aspect_ratio == "9:16"
        assert config.interactions.auto_reply_enabled is True
    
    def test_save_and_load_config(self, tmp_path):
        config = Config()
        config.higgsfield.api_key = "test-key"
        config.instagram.app_id = "test-app-id"
        
        config_file = tmp_path / "test_config.yaml"
        save_config(config, str(config_file))
        
        loaded = load_config(str(config_file))
        assert loaded.higgsfield.api_key == "test-key"
        assert loaded.instagram.app_id == "test-app-id"
    
    def test_config_yaml_structure(self):
        """Test that config.yaml has valid structure."""
        with open("config.yaml") as f:
            data = yaml.safe_load(f)
        
        assert "higgsfield" in data
        assert "instagram" in data
        assert "pipeline" in data
        assert "interactions" in data
        assert "logging" in data


class TestHiggsfieldClient:
    """Test Higgsfield client (requires API key for full tests)."""
    
    def test_client_creation(self):
        from src.higgsfield_client import HiggsfieldClient
        config = HiggsfieldConfig(api_key="test")
        client = HiggsfieldClient(config)
        assert client.config.api_key == "test"
    
    def test_video_request_creation(self):
        from src.higgsfield_client import VideoGenerationRequest
        request = VideoGenerationRequest(
            prompt="Test video",
            duration_seconds=10,
            aspect_ratio="9:16"
        )
        assert request.prompt == "Test video"
        assert request.duration_seconds == 10


class TestInstagramClient:
    """Test Instagram client (requires credentials for full tests)."""
    
    def test_client_creation(self):
        from src.instagram_client import InstagramClient
        config = InstagramConfig(
            app_id="test",
            app_secret="test",
            access_token="test",
            business_account_id="123"
        )
        client = InstagramClient(config)
        assert client.config.app_id == "test"


class TestInteractionManager:
    """Test interaction management."""
    
    def test_database_creation(self, tmp_path):
        from src.interaction_manager import InteractionDatabase, InteractionLog
        from datetime import datetime
        
        db = InteractionDatabase(tmp_path / "test.db")
        
        log = InteractionLog(
            action_type="reply",
            target_id="comment_123",
            target_type="comment",
            content="Thanks!",
            status="success"
        )
        
        log_id = db.log_interaction(log)
        assert log_id > 0
        
        # Retrieve
        recent = db.get_recent_interactions("comment_123", "comment")
        assert len(recent) == 1
        assert recent[0].content == "Thanks!"
    
    def test_engagement_rule(self):
        from src.interaction_manager import EngagementRule
        rule = EngagementRule(
            keywords=["love", "amazing"],
            response_templates=["Thanks!"],
            probability=0.5
        )
        assert "love" in rule.keywords
        assert rule.probability == 0.5


class TestPipeline:
    """Test pipeline orchestrator."""
    
    def test_pipeline_state(self):
        from src.pipeline import PipelineState, PipelineStage
        from datetime import datetime
        
        state = PipelineState(
            pipeline_id="test-123",
            topic="Test Topic",
            script="Test Script"
        )
        
        assert state.pipeline_id == "test-123"
        assert state.stage == PipelineStage.INIT
        
        state.update_stage(PipelineStage.GENERATING_VIDEO)
        assert state.stage == PipelineStage.GENERATING_VIDEO


# Run with: pytest tests/ -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
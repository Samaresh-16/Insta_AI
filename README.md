# AI Video Pipeline for Instagram

A complete Python pipeline for generating AI videos (locally or via API), managing approval workflows, posting to Instagram Reels via Meta Graph API, and handling interactions (comments, engagement, analytics).

## Features

- 🎬 **AI Video Generation** - Generate videos locally (free, runs on your GPU) or via Higgsfield API
- ✅ **Approval Workflow** - Review videos before posting (CLI or web-ready)
- 📱 **Instagram Integration** - Post Reels via official Meta Graph API
- 💬 **Interaction Management** - Auto-reply to comments, like engagement, fetch analytics
- 📊 **Analytics Dashboard** - Track views, likes, comments, engagement rates
- 🔧 **CLI Interface** - Easy command-line usage for all operations
- 🔐 **Secure Config** - Environment variable support, local config files
- 🆓 **Free Local Generation** - Run on RTX 4050 (6GB VRAM) with optimized models

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐     ┌─────────────┐
│   Topic/    │────▶│  Local / Higgs   │────▶│  Approval   │────▶│  Instagram  │
│   Script    │     │   Video Gen      │     │  (Manual)   │     │  Reels Post │
└─────────────┘     └──────────────────┘     └─────────────┘     └─────────────┘
                                                                        │
┌─────────────┐     ┌─────────────┐     ┌─────────────┐                │
│  Analytics  │◀────│ Interaction │◀────│  Published  │◀───────────────┘
│  Dashboard  │     │  Manager    │     │   Media     │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd Insta_AI
pip install -r requirements.txt
# Or with pipx/uv for isolated environment
pipx install -e .
```

### 2. Configure Credentials

Run the interactive setup:

```bash
insta-ai init
```

Or create `config.local.yaml` manually:

```yaml
higgsfield:
  api_key: "your-higgsfield-api-key"      # Optional if using local generation
  base_url: "https://api.higgsfield.ai/v1"
  model: "higgsfield-video-v1"

instagram:
  app_id: "your-meta-app-id"
  app_secret: "your-meta-app-secret"
  access_token: "your-long-lived-access-token"
  business_account_id: "your-instagram-business-account-id"
  refresh_token: "optional-refresh-token"
```

You can also use environment variables:
```bash
export HIGGSFIELD_API_KEY="your-key"       # Optional for local
export INSTAGRAM_APP_ID="your-app-id"
export INSTAGRAM_APP_SECRET="your-secret"
export INSTAGRAM_ACCESS_TOKEN="your-token"
export INSTAGRAM_BUSINESS_ACCOUNT_ID="your-account-id"
```

### 3. Validate Setup

```bash
insta-ai validate
```

This checks both API credentials and local GPU/dependencies.

### 4. Run the Pipeline

**List available local models:**
```bash
insta-ai models
```

**Full pipeline with local generation (free, runs on your GPU):**
```bash
insta-ai pipeline "AI generated cat video" --script "A cute cat coding on a laptop" --duration 15 --backend local
```

**Full pipeline with Higgsfield (paid API):**
```bash
insta-ai pipeline "AI video" --backend higgsfield
```

**Auto-post without approval:**
```bash
insta-ai pipeline "AI video" --auto-post --backend local
```

**Generate only (local):**
```bash
insta-ai generate "A futuristic city" --duration 10 --aspect 9:16 --backend local --model animatediff-sd15
```

**Generate with specific local model:**
```bash
insta-ai generate "Robot dancing" --backend local --model modelscope-t2v --fps 8 --steps 30
```

**Image-to-video (SVD):**
```bash
insta-ai generate "Camera pan across landscape" --backend local --model svd-base --init-image ./landscape.jpg
```

**Post existing video:**
```bash
insta-ai post ./output/videos/my_video.mp4 --caption "Check this out! #AI"
```

**Manage interactions for a post:**
```bash
insta-ai interactions <MEDIA_ID>
```

**View analytics:**
```bash
insta-ai analytics --days 30 --export report.json
```

## Local Video Generation Models (Free, Run on Your GPU)

Optimized for **6GB VRAM** (RTX 4050 laptop) with memory optimizations enabled:

| Model | Type | Resolution | Frames | Est. VRAM | Best For |
|-------|------|------------|--------|-----------|----------|
| `animatediff-sd15` | Text-to-video | 512×512 | 16 | ~4.5 GB | General purpose, characters |
| `animatediff-sdxl` | Text-to-video | 512×512 | 16 | ~5.5 GB | Higher quality (tight fit) |
| `modelscope-t2v` | Text-to-video | 320×240 | 16 | ~5.5 GB | Quick previews |
| `zeroscope-v2` | Text-to-video | 576×320 | 24 | ~5.8 GB | Higher resolution |
| `svd-base` | Image-to-video | 576×1024 | 14 | ~5.5 GB | Animating photos |
| `svd-xt` | Image-to-video | 576×1024 | 25 | ~6.2 GB | Longer animations |

**Memory optimizations applied automatically:**
- `enable_model_cpu_offload()` - Offloads to CPU when not in use
- `enable_vae_slicing()` - Processes VAE in chunks
- `enable_sequential_cpu_offload()` - Maximum memory savings
- `enable_xformers_memory_efficient_attention()` - If xformers installed
- FP16 precision

**To install xformers (recommended for speed):**
```bash
# For CUDA 12.1 (adjust for your version)
pip install xformers --index-url https://download.pytorch.org/whl/cu121
```

## Detailed Usage

### Higgsfield API Access (Optional)

1. Sign up at [higgsfield.ai](https://higgsfield.ai)
2. Get your API key from the dashboard
3. Add to config or use `HIGGSFIELD_API_KEY` env var

### Instagram/Meta Graph API Setup

**Prerequisites:**
- Facebook Developer Account
- Instagram Business or Creator Account (linked to Facebook Page)
- Meta App with Instagram Graph API product

**Steps:**

1. **Create Meta App:** Go to [Meta Developer Console](https://developers.facebook.com/apps/)
2. **Add Instagram Graph API:** In your app, add "Instagram Graph API" product
3. **Configure App:** Set up Graph API (required for posting)
4. **Get Access Token:**
   - Use [Graph API Explorer](https://developers.facebook.com/tools/explorer/)
   - Select your app
   - Get User Access Token with permissions: `instagram_content_publish`, `instagram_manage_insights`, `pages_show_list`, `pages_read_engagement`
   - Exchange for long-lived token (60 days)
5. **Get Business Account ID:**
   - In Graph API Explorer: `GET /me/accounts`
   - Find your Instagram Business Account ID
6. **App Review:** For production, submit for App Review to get `instagram_content_publish` permission

### Configuration Options

```yaml
pipeline:
  video_output_dir: "./output/videos"    # Where generated videos are saved
  temp_dir: "./temp"                      # Temporary files
  max_video_duration_seconds: 60          # Max video length
  default_aspect_ratio: "9:16"            # Instagram Reels format
  default_fps: 8                          # Lower for local generation

interactions:
  auto_reply_enabled: true
  auto_reply_templates:
    - "Thanks for watching! 🙏"
    - "Appreciate the support! ❤️"
  engagement_keywords: ["love", "amazing", "great", "awesome", "cool", "fire", "🔥", "❤️"]
  engagement_cooldown_minutes: 30
  analytics_retention_days: 90
```

## Programmatic Usage

```python
import asyncio
from src import VideoPipeline, load_config, VideoBackend

async def main():
    config = load_config()
    
    # Full pipeline with LOCAL generation (free!)
    pipeline = VideoPipeline(
        config, 
        video_backend=VideoBackend.LOCAL,
        local_model_preset="animatediff-sd15"
    )
    
    result = await pipeline.run(
        topic="AI Tutorial",
        script="Learn how to build AI agents in 60 seconds",
        duration_seconds=30,
        auto_post=False  # Will prompt for approval
    )
    
    if result.success:
        print(f"Posted! URL: {result.instagram_url}")
    
    # Or just generate locally
    from src import LocalVideoClient, VideoGenerationRequest as LocalRequest
    async with LocalVideoClient(model_preset="animatediff-sd15") as client:
        response = await client.generate_and_download(
            LocalRequest(prompt="Cool animation", duration_seconds=10),
            Path("./my_video.mp4")
        )

asyncio.run(main())
```

## Web UI Integration

For web-based approval, implement `ApprovalCallback`:

```python
from src.pipeline import ApprovalCallback

class WebApprovalCallback(ApprovalCallback):
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
        self.pending = {}
    
    async def request_approval(self, pipeline_id, video_path, topic, script):
        future = asyncio.Future()
        self.pending[pipeline_id] = future
        await notify_web_ui(pipeline_id, video_path, topic, script)
        return await future
    
    def resolve(self, pipeline_id, approved):
        if pipeline_id in self.pending:
            self.pending[pipeline_id].set_result(approved)

# Use with pipeline
pipeline = VideoPipeline(approval_callback=WebApprovalCallback("https://your-ui.com/webhook"))
```

## Project Structure

```
Insta_AI/
├── config.yaml              # Default configuration template
├── config.local.yaml        # Your credentials (gitignored)
├── .env.example             # Environment variable template
├── pyproject.toml           # Package configuration
├── requirements.txt         # Dependencies
├── README.md                # This file
├── run.bat / run.ps1        # Quick start scripts
├── src/
│   ├── __init__.py          # Package exports
│   ├── config.py            # Configuration management
│   ├── higgsfield_client.py # Higgsfield API client
│   ├── local_video_client.py # Local video generation (diffusers)
│   ├── instagram_client.py  # Meta Graph API client
│   ├── interaction_manager.py # Comments, engagement, analytics
│   ├── pipeline.py          # Main orchestrator
│   └── cli.py               # Command-line interface
├── output/videos/           # Generated videos (gitignored)
├── temp/                    # Temporary files (gitignored)
├── data/                    # SQLite databases (gitignored)
└── logs/                    # Log files (gitignored)
```

## Interaction Management Details

### Auto-Reply
- Replies to new comments using configurable templates
- Respects cooldown period (default 30 min)
- Limits per user per day (default 2)

### Engagement
- Likes comments containing positive keywords
- Configurable probability (default 30%)
- Tracks engagement history in SQLite

### Analytics
- Fetches insights: views, likes, comments, shares, saves, reach
- Calculates engagement rate
- Stores historical data for trending
- Exports JSON reports

## Rate Limits & Best Practices

- **Higgsfield:** Check their API docs for rate limits
- **Instagram Graph API:** 200 requests/hour per user (standard)
- **Local generation:** Limited by GPU memory, not API quotas
- **Interaction delays:** Built-in delays between actions
- **Monitoring:** Use `start_monitoring()` for background engagement

## Troubleshooting

### "Access token expired"
```bash
# Refresh token manually or use refresh_token in config
insta-ai validate  # Will show token status
```

### "CUDA out of memory"
- Reduce `--duration` or `--fps`
- Use smaller model (`modelscope-t2v` uses less VRAM)
- Ensure optimizations are enabled (they are by default)
- Close other GPU applications

### "Video generation timeout"
- Increase timeout in `HiggsfieldClient` (default 5 min)
- Check Higgsfield status page

### "Media processing stuck"
- Instagram Reels can take 2-5 minutes to process
- Pipeline polls every 10s, max 5 min wait

### "Permission denied"
- Ensure App Review approved for `instagram_content_publish`
- Check Instagram account is Business/Creator type
- Verify Page-Instagram linking

### "Model not found / download failed"
- First run downloads models (~2-6GB) from Hugging Face
- Check internet connection
- Models cached in `~/.cache/huggingface/`

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
ruff check src/

# Type check
mypy src/
```

## Security Notes

- Never commit `config.local.yaml` or `.env` files
- Use environment variables in production
- Rotate access tokens regularly
- Store refresh tokens securely
- Monitor API usage for anomalies

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit PR

## Support

- Check [Issues](https://github.com/your-repo/issues) for common problems
- Local models: Hugging Face model cards
- Higgsfield docs: https://docs.higgsfield.ai
- Meta Graph API docs: https://developers.facebook.com/docs/instagram-api
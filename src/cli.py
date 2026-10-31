"""Command-line interface for the AI Video Pipeline."""
import asyncio
import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.prompt import Prompt, Confirm

from .config import load_config, save_config, Config
from .pipeline import VideoPipeline, ConsoleApprovalCallback, PipelineResult, VideoBackend
from .higgsfield_client import HiggsfieldClient, VideoGenerationRequest as HiggsfieldRequest
from .local_video_client import LocalVideoClient, list_available_models, VideoGenerationRequest as LocalVideoRequest
from .instagram_client import InstagramClient
from .interaction_manager import InteractionManager, manage_post_interactions

app = typer.Typer(
    name="insta-ai",
    help="AI Video Pipeline for Instagram - Generate, Approve, Post, Engage",
    add_completion=False
)

console = Console()


def print_banner():
    console.print(Panel.fit(
        "[bold cyan]AI Video Pipeline for Instagram[/bold cyan]\n"
        "Generate → Approve → Post → Engage",
        border_style="cyan"
    ))


def print_error(message: str):
    console.print(f"[red]Error:[/red] {message}")


def print_success(message: str):
    console.print(f"[green]Success:[/green] {message}")


def print_info(message: str):
    console.print(f"[blue]Info:[/blue] {message}")


@app.command()
def init(
    config_path: str = typer.Option("config.local.yaml", "--path", "-p", help="Config file path")
):
    """Initialize configuration file with your credentials."""
    print_banner()
    console.print("\n[bold]Let's set up your configuration[/bold]\n")
    
    config = Config()
    
    # Higgsfield
    console.print("[cyan]Higgsfield API Configuration[/cyan]")
    console.print("Get your API key from: https://higgsfield.ai")
    config.higgsfield.api_key = Prompt.ask("Higgsfield API Key", password=True)
    config.higgsfield.base_url = Prompt.ask("Base URL", default="https://api.higgsfield.ai/v1")
    config.higgsfield.model = Prompt.ask("Model", default="higgsfield-video-v1")
    
    # Instagram
    console.print("\n[cyan]Instagram Meta Graph API Configuration[/cyan]")
    console.print("Create an app at: https://developers.facebook.com/apps/")
    console.print("You need: Instagram Business Account + App Review for posting")
    config.instagram.app_id = Prompt.ask("App ID")
    config.instagram.app_secret = Prompt.ask("App Secret", password=True)
    config.instagram.access_token = Prompt.ask("Long-lived Access Token", password=True)
    config.instagram.business_account_id = Prompt.ask("Instagram Business Account ID")
    config.instagram.refresh_token = Prompt.ask("Refresh Token (optional)", default="", password=True)
    
    # Save
    save_config(config, config_path)
    print_success(f"Configuration saved to {config_path}")
    console.print("\n[dim]You can also set these via environment variables:[/dim]")
    console.print("  HIGGSFIELD_API_KEY, INSTAGRAM_APP_ID, INSTAGRAM_APP_SECRET,")
    console.print("  INSTAGRAM_ACCESS_TOKEN, INSTAGRAM_BUSINESS_ACCOUNT_ID")


@app.command()
def generate(
    topic: str = typer.Argument(..., help="Video topic"),
    script: str = typer.Option("", "--script", "-s", help="Video script/prompt (defaults to topic)"),
    duration: int = typer.Option(10, "--duration", "-d", help="Video duration in seconds"),
    aspect: str = typer.Option("9:16", "--aspect", "-a", help="Aspect ratio (9:16, 16:9, 1:1)"),
    output: str = typer.Option("", "--output", "-o", help="Output filename"),
    backend: str = typer.Option("local", "--backend", "-b", help="Generation backend: local or higgsfield"),
    model: str = typer.Option("", "--model", "-m", help="Model to use (local: animatediff-sd15, modelscope-t2v, etc.; higgsfield: model name)"),
    fps: int = typer.Option(8, "--fps", help="Frames per second (local only)"),
    guidance: float = typer.Option(7.5, "--guidance", help="Guidance scale (local only)"),
    steps: int = typer.Option(25, "--steps", help="Inference steps (local only)"),
    seed: int = typer.Option(None, "--seed", help="Random seed for reproducibility"),
    init_image: str = typer.Option(None, "--init-image", help="Initial image for img2vid models (SVD)"),
    motion_scale: float = typer.Option(1.0, "--motion", help="Motion scale for AnimateDiff"),
):
    """Generate a video using local models or Higgsfield AI."""
    print_banner()
    
    async def _generate():
        config = load_config()
        script_text = script or topic
        output_path = Path(output) if output else Path(f"./output/videos/{topic[:30].replace(' ', '_')}_{duration}s.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if backend == "higgsfield":
            if not config.higgsfield.api_key:
                print_error("Higgsfield API key not configured. Run 'insta-ai init' first.")
                raise typer.Exit(1)
            
            request = HiggsfieldRequest(
                prompt=script_text,
                duration_seconds=duration,
                aspect_ratio=aspect,
                model=model or config.higgsfield.model
            )
            
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                task = progress.add_task("Generating video via Higgsfield...", total=None)
                async with HiggsfieldClient(config.higgsfield) as client:
                    response = await client.generate_and_download(request, output_path)
                progress.update(task, completed=True)
            
            if response.status.value == "completed":
                print_success(f"Video generated: {output_path}")
                console.print(f"  Video ID: {response.video_id}")
            else:
                print_error(f"Generation failed: {response.error}")
                raise typer.Exit(1)
        
        else:  # local backend
            local_model = model or "animatediff-sd15"
            
            # Validate model
            available = list_available_models()
            if local_model not in available:
                print_error(f"Unknown local model: {local_model}")
                console.print("Available models:")
                for k, v in available.items():
                    console.print(f"  {k}: {v['name']} (~{v['vram_gb']}GB VRAM)")
                raise typer.Exit(1)
            
            model_info = available[local_model]
            console.print(f"[cyan]Using {model_info['name']} (~{model_info['vram_gb']}GB VRAM)[/cyan]")
            
            # Check if SVD model needs init image
            if model_info.get("requires_image") and not init_image:
                print_error(f"Model {local_model} requires --init-image parameter")
                raise typer.Exit(1)
            
            request_data = {
                "prompt": script_text,
                "duration_seconds": duration,
                "aspect_ratio": aspect,
                "fps": fps,
                "guidance_scale": guidance,
                "num_inference_steps": steps,
                "seed": seed,
                "motion_scale": motion_scale,
            }
            if init_image:
                request_data["init_image"] = init_image
            
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                task = progress.add_task(f"Generating video locally ({local_model})...", total=None)
                client = LocalVideoClient(model_preset=local_model)
                try:
                    from .local_video_client import VideoGenerationRequest as LocalRequest
                    local_request = LocalRequest(**request_data)
                    response = await client.generate_and_download(local_request, output_path)
                finally:
                    await client.close()
                progress.update(task, completed=True)
            
            if response.status.value == "completed":
                print_success(f"Video generated: {output_path}")
                console.print(f"  Video ID: {response.video_id}")
                if response.metadata:
                    console.print(f"  Frames: {response.metadata.get('frames')}")
                    console.print(f"  FPS: {response.metadata.get('fps')}")
                    console.print(f"  Resolution: {response.metadata.get('resolution')}")
                    console.print(f"  Duration: {response.metadata.get('duration_seconds'):.1f}s")
            else:
                print_error(f"Generation failed: {response.error}")
                raise typer.Exit(1)
    
    asyncio.run(_generate())


@app.command()
def post(
    video: str = typer.Argument(..., help="Path to video file"),
    caption: str = typer.Option("", "--caption", "-c", help="Post caption"),
    manage: bool = typer.Option(True, "--manage/--no-manage", help="Manage interactions after posting"),
):
    """Post an existing video to Instagram Reels."""
    print_banner()
    
    video_path = Path(video)
    if not video_path.exists():
        print_error(f"Video not found: {video_path}")
        raise typer.Exit(1)
    
    async def _post():
        config = load_config()
        
        if not config.instagram.access_token:
            print_error("Instagram credentials not configured. Run 'insta-ai init' first.")
            raise typer.Exit(1)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Posting to Instagram...", total=None)
            
            pipeline = VideoPipeline(config)
            result = await pipeline.post_existing_video(video_path, caption, manage)
            
            progress.update(task, completed=True)
        
        if result.success:
            print_success("Video posted successfully!")
            console.print(f"  Media ID: {result.media_id}")
            console.print(f"  URL: {result.instagram_url}")
            if result.interaction_results:
                console.print(f"  Interactions: {result.interaction_results}")
        else:
            print_error(f"Posting failed: {result.error}")
            raise typer.Exit(1)
    
    asyncio.run(_post())


@app.command()
def models():
    """List available local video generation models."""
    print_banner()
    
    available = list_available_models()
    
    table = Table(title="Available Local Video Models (Optimized for 6GB VRAM)")
    table.add_column("Key", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Type", style="yellow")
    table.add_column("Resolution", justify="center")
    table.add_column("Default Frames", justify="center")
    table.add_column("Default FPS", justify="center")
    table.add_column("Est. VRAM", justify="right")
    table.add_column("Notes")
    
    for key, info in available.items():
        notes = ""
        if info.get("requires_image"):
            notes = "Requires --init-image (img2vid)"
        table.add_row(
            key,
            info["name"],
            info["type"],
            f"{info['resolution'][0]}x{info['resolution'][1]}",
            str(info["default_frames"]),
            str(info["default_fps"]),
            f"{info['vram_gb']} GB",
            notes
        )
    
    console.print(table)
    console.print("\n[dim]Usage: insta-ai generate 'topic' --backend local --model animatediff-sd15[/dim]")
    console.print("[dim]For SVD models, add --init-image path/to/image.jpg[/dim]")


@app.command()
def pipeline(
    topic: str = typer.Argument(..., help="Video topic"),
    script: str = typer.Option("", "--script", "-s", help="Video script/prompt"),
    duration: int = typer.Option(10, "--duration", "-d", help="Video duration in seconds"),
    aspect: str = typer.Option("9:16", "--aspect", "-a", help="Aspect ratio"),
    auto_post: bool = typer.Option(False, "--auto-post", help="Skip approval and post automatically"),
    no_interactions: bool = typer.Option(False, "--no-interactions", help="Skip interaction management"),
    backend: str = typer.Option("local", "--backend", "-b", help="Video backend: local or higgsfield"),
    local_model: str = typer.Option("animatediff-sd15", "--local-model", help="Local model preset"),
    fps: int = typer.Option(8, "--fps", help="FPS for local generation"),
    guidance: float = typer.Option(7.5, "--guidance", help="Guidance scale for local"),
    steps: int = typer.Option(25, "--steps", help="Inference steps for local"),
):
    """Run the full pipeline: Generate → Approve → Post → Engage."""
    print_banner()
    
    async def _pipeline():
        config = load_config()
        
        if backend == "higgsfield" and not config.higgsfield.api_key:
            print_error("Higgsfield API key not configured. Run 'insta-ai init' first.")
            raise typer.Exit(1)
        
        if not config.instagram.access_token and auto_post:
            print_error("Instagram credentials not configured for auto-post. Run 'insta-ai init' first.")
            raise typer.Exit(1)
        
        script_text = script or topic
        
        def progress_callback(state):
            stage_colors = {
                "generating_video": "yellow",
                "video_ready": "green",
                "awaiting_approval": "cyan",
                "uploading": "yellow",
                "processing": "yellow",
                "publishing": "yellow",
                "published": "green",
                "managing_interactions": "blue",
                "completed": "green",
                "failed": "red",
            }
            color = stage_colors.get(state.stage.value, "white")
            console.print(f"  [{color}]{state.stage.value}[/{color}] {state.error or ''}")
        
        backend_enum = VideoBackend.HIGGSFIELD if backend == "higgsfield" else VideoBackend.LOCAL
        
        pipeline = VideoPipeline(
            config, 
            progress_callback=progress_callback,
            video_backend=backend_enum,
            local_model_preset=local_model,
        )
        
        console.print(f"\n[bold]Starting pipeline for:[/bold] {topic}")
        console.print(f"Backend: {backend} ({local_model if backend == 'local' else 'Higgsfield'})")
        console.print(f"Script: {script_text[:100]}{'...' if len(script_text) > 100 else ''}\n")
        
        result = await pipeline.run(
            topic=topic,
            script=script_text,
            duration_seconds=duration,
            aspect_ratio=aspect,
            auto_post=auto_post,
            manage_interactions=not no_interactions,
            fps=fps,
            guidance_scale=guidance,
            num_inference_steps=steps,
        )
        
        console.print("\n" + "="*50)
        if result.success:
            print_success("Pipeline completed successfully!")
            console.print(f"  Video: {result.video_path}")
            console.print(f"  Instagram: {result.instagram_url}")
            console.print(f"  Media ID: {result.media_id}")
        else:
            print_error(f"Pipeline failed: {result.error}")
            raise typer.Exit(1)
    
    asyncio.run(_pipeline())


@app.command()
def interactions(
    media_id: str = typer.Argument(..., help="Instagram Media ID"),
    reply: bool = typer.Option(True, "--reply/--no-reply", help="Auto-reply to comments"),
    engage: bool = typer.Option(True, "--engage/--no-engage", help="Like matching comments"),
    analytics: bool = typer.Option(True, "--analytics/--no-analytics", help="Fetch analytics"),
):
    """Manage interactions for a posted media."""
    print_banner()
    
    async def _interactions():
        config = load_config()
        
        if not config.instagram.access_token:
            print_error("Instagram credentials not configured. Run 'insta-ai init' first.")
            raise typer.Exit(1)
        
        async with InstagramClient(config.instagram) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Managing interactions...", total=None)
                
                results = await manage_post_interactions(
                    media_id, client,
                    auto_reply=reply,
                    engage_comments=engage,
                    fetch_analytics=analytics
                )
                
                progress.update(task, completed=True)
        
        print_success("Interaction management completed!")
        
        if results.get("replies"):
            console.print(f"  Replies sent: {len(results['replies'])}")
        if results.get("engagements"):
            console.print(f"  Comments liked: {len(results['engagements'])}")
        if results.get("analytics"):
            a = results["analytics"]
            console.print(f"  Views: {a['views']}, Likes: {a['likes']}, Comments: {a['comments']}")
            console.print(f"  Engagement Rate: {a['engagement_rate']}%")
    
    asyncio.run(_interactions())


@app.command()
def analytics(
    days: int = typer.Option(30, "--days", "-d", help="Days of analytics to show"),
    export: str = typer.Option("", "--export", "-e", help="Export report to JSON file"),
):
    """View analytics dashboard."""
    print_banner()
    
    config = load_config()
    
    if not config.instagram.access_token:
        print_error("Instagram credentials not configured. Run 'insta-ai init' first.")
        raise typer.Exit(1)
    
    async def _analytics():
        async with InstagramClient(config.instagram) as client:
            manager = InteractionManager(client, config.interactions)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Fetching analytics...", total=None)
                
                # Fetch recent media analytics
                await manager.fetch_all_recent_analytics(limit=25)
                
                # Get stats
                stats = manager.get_interaction_stats(days)
                
                progress.update(task, completed=True)
            
            # Display analytics
            media_analytics = manager.db.get_all_media_analytics(days)
            
            if media_analytics:
                table = Table(title=f"Media Analytics (Last {days} Days)")
                table.add_column("Media ID", style="cyan")
                table.add_column("Views", justify="right")
                table.add_column("Likes", justify="right")
                table.add_column("Comments", justify="right")
                table.add_column("Shares", justify="right")
                table.add_column("Engagement %", justify="right")
                table.add_column("Fetched", style="dim")
                
                for a in media_analytics:
                    table.add_row(
                        a["media_id"][:12] + "...",
                        str(a["views"]),
                        str(a["likes"]),
                        str(a["comments"]),
                        str(a["shares"]),
                        f"{a['engagement_rate']:.1f}%",
                        a["fetched_at"][:16]
                    )
                
                console.print(table)
                
                # Summary
                total_views = sum(a["views"] for a in media_analytics)
                total_likes = sum(a["likes"] for a in media_analytics)
                total_comments = sum(a["comments"] for a in media_analytics)
                avg_engagement = sum(a["engagement_rate"] for a in media_analytics) / len(media_analytics)
                
                console.print(Panel(
                    f"[bold]Total Media:[/bold] {len(media_analytics)}\n"
                    f"[bold]Total Views:[/bold] {total_views:,}\n"
                    f"[bold]Total Likes:[/bold] {total_likes:,}\n"
                    f"[bold]Total Comments:[/bold] {total_comments:,}\n"
                    f"[bold]Avg Engagement:[/bold] {avg_engagement:.1f}%",
                    title="Summary",
                    border_style="green"
                ))
                
                if export:
                    manager.export_analytics_report(Path(export), days)
                    print_success(f"Report exported to {export}")
            else:
                print_info("No analytics data found. Post some content first!")
    
    asyncio.run(_analytics())


@app.command()
def validate():
    """Validate API credentials and connections."""
    print_banner()
    
    async def _validate():
        config = load_config()
        all_ok = True
        
        # Validate Higgsfield
        console.print("\n[cyan]Validating Higgsfield API...[/cyan]")
        if config.higgsfield.api_key:
            try:
                async with HiggsfieldClient(config.higgsfield) as client:
                    print_success("Higgsfield API key configured")
            except Exception as e:
                print_error(f"Higgsfield validation failed: {e}")
                all_ok = False
        else:
            print_error("Higgsfield API key not configured")
            all_ok = False
        
        # Validate local video dependencies
        console.print("\n[cyan]Checking local video dependencies...[/cyan]")
        try:
            import torch
            import diffusers
            import transformers
            import accelerate
            
            print_success(f"PyTorch {torch.__version__}")
            print_success(f"Diffusers {diffusers.__version__}")
            print_success(f"Transformers {transformers.__version__}")
            print_success(f"Accelerate {accelerate.__version__}")
            
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                print_success(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
                
                if vram_gb < 5:
                    console.print("[yellow]Warning: Less than 5GB VRAM may limit model options[/yellow]")
            else:
                console.print("[yellow]No CUDA GPU detected - local generation will be very slow on CPU[/yellow]")
                
        except ImportError as e:
            print_error(f"Missing dependency: {e}")
            console.print("Run: pip install -r requirements.txt")
            all_ok = False
        
        # Validate Instagram
        console.print("\n[cyan]Validating Instagram API...[/cyan]")
        if config.instagram.access_token:
            try:
                async with InstagramClient(config.instagram) as client:
                    valid = await client.validate_token()
                    if valid:
                        account = await client.get_account_info()
                        print_success(f"Instagram connected: @{account.get('username', 'unknown')}")
                        console.print(f"  Followers: {account.get('followers_count', 'N/A')}")
                        console.print(f"  Media Count: {account.get('media_count', 'N/A')}")
                    else:
                        print_error("Instagram access token invalid or expired")
                        all_ok = False
            except Exception as e:
                print_error(f"Instagram validation failed: {e}")
                all_ok = False
        else:
            print_error("Instagram credentials not configured")
            all_ok = False
        
        if all_ok:
            console.print("\n[green]All validations passed![/green]")
        else:
            console.print("\n[red]Some validations failed. Check configuration.[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_validate())


@app.command()
def config_show():
    """Show current configuration (with masked secrets)."""
    config = load_config()
    
    console.print(Panel("Current Configuration", border_style="cyan"))
    
    def mask(value: str) -> str:
        if not value:
            return "[dim]not set[/dim]"
        if len(value) <= 8:
            return "*" * len(value)
        return value[:4] + "*" * (len(value) - 8) + value[-4:]
    
    table = Table()
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    
    table.add_row("Higgsfield API Key", mask(config.higgsfield.api_key))
    table.add_row("Higgsfield Base URL", config.higgsfield.base_url)
    table.add_row("Higgsfield Model", config.higgsfield.model)
    table.add_row("Instagram App ID", mask(config.instagram.app_id))
    table.add_row("Instagram App Secret", mask(config.instagram.app_secret))
    table.add_row("Instagram Access Token", mask(config.instagram.access_token))
    table.add_row("Business Account ID", mask(config.instagram.business_account_id))
    table.add_row("Refresh Token", mask(config.instagram.refresh_token))
    table.add_row("Video Output Dir", config.pipeline.video_output_dir)
    table.add_row("Max Duration", f"{config.pipeline.max_video_duration_seconds}s")
    table.add_row("Default Aspect", config.pipeline.default_aspect_ratio)
    
    console.print(table)


if __name__ == "__main__":
    app()
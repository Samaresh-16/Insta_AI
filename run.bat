@echo off
REM AI Video Pipeline - Quick Start Script
REM Run this after installing dependencies

echo ========================================
echo AI Video Pipeline for Instagram
echo ========================================
echo.

echo Checking Python...
python --version
if errorlevel 1 (
    echo Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

echo.
echo Checking dependencies...
pip list | findstr "pydantic httpx typer rich"
if errorlevel 1 (
    echo Some dependencies missing. Installing...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo Checking configuration...
if not exist config.local.yaml (
    echo No config.local.yaml found.
    echo Run 'insta-ai init' to create one, or copy config.yaml to config.local.yaml and edit.
    echo.
)

echo.
echo Available commands:
echo   insta-ai init              - Setup configuration
echo   insta-ai validate          - Check API credentials
echo   insta-ai pipeline "topic"  - Full pipeline (generate, approve, post)
echo   insta-ai generate "topic"  - Generate video only
echo   insta-ai post video.mp4    - Post existing video
echo   insta-ai interactions ID   - Manage post interactions
echo   insta-ai analytics         - View analytics dashboard
echo   insta-ai config-show       - Show current config
echo.

echo Run 'insta-ai --help' for more options
pause
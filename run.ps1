<#
.SYNOPSIS
    AI Video Pipeline for Instagram - Quick Start Script

.DESCRIPTION
    Sets up and runs the AI video generation pipeline with Higgsfield and Instagram integration.
#>

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AI Video Pipeline for Instagram" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found! Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Check/install dependencies
Write-Host "Checking dependencies..." -ForegroundColor Yellow
$requiredPackages = @("pydantic", "httpx", "typer", "rich", "pydantic-settings", "pyyaml", "tenacity")
$missingPackages = @()

foreach ($pkg in $requiredPackages) {
    $result = pip list --disable-pip-version-check 2>$null | Select-String $pkg
    if (-not $result) {
        $missingPackages += $pkg
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host "Missing packages: $($missingPackages -join ', ')" -ForegroundColor Yellow
    Write-Host "Installing..." -ForegroundColor Yellow
    pip install -r requirements.txt --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "All dependencies installed" -ForegroundColor Green
}

Write-Host ""

# Check configuration
if (-not (Test-Path "config.local.yaml")) {
    Write-Host "No config.local.yaml found." -ForegroundColor Yellow
    Write-Host "Run 'insta-ai init' to create one, or copy config.yaml to config.local.yaml and edit." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "Available commands:" -ForegroundColor Cyan
Write-Host "  insta-ai init              - Setup configuration" -ForegroundColor White
Write-Host "  insta-ai validate          - Check API credentials" -ForegroundColor White
Write-Host "  insta-ai pipeline 'topic'  - Full pipeline (generate, approve, post)" -ForegroundColor White
Write-Host "  insta-ai generate 'topic'  - Generate video only" -ForegroundColor White
Write-Host "  insta-ai post video.mp4    - Post existing video" -ForegroundColor White
Write-Host "  insta-ai interactions ID   - Manage post interactions" -ForegroundColor White
Write-Host "  insta-ai analytics         - View analytics dashboard" -ForegroundColor White
Write-Host "  insta-ai config-show       - Show current config" -ForegroundColor White
Write-Host ""

Write-Host "Run 'insta-ai --help' for more options" -ForegroundColor Gray
Write-Host ""

Read-Host "Press Enter to continue..."
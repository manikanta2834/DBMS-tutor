# DBMS Tutor AI - Native PowerShell Web Launcher
[CmdletBinding()]
param (
    [int]$Port = 7860
)

$host.UI.RawUI.WindowTitle = "DBMS Tutor AI - Web Chatbot"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "         DBMS Tutor AI - Web Application" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to script directory
Set-Location $PSScriptRoot

# 1. Check Python installation
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python was not found in your system PATH!" -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from python.org and ensure 'Add Python to PATH' is checked." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit..."
    exit 1
}

# 2. Check and activate virtual environment
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "[INFO] Activating virtual environment (venv)..." -ForegroundColor Green
    & ".\venv\Scripts\Activate.ps1"
} elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "[INFO] Activating virtual environment (.venv)..." -ForegroundColor Green
    & ".\.venv\Scripts\Activate.ps1"
} else {
    Write-Host "[INFO] No local virtual environment found. Using system Python." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[INFO] Starting DBMS Tutor Web Server on port $Port..." -ForegroundColor Cyan
Write-Host "[INFO] URL: http://localhost:$Port" -ForegroundColor Green
Write-Host "[INFO] Press Ctrl+C in this window to stop the server." -ForegroundColor Gray
Write-Host ""

# 3. Automatically launch default browser after 2 seconds in a background job
Start-Job -ScriptBlock {
    param($p)
    Start-Sleep -Seconds 2
    Start-Process "http://localhost:$p"
} -ArgumentList $Port | Out-Null

# 4. Run the web server
python web_chat.py --port $Port

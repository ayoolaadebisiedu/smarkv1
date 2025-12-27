# Smark Backend Startup Script

Write-Host "Checking for virtual environment..." -ForegroundColor Cyan
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment with Python 3.12..." -ForegroundColor Yellow
    py -3.12 -m venv venv
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
# Activate based on available script
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    . .\venv\Scripts\Activate.ps1
}

Write-Host "Installing dependencies..." -ForegroundColor Cyan
pip install -r backend/requirements.txt

Write-Host "Starting Smark Backend..." -ForegroundColor Green
Write-Host "Access API at: http://localhost:8000" -ForegroundColor Gray
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray

# Run the server
uvicorn backend.main:app --reload

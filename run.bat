@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo First run: setting up Local Transcriber. This downloads about 1 GB and takes a few minutes...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
    if errorlevel 1 (
        echo.
        echo Setup did not finish. Fix the problem above, then run this again.
        pause
        exit /b 1
    )
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0app.py"

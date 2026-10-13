<#
  One-time setup for Local Transcriber.
  Creates .venv next to this script and installs the pinned dependencies.
  Download caches stay inside this folder, because C: is often short on space.

  Usage:  powershell -ExecutionPolicy Bypass -File setup.ps1
#>
$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
Set-Location $root
$env:UV_CACHE_DIR = Join-Path $root ".cache\uv"
$env:PIP_CACHE_DIR = Join-Path $root ".cache\pip"
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$check = "import sys, struct; ok = (3, 10) <= sys.version_info[:2] <= (3, 13) and struct.calcsize('P') == 8; print(sys.executable if ok else '')"

function Find-Python {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($version in "3.12", "3.11", "3.13", "3.10") {
            $exe = & py "-$version" -c $check 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
        }
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        $exe = & $python.Source -c $check 2>$null
        if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
    }
    return $null
}

Write-Host "== Local Transcriber setup ==" -ForegroundColor Cyan
$uv = Get-Command uv -ErrorAction SilentlyContinue

if (-not (Test-Path $venvPython)) {
    $python = Find-Python
    if (-not $python) {
        Write-Host "64-bit Python 3.10-3.13 was not found." -ForegroundColor Red
        Write-Host "Install it with:  winget install Python.Python.3.12   and run this again."
        exit 1
    }
    Write-Host "Creating .venv with $python"
    if ($uv) { & $uv.Source venv --python $python .venv } else { & $python -m venv .venv }
    if ($LASTEXITCODE -ne 0) { Write-Host "Creating the virtual environment failed." -ForegroundColor Red; exit 1 }
}

Write-Host "Installing packages (first time: about 1 GB download, a few minutes)..."
if ($uv) {
    & $uv.Source pip install --python $venvPython -r requirements.txt
} else {
    & $venvPython -m ensurepip --upgrade | Out-Null
    & $venvPython -m pip install -r requirements.txt
}
if ($LASTEXITCODE -ne 0) { Write-Host "Package installation failed (see the messages above)." -ForegroundColor Red; exit 1 }

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "ffmpeg is not installed. It is needed to read audio and video files." -ForegroundColor Yellow
    Write-Host "Install it with:  winget install Gyan.FFmpeg   then open a new terminal."
}

Write-Host ""
Write-Host "Setup complete. Start the app by double-clicking run.bat" -ForegroundColor Green
exit 0

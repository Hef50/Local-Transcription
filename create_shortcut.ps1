<#
  Creates "Local Transcriber.lnk" in this folder. Windows can pin a shortcut to the
  taskbar or Start (it can't pin run.bat). Re-run this if you move the folder.

  Usage:  powershell -ExecutionPolicy Bypass -File create_shortcut.ps1
#>
$root = $PSScriptRoot
$pythonw = Join-Path $root ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pythonw)) {
    Write-Host "Run setup.ps1 (or run.bat) first so .venv exists." -ForegroundColor Red
    exit 1
}
$path = Join-Path $root "Local Transcriber.lnk"
$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($path)
$shortcut.TargetPath = $pythonw
$shortcut.Arguments = '"' + (Join-Path $root "app.py") + '"'
$shortcut.WorkingDirectory = $root
$shortcut.IconLocation = (Join-Path $root "assets\app.ico") + ",0"
$shortcut.Description = "Local Transcriber: offline transcripts with speaker labels"
$shortcut.Save()
Write-Host "Created $path"
Write-Host "Right-click it > Show more options > Pin to taskbar."

<#
.SYNOPSIS
    Installs ASPIRE AI OS as a Windows startup background service.
.DESCRIPTION
    Registers ASPIRE to start automatically with Windows, running
    minimized to the system tray. Also creates a scheduled task
    for reliable startup.
#>

param(
    [string]$ProjectPath = (Resolve-Path "$PSScriptRoot\..").Path,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

$backendDir = Join-Path $ProjectPath "backend"
$frontendDir = Join-Path $ProjectPath "frontend"
$venvDir = Join-Path $backendDir ".venv"
$logDir = Join-Path $ProjectPath "logs"
$taskName = "ASPIRE-AI-OS"
$shortcutName = "ASPIRE AI OS.lnk"

# Ensure logs directory exists
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

if ($Uninstall) {
    Write-Host "Uninstalling ASPIRE startup..." -ForegroundColor Yellow

    # Remove scheduled task
    schtasks /Delete /TN $taskName /F 2>$null
    Write-Host "  Removed scheduled task." -ForegroundColor Green

    # Remove startup shortcut
    $startupPath = [Environment]::GetFolderPath("Startup")
    $shortcut = Join-Path $startupPath $shortcutName
    if (Test-Path $shortcut) {
        Remove-Item $shortcut -Force
        Write-Host "  Removed startup shortcut." -ForegroundColor Green
    }

    # Kill running processes
    Get-Process -Name "python", "node", "uvicorn" -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowTitle -eq "" } |
        Stop-Process -Force -ErrorAction SilentlyContinue

    Write-Host "ASPIRE uninstalled successfully." -ForegroundColor Green
    return
}

Write-Host "Installing ASPIRE AI OS startup..." -ForegroundColor Cyan

# 1. Create startup script
$startScript = @"
@echo off
title ASPIRE AI OS
echo Starting ASPIRE AI OS...

:: Start Backend
cd /d "$backendDir"
start /B /MIN "" python -m uvicorn app.integration.app:app --host 0.0.0.0 --port 8000 > "$logDir\backend.log" 2>&1

:: Wait for backend to start
timeout /t 5 /nobreak >nul

:: Start Frontend
cd /d "$frontendDir"
start /B /MIN "" npx vite --host > "$logDir\frontend.log" 2>&1

:: Open browser
timeout /t 3 /nobreak >nul
start "" http://localhost:5173

echo ASPIRE AI OS is running.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
echo Logs: $logDir
"@

$batPath = Join-Path $ProjectPath "scripts\start_aspire.bat"
Set-Content -Path $batPath -Value $startScript -Encoding ASCII
Write-Host "  Created startup script: $batPath" -ForegroundColor Green

# 2. Add to Windows Startup folder
try {
    $startupPath = [Environment]::GetFolderPath("Startup")
    $shortcutPath = Join-Path $startupPath $shortcutName

    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $batPath
    $shortcut.WorkingDirectory = $ProjectPath
    $shortcut.Description = "ASPIRE AI OS Background Service"
    $shortcut.WindowStyle = 7  # Minimized
    $shortcut.Save()

    Write-Host "  Added to Windows Startup: $shortcutPath" -ForegroundColor Green
} catch {
    Write-Warning "  Could not add to startup folder: $_"
}

# 3. Create scheduled task for reliability
$psScript = @"
`$ProjectPath = "$ProjectPath"
`$backendDir = "$backendDir"
`$frontendDir = "$frontendDir"
`$logDir = "$logDir"
`$batPath = "$batPath"

if (-not (Test-Path `$batPath)) {
    Write-EventLog -LogName Application -Source "ASPIRE" -EntryType Error -EventId 1000 -Message "ASPIRE startup script not found at `$batPath"
    exit 1
}

Start-Process -FilePath `$batPath -WindowStyle Hidden -WorkingDirectory `$ProjectPath
"@

$psScriptPath = Join-Path $ProjectPath "scripts\start_aspire.ps1"
Set-Content -Path $psScriptPath -Value $psScript -Encoding UTF8

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-WindowStyle Hidden -ExecutionPolicy Bypass -File `"$psScriptPath`""
$trigger = New-ScheduledTaskTrigger -AtStartup -RandomDelay (New-TimeSpan -Seconds 30)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

try {
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest -Force
    Write-Host "  Created scheduled task: $taskName (runs at startup)" -ForegroundColor Green
} catch {
    Write-Warning "  Could not create scheduled task: $_"
    Write-Warning "  Run as Administrator to create scheduled task."
}

Write-Host ""
Write-Host "ASPIRE AI OS installed successfully!" -ForegroundColor Green
Write-Host "  - Starts automatically with Windows" -ForegroundColor Cyan
Write-Host "  - Runs minimized to system tray" -ForegroundColor Cyan
Write-Host "  - Logs: $logDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "To uninstall, run:" -ForegroundColor Yellow
Write-Host "  powershell -File scripts\install_startup.ps1 -Uninstall" -ForegroundColor Yellow

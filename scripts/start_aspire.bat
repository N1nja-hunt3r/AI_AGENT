@echo off
title ASPIRE AI OS
echo ============================================
echo    ASPIRE AI OS - Starting...
echo ============================================

:: Get the directory where this script is located
set SCRIPTS_DIR=%~dp0
set PROJECT_DIR=%SCRIPTS_DIR%..
set BACKEND_DIR=%PROJECT_DIR%\backend
set FRONTEND_DIR=%PROJECT_DIR%\frontend
set LOG_DIR=%PROJECT_DIR%\logs

:: Ensure logs directory exists
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo Starting backend server...
cd /d "%BACKEND_DIR%"
start /B /MIN "" python -m uvicorn app.integration.app:app --host 0.0.0.0 --port 8000 > "%LOG_DIR%\backend.log" 2>&1
echo Backend starting on http://localhost:8000

:: Wait for backend
timeout /t 5 /nobreak >nul

echo Starting frontend dev server...
cd /d "%FRONTEND_DIR%"
start /B /MIN "" npx vite --host --port 5173 > "%LOG_DIR%\frontend.log" 2>&1
echo Frontend starting on http://localhost:5173

:: Wait a moment then open browser
timeout /t 3 /nobreak >nul
start "" http://localhost:5173

echo.
echo ============================================
echo    ASPIRE AI OS is now running!
echo    Backend:  http://localhost:8000
echo    Frontend: http://localhost:5173
echo    Logs:     %LOG_DIR%
echo ============================================
echo.
echo Say "Hey ASPIRE" to activate the assistant.

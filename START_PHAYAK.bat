@echo off
title PHAYAK - Thai Product Taxonomy Engine
echo ====================================================
echo   PHAYAK Engine is starting (LAN Mode)...
echo ====================================================

:: --- Get REAL LAN IP (from default gateway interface, not virtual adapters) ---
for /f "tokens=4" %%a in ('route print 0.0.0.0 ^| findstr " 0.0.0.0 "') do set LAN_IP=%%a

:: --- Open Firewall ports for LAN access (requires admin) ---
echo [0/3] Opening Firewall ports for LAN access...
netsh advfirewall firewall add rule name="Phayak-Backend-8000" protocol=TCP dir=in localport=8000 action=allow >nul 2>&1
netsh advfirewall firewall add rule name="Phayak-Frontend-3000" protocol=TCP dir=in localport=3000 action=allow >nul 2>&1
netsh advfirewall firewall add rule name="Phayak-Supabase-54331" protocol=TCP dir=in localport=54331 action=allow >nul 2>&1

:: 1. Start Backend (FastAPI) in a new window - binds 0.0.0.0
echo [1/3] Starting AI Backend (FastAPI on Port 8000)...
start "Phayak Backend" cmd /c "cd /d %~dp0 && .\.venv\Scripts\python -m src.api.api_server"

:: 2. Start Frontend (Next.js) in a new window - binds 0.0.0.0 for LAN access
echo [2/3] Starting UI Dashboard (Next.js on Port 3000 - LAN enabled)...
start "Phayak Frontend" cmd /c "cd /d %~dp0\taxonomy-app && npx next dev --hostname 0.0.0.0"

:: 3. Wait a few seconds for services to warm up
echo [3/3] Waiting for services to stabilize...
timeout /t 6 /nobreak > nul

:: 4. Open Browser (local)
echo.
echo ====================================================
echo   All services are running!
echo ====================================================
echo.
echo   [This Machine]
echo   UI Dashboard : http://localhost:3000
echo   AI Backend   : http://127.0.0.1:8000/docs
echo.
echo   [LAN Access - other devices]
echo   UI Dashboard : http://%LAN_IP%:3000
echo   AI Backend   : http://%LAN_IP%:8000/docs
echo.
echo   Keep the other windows open to maintain the service.
echo ====================================================
echo.
start http://localhost:3000
pause

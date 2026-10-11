@echo off
title Cloudflare Tunnel - smilesboo.com
echo ====================================================
echo   Starting Cloudflare Tunnel for PHAYAK
echo   Subdomains:
echo     - https://ckeck.smilesboo.com
echo     - https://check.smilesboo.com
echo   Forwarding to: http://localhost:3000
echo ====================================================
echo.

set "PATH=%LOCALAPPDATA%\Programs\cloudflared;%PATH%"

where cloudflared >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] cloudflared.exe not found!
    echo Looked in: %LOCALAPPDATA%\Programs\cloudflared
    pause
    exit /b 1
)

echo Running tunnel: product-checker ...
echo Press Ctrl+C to stop the tunnel.
echo.

cloudflared.exe tunnel run product-checker
pause

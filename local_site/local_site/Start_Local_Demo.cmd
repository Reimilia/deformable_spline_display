@echo off
cd /d "%~dp0"
start "Darboux Shape Lab" /min ".runtime\node.exe" "server.mjs"
timeout /t 1 /nobreak >nul
start "" "http://localhost:8787"

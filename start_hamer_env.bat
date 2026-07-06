@echo off
echo ==========================================
echo   Activating hamer_env in PowerShell...
echo ==========================================
powershell -NoExit -ExecutionPolicy Bypass -Command "& '%~dp0hamer_env\Scripts\Activate.ps1'"

@echo off
echo ==========================================
echo   Activating asl_env in PowerShell...
echo ==========================================
powershell -NoExit -ExecutionPolicy Bypass -Command "& '%~dp0asl_env\Scripts\Activate.ps1'"

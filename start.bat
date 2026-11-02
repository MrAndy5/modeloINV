@echo off
title modeloINV Control Center
cd /d "%~dp0"

echo ========================================================================
echo                 Checking modeloINV Python Environment
echo ========================================================================

python -c "import yaml, numpy, pandas, pyarrow, sklearn, torch, fastapi, uvicorn" 2>nul
if %errorlevel% neq 0 (
    echo [!] Missing or incompatible dependencies detected.
    echo [!] Upgrading pip to ensure pre-compiled binary wheels are preferred...
    python -m pip install --upgrade pip
    echo [!] Installing dependencies with binary wheels (avoiding C++ compiler)...
    python -m pip install -r requirements.txt --prefer-binary
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Failed to install dependencies automatically.
        echo Try running: python -m pip install -r requirements.txt --prefer-binary
        pause
        exit /b 1
    )
    echo [OK] All dependencies successfully verified!
)

echo Starting modeloINV Interactive Control Center...
python interactive.py
pause

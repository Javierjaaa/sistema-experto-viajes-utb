@echo off
title Sistema Experto de Viajes
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Creando entorno virtual e instalando dependencias...
    python -m venv venv
    call venv\Scripts\python.exe -m pip install --quiet flask
)

echo.
echo  Sistema Experto de Viajes
echo  Servidor: http://127.0.0.1:5000
echo  Presiona Ctrl+C para detenerlo.
echo.

start "" /b cmd /c "timeout /t 2 >nul & start http://127.0.0.1:5000"
venv\Scripts\python.exe app.py
pause

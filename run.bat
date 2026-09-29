@echo off
setlocal

title DBMS Tutor AI - Web Chatbot
color 0b

echo =======================================================
echo          DBMS Tutor AI - Web Application
echo =======================================================
echo.

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    color 0c
    echo [ERROR] Python was not found in your system PATH.
    echo Please install Python 3.10+ and ensure it is added to PATH.
    echo.
    pause
    exit /b 1
)

if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment: venv
    call "venv\Scripts\activate.bat"
    goto :venv_ready
)

if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment: .venv
    call ".venv\Scripts\activate.bat"
    goto :venv_ready
)

echo [INFO] No local virtual environment found. Running with system Python.

:venv_ready
echo.
echo [INFO] Starting DBMS Tutor Web Server
echo [INFO] URL: http://localhost:7860
echo [INFO] Press Ctrl+C in this terminal to stop the server.
echo.

start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:7860"

python web_chat.py --port 7860

if errorlevel 1 (
    echo.
    color 0c
    echo [WARNING] Server stopped.
    pause
)

endlocal

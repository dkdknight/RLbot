@echo off
setlocal enabledelayedexpansion
echo ================================================
echo RLBot Bridge - Launcher
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.12 from https://www.python.org/
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not installed
    pause
    exit /b 1
)

echo Checking dependencies...
echo.

REM Check if requirements.txt exists
if not exist requirements.txt (
    echo WARNING: requirements.txt not found
    echo Creating requirements.txt...
    echo numpy>=1.24.0 > requirements.txt
    echo torch>=2.0.0 >> requirements.txt
    echo Creating requirements file...
)

REM Install/upgrade dependencies
echo Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Dependencies installed successfully!
echo.

REM Check if bots directory exists
if not exist bots (
    echo Creating 'bots' directory...
    mkdir bots
    echo Please place your bot models in the 'bots' directory
    echo Each bot should be in its own subdirectory with a .pt file
    echo.
)

REM Check Rocket League installation (optional)
echo Checking for Rocket League installation...
set RL_PATH=
if exist "C:\Program Files\Epic Games\rocketleague\Binaries\Win64\RocketLeague.exe" (
    set RL_PATH=C:\Program Files\Epic Games\rocketleague
    echo Rocket League found: !RL_PATH!
) else if exist "C:\Program Files (x86)\Steam\steamapps\common\rocketleague\Binaries\Win64\RocketLeague.exe" (
    set RL_PATH=C:\Program Files (x86)\Steam\steamapps\common\rocketleague
    echo Rocket League found: !RL_PATH!
) else (
    echo WARNING: Rocket League installation not found in default locations
    echo Make sure Rocket League and BakkesMod are running
)
echo.

echo ================================================
echo Starting RLBot Bridge...
echo ================================================
echo.
echo INSTRUCTIONS:
echo 1. Make sure Rocket League is running
echo 2. Make sure BakkesMod is loaded with the RLBot Bridge plugin
echo 3. Select a bot when prompted
echo 4. Enter a game (Freeplay recommended)
echo 5. Press F1 to enable bot control
echo.
echo Press Ctrl+C to stop the bot
echo ================================================
echo.

REM Run the Python script
python main.py

if errorlevel 1 (
    echo.
    echo ERROR: Bot crashed or exited with an error
    pause
    exit /b 1
)

echo.
echo Bot stopped successfully
pause

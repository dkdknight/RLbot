# Installation Guide

This guide will walk you through setting up the RLBot Bridge system from scratch.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [BakkesMod Plugin Setup](#bakkesmod-plugin-setup)
3. [Python Environment Setup](#python-environment-setup)
4. [Bot Model Setup](#bot-model-setup)
5. [First Run](#first-run)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

1. **Rocket League**
   - Epic Games or Steam version
   - Fully updated to latest version

2. **BakkesMod**
   - Download: https://www.bakkesmod.com/
   - Install and run at least once to set up directories

3. **Python 3.12 or newer**
   - Download: https://www.python.org/downloads/
   - Make sure to check "Add Python to PATH" during installation

4. **Visual Studio 2019 or newer** (for building C++ plugin)
   - Download: https://visualstudio.microsoft.com/
   - Install "Desktop development with C++" workload
   - Community edition is free and sufficient

5. **CMake 3.15 or newer**
   - Download: https://cmake.org/download/
   - Or install via Visual Studio installer

6. **BakkesMod SDK**
   - Download: https://github.com/bakkesmodorg/BakkesModSDK
   - Extract to a known location (e.g., `C:\BakkesModSDK`)

## BakkesMod Plugin Setup

### Step 1: Download BakkesMod SDK

1. Go to https://github.com/bakkesmodorg/BakkesModSDK
2. Download the latest release or clone the repository
3. Extract to `C:\BakkesModSDK` (or your preferred location)

### Step 2: Configure Build

1. Open a terminal in the `BakkesModPlugin` directory
2. Edit `CMakeLists.txt` and update the SDK path:
   ```cmake
   set(BAKKESMOD_SDK_PATH "C:/BakkesModSDK" CACHE PATH "Path to BakkesMod SDK")
   ```

### Step 3: Build the Plugin

```bash
# Create build directory
mkdir build
cd build

# Generate Visual Studio project
cmake .. -G "Visual Studio 16 2019" -A x64

# Build in Release mode
cmake --build . --config Release
```

The compiled plugin will be in `build/bin/Release/RLBotBridge.dll`

### Step 4: Install the Plugin

1. Copy `RLBotBridge.dll` to BakkesMod plugins folder:
   ```
   %APPDATA%\bakkesmod\bakkesmod\plugins\
   ```
   
2. Typical path: `C:\Users\YourUsername\AppData\Roaming\bakkesmod\bakkesmod\plugins\`

### Step 5: Verify Plugin

1. Start Rocket League with BakkesMod
2. Open BakkesMod console (F6)
3. Type: `plugin load rlbotbridge`
4. You should see: "RLBot Bridge plugin loaded"
5. Check for socket server message: "Socket server listening on port 5000"

## Python Environment Setup

### Step 1: Verify Python Installation

```bash
python --version
```

Should show Python 3.12 or newer.

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
# In the RLbot directory
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- numpy (numerical operations)
- torch (PyTorch for model loading and inference)

### Step 4: Verify PyTorch Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch 2.x.x
CUDA available: True/False
```

## Bot Model Setup

### Step 1: Create Bot Directory

The `bots/` directory should already exist. Create subdirectories for each bot:

```
bots/
├── Seer/
│   └── Seer.pt
├── Ripple/
│   └── Ripple.pt
└── MyCustomBot/
    └── model.pt
```

### Step 2: Add Your Bot Models

1. Copy your trained PyTorch model (.pt file) to the appropriate subdirectory
2. Ensure the model is compatible:
   - Input: 159 observations
   - Output: 8 actions (or compatible format)

### Step 3: Test Model Loading (Optional)

```python
import torch

# Try loading your model
model = torch.load('bots/Seer/Seer.pt')
print(f"Model type: {type(model)}")
print(f"Model: {model}")
```

## First Run

### Step 1: Start Everything

1. **Launch Rocket League** with BakkesMod
2. **Load the plugin** (if not auto-loaded):
   ```
   plugin load rlbotbridge
   ```
3. **Run the Python bot manager**:
   ```bash
   python main.py
   ```
   or double-click `start_bot.bat`

### Step 2: Select Your Bot

The script will display available bots:
```
=== Available Bots ===
1. Seer/Seer.pt
2. Ripple/Ripple.pt

Select a bot (enter number):
```

Enter the number of your desired bot.

### Step 3: Connect

The script will attempt to connect to the plugin:
```
Connecting to BakkesMod plugin at localhost:5000...
Connected successfully!

Bot is running! Waiting for observations...
Press F1 in Rocket League to enable bot control
Press Ctrl+C to stop
```

### Step 4: Enter a Game

1. Start a Freeplay match (easiest for testing)
2. Once spawned, press **F1** to enable bot control
3. The bot should now control your car!

### Step 5: Monitor

- Watch the console for any errors
- Check BakkesMod console for plugin logs
- Press F1 again to disable bot control

## Troubleshooting

### Plugin Won't Load

**Symptom**: "Plugin not found" or similar error

**Solutions**:
- Verify DLL is in correct folder: `%APPDATA%\bakkesmod\bakkesmod\plugins\`
- Check file name is exactly `RLBotBridge.dll`
- Ensure Visual C++ Redistributable is installed
- Try rebuilding in Release mode (not Debug)

### Connection Refused

**Symptom**: "Connection refused" or timeout

**Solutions**:
- Ensure BakkesMod plugin is loaded
- Check BakkesMod console for "Socket server listening on port 5000"
- Verify firewall isn't blocking port 5000
- Try running Python as administrator
- Check if another program is using port 5000:
  ```bash
  netstat -an | findstr :5000
  ```

### Model Loading Errors

**Symptom**: "Error loading model" or tensor shape mismatch

**Solutions**:
- Verify model file is valid PyTorch model
- Check PyTorch version compatibility
- Try loading model in standalone Python script
- Ensure model expects 159 inputs and produces 8 outputs
- Check if CUDA model is being loaded on CPU-only system

### Bot Doesn't Respond

**Symptom**: Bot connected but car doesn't move

**Solutions**:
- Press F1 to enable bot control
- Verify you're in a game (not menu)
- Check console for errors
- Ensure observations are being received
- Try toggling F1 off and on

### Poor Performance

**Symptom**: Lag, stuttering, or delayed actions

**Solutions**:
- Check if GPU is being utilized (if available)
- Close other programs to free resources
- Verify frame skip is set correctly (8 for Seer)
- Monitor CPU/GPU usage
- Try reducing game graphics settings

### Module Not Found

**Symptom**: "ModuleNotFoundError: No module named 'torch'"

**Solutions**:
- Activate virtual environment if using one
- Reinstall dependencies: `pip install -r requirements.txt`
- Verify Python version: `python --version`
- Try: `pip list` to see installed packages

## Next Steps

After successful installation:

1. **Test in Freeplay** - Start with simple scenarios
2. **Monitor Performance** - Check frame rates and latency
3. **Tune Frame Skip** - Adjust if model requires different rate
4. **Try Different Bots** - Test various models
5. **Record Gameplay** - Document bot behavior for analysis

## Getting Help

If you're still having issues:

1. Check the main README.md for additional information
2. Review BakkesMod console logs for detailed errors
3. Check Python console output for stack traces
4. Open an issue on GitHub with:
   - Your system specs
   - Error messages
   - Steps to reproduce
   - Relevant log files

## Advanced Configuration

### Custom Port

To use a different port:

1. Edit `RLBotBridge.cpp`, change `const int PORT = 5000;`
2. Rebuild plugin
3. Edit `main.py`, change `port=5000` parameter
4. Restart both components

### Frame Skip Adjustment

Edit `main.py`:
```python
self.frame_skip = 8  # Change this value
```

Common values:
- 8 frames = 15 FPS (Seer standard)
- 4 frames = 30 FPS (faster response)
- 12 frames = 10 FPS (slower, less CPU)

### CUDA Configuration

Force CPU mode:
```python
self.device = torch.device('cpu')
```

Force GPU mode:
```python
self.device = torch.device('cuda:0')
```

## Quick Reference

### File Locations

- **Plugin DLL**: `%APPDATA%\bakkesmod\bakkesmod\plugins\RLBotBridge.dll`
- **Bot Models**: `RLbot/bots/*/`
- **Python Script**: `RLbot/main.py`
- **Dependencies**: `RLbot/requirements.txt`

### Console Commands

- Load plugin: `plugin load rlbotbridge`
- Unload plugin: `plugin unload rlbotbridge`
- Toggle bot: `rlbot_toggle` (or F1 hotkey)

### File Structure Check

Your installation should look like:
```
RLbot/
├── BakkesModPlugin/
│   ├── build/
│   │   └── bin/Release/RLBotBridge.dll
│   ├── RLBotBridge.cpp
│   ├── RLBotBridge.h
│   └── CMakeLists.txt
├── bots/
│   └── YourBot/
│       └── model.pt
├── main.py
├── start_bot.bat
├── requirements.txt
└── README.md
```

---

**Congratulations!** You should now have a fully functional RLBot Bridge system. Happy botting!

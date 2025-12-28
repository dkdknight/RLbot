# Quick Start Guide

Get started with RLBot Bridge in 5 minutes!

## Prerequisites Check

Before starting, make sure you have:
- ✅ Rocket League installed
- ✅ BakkesMod installed and working
- ✅ Python 3.12+ installed

## Quick Installation

### 1. Build the Plugin (Windows)

```bash
cd BakkesModPlugin
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

Copy `build/bin/Release/RLBotBridge.dll` to:
```
%APPDATA%\bakkesmod\bakkesmod\plugins\
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or just run:
```bash
start_bot.bat
```

### 3. Add Your Bot

Place your trained model in the `bots/` directory:
```
bots/
  └── YourBot/
      └── model.pt
```

### 4. Launch!

1. Start Rocket League with BakkesMod
2. Run `start_bot.bat` or `python main.py`
3. Select your bot
4. Enter Freeplay
5. Press **F1** to enable

## Common Issues

### "Connection refused"
- Make sure BakkesMod plugin is loaded
- Check BakkesMod console: `plugin load rlbotbridge`

### "No bot models found"
- Add `.pt` files to `bots/YourBot/` directory
- Each bot needs its own subdirectory

### "Module not found"
- Run: `pip install -r requirements.txt`
- Or use `start_bot.bat` which installs automatically

## Next Steps

- Read [INSTALL.md](INSTALL.md) for detailed instructions
- Check [README.md](README.md) for architecture details
- See [bots/example_bot/README.md](bots/example_bot/README.md) for model requirements

## Key Features

- 🎮 **F1 Toggle**: Enable/disable bot control instantly
- ⚡ **120Hz**: High-frequency observation extraction
- 🧠 **LSTM Support**: Maintains hidden state for recurrent models
- 🎯 **Frame Skip**: Configurable to match training (default: 15 FPS)
- 🔌 **Direct Integration**: No RLBot framework needed

## File Locations

**Plugin**: `%APPDATA%\bakkesmod\bakkesmod\plugins\RLBotBridge.dll`  
**Bots**: `RLbot/bots/`  
**Main Script**: `RLbot/main.py`  
**Launcher**: `RLbot/start_bot.bat`

## Test Your Setup

Run the test suite:
```bash
python test_bot_manager.py
```

All tests should pass (8/8).

## Getting Help

1. Check the console output for errors
2. Read the [Troubleshooting](INSTALL.md#troubleshooting) section
3. Review BakkesMod console logs (F6 in-game)
4. Open an issue on GitHub

---

**Happy Botting!** 🚀

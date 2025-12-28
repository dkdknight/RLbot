# Troubleshooting Checklist

Use this checklist to diagnose and fix common issues with RLBot Bridge.

## ✅ Pre-Flight Checklist

Before running, verify:

- [ ] Rocket League is installed
- [ ] BakkesMod is installed and working
- [ ] Python 3.12+ is installed (`python --version`)
- [ ] Plugin DLL is in `%APPDATA%\bakkesmod\bakkesmod\plugins\`
- [ ] Dependencies are installed (`pip install -r requirements.txt`)
- [ ] At least one bot model exists in `bots/` directory

## 🔌 Plugin Issues

### Plugin Won't Load

**Symptoms:**
- "Plugin not found" error
- No socket server message in BakkesMod console

**Checklist:**
- [ ] DLL file exists: `%APPDATA%\bakkesmod\bakkesmod\plugins\RLBotBridge.dll`
- [ ] File name is exactly `RLBotBridge.dll` (case-sensitive)
- [ ] Plugin built in Release mode (not Debug)
- [ ] Visual C++ Redistributable installed
- [ ] Try manual load: `plugin load rlbotbridge` in BakkesMod console

**Fix:**
```bash
# Rebuild plugin
cd BakkesModPlugin/build
cmake --build . --config Release

# Copy to correct location
copy bin\Release\RLBotBridge.dll %APPDATA%\bakkesmod\bakkesmod\plugins\
```

### Plugin Loads but No Socket Server

**Symptoms:**
- Plugin loaded message appears
- No "Socket server listening on port 5000" message

**Checklist:**
- [ ] Check BakkesMod console for errors
- [ ] Port 5000 not in use: `netstat -an | findstr :5000`
- [ ] Firewall not blocking BakkesMod
- [ ] Try running as administrator

**Fix:**
- Close any program using port 5000
- Add BakkesMod to firewall exceptions
- Restart BakkesMod

## 🐍 Python Issues

### Module Not Found

**Symptoms:**
- `ModuleNotFoundError: No module named 'torch'`
- `ModuleNotFoundError: No module named 'numpy'`

**Checklist:**
- [ ] Python version is 3.12+: `python --version`
- [ ] Dependencies installed: `pip list | findstr torch`
- [ ] Using correct Python environment
- [ ] pip is up to date: `python -m pip install --upgrade pip`

**Fix:**
```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install numpy torch
```

### Connection Refused

**Symptoms:**
- "Connection refused" error
- "Failed to connect to BakkesMod plugin"

**Checklist:**
- [ ] BakkesMod plugin is loaded
- [ ] Socket server message appears in BakkesMod console
- [ ] Rocket League is running
- [ ] No firewall blocking localhost:5000
- [ ] Plugin loaded before running Python script

**Fix:**
1. Open BakkesMod console (F6)
2. Type: `plugin load rlbotbridge`
3. Look for: "Socket server listening on port 5000"
4. Run Python script again

### Model Loading Errors

**Symptoms:**
- "Error loading model"
- Tensor shape mismatch errors
- CUDA errors on CPU-only system

**Checklist:**
- [ ] Model file is valid PyTorch model (.pt extension)
- [ ] Model expects 159 inputs, 8 outputs
- [ ] PyTorch version compatible
- [ ] CUDA available if model requires GPU
- [ ] File path is correct

**Fix:**
```python
# Test loading model separately
import torch
model = torch.load('bots/YourBot/model.pt', weights_only=False)
print(type(model))
```

## 🎮 Game Issues

### Bot Doesn't Respond

**Symptoms:**
- Python connected
- In game but car doesn't move
- No errors shown

**Checklist:**
- [ ] Bot control enabled (press F1)
- [ ] Actually in a game (not menu)
- [ ] Using local player (not spectating)
- [ ] BakkesMod console shows no errors
- [ ] Python console shows observations being received

**Fix:**
1. Press F1 to toggle bot off then on
2. Check BakkesMod console for messages
3. Restart both Python script and game

### Poor Performance/Lag

**Symptoms:**
- Stuttering movement
- Delayed actions
- Low FPS

**Checklist:**
- [ ] GPU being utilized (if available)
- [ ] Frame skip set correctly (default: 8)
- [ ] No other heavy programs running
- [ ] Game graphics settings reasonable
- [ ] Python console not showing errors

**Fix:**
- Close unnecessary programs
- Lower game graphics settings
- Check CPU/GPU usage
- Adjust frame skip in `main.py`

### Bot Behavior Is Random/Bad

**Symptoms:**
- Car moves randomly
- Poor decisions
- Doesn't play properly

**Checklist:**
- [ ] Using trained model (not example bot)
- [ ] Model trained on Rocket League data
- [ ] Correct observation format
- [ ] Frame skip matches training
- [ ] LSTM hidden state maintained

**This is expected for:**
- Example bots (untrained)
- Improperly trained models
- Wrong observation format

## 🔍 Diagnostic Commands

### Check Python Environment
```bash
python --version
pip list
pip show torch numpy
```

### Check Port Usage
```bash
# Windows
netstat -an | findstr :5000

# Check if anything is listening
netstat -an | findstr LISTENING | findstr :5000
```

### Check BakkesMod Plugin
```
# In BakkesMod console (F6)
plugin list
plugin load rlbotbridge
```

### Test Python Components
```bash
python test_bot_manager.py
```

### Test Model Loading
```python
import torch
model = torch.load('bots/YourBot/model.pt', weights_only=False)
print(f"Model type: {type(model)}")
```

## 📝 Log Locations

Check these logs for detailed errors:

**BakkesMod Console:**
- Open with F6 in game
- Look for "RLBot Bridge" messages

**Python Console:**
- Running `python main.py` shows output
- Look for errors, connection status

**Windows Event Viewer:**
- May show application crashes
- Start > Event Viewer > Windows Logs > Application

## 🆘 Still Having Issues?

If none of the above helps:

1. **Gather Information:**
   - OS version
   - Python version
   - PyTorch version
   - Error messages (full text)
   - BakkesMod console output
   - Steps to reproduce

2. **Check Documentation:**
   - [README.md](README.md)
   - [INSTALL.md](INSTALL.md)
   - [QUICKSTART.md](QUICKSTART.md)

3. **Search Existing Issues:**
   - GitHub Issues page
   - Check closed issues too

4. **Ask for Help:**
   - Open a GitHub issue
   - Include all gathered information
   - Be detailed and specific

## 🔧 Emergency Reset

If everything is broken:

```bash
# 1. Clean Python environment
pip uninstall torch numpy -y
pip install -r requirements.txt

# 2. Rebuild plugin
cd BakkesModPlugin
rmdir /s build
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release

# 3. Reinstall plugin
copy bin\Release\RLBotBridge.dll %APPDATA%\bakkesmod\bakkesmod\plugins\

# 4. Restart everything
# - Close Rocket League
# - Close BakkesMod
# - Restart both
# - Load plugin
# - Run Python script
```

## ✅ Verification Steps

After fixing issues:

1. **Test Plugin:**
   - Load in BakkesMod
   - Check for socket server message
   
2. **Test Python:**
   - Run `python test_bot_manager.py`
   - All tests should pass (8/8)
   
3. **Test Connection:**
   - Start Python script
   - Should connect successfully
   
4. **Test In-Game:**
   - Enter Freeplay
   - Press F1
   - Bot should control car

## 📊 Success Indicators

You know it's working when:

- ✅ Plugin loads without errors
- ✅ "Socket server listening on port 5000" appears
- ✅ Python connects successfully
- ✅ "Bot is running! Waiting for observations..." appears
- ✅ Pressing F1 shows "Bot control ENABLED"
- ✅ Car moves when in game with F1 enabled

---

**Need more help?** Check [INSTALL.md](INSTALL.md) or open a GitHub issue.

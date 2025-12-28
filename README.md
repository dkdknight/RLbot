# RLBot - Custom BakkesMod Bridge for Rocket League AI

A custom bridge system that connects BakkesMod (C++) with Python AI bots for Rocket League, enabling models like Seer and Ripple to run without the standard RLBot interface.

## Overview

This project creates an autopilot system that allows AI models to control Rocket League directly through BakkesMod, bypassing the traditional RLBot framework. The system consists of three main components:

1. **BakkesMod Plugin (C++)**: Extracts game state and injects bot actions
2. **Python Bot Manager**: Loads AI models and performs inference
3. **Batch Launcher**: Simplifies setup and execution

## Features

- ✅ **Direct Game Integration**: Works through BakkesMod for minimal latency
- ✅ **Socket Communication**: TCP server/client on port 5000
- ✅ **120Hz Physics Tick**: High-frequency game state extraction
- ✅ **Seer/Ripple Compatible**: Supports 159-observation format
- ✅ **LSTM State Management**: Maintains hidden states between frames
- ✅ **Frame Skip**: Configurable to 15 FPS (8-frame skip) for model compatibility
- ✅ **F1 Hotkey**: Easy toggle for bot control
- ✅ **PyTorch Integration**: Load and run .pt model files
- ✅ **Multi-Bot Support**: Scan and select from multiple bot models

## System Architecture

```
┌─────────────────────┐         TCP Socket (Port 5000)         ┌──────────────────┐
│  Rocket League +    │◄────────────────────────────────────────►│  Python Bot      │
│  BakkesMod Plugin   │                                         │  Manager         │
│                     │   Observations (159 floats @ 120Hz)     │                  │
│  - Extract State    │──────────────────────────────────────────►│  - Load Model    │
│  - Apply Actions    │◄──────────────────────────────────────────│  - Inference     │
│  - F1 Toggle        │   Actions (8 floats)                     │  - Frame Skip    │
└─────────────────────┘                                         └──────────────────┘
```

## Installation

### Prerequisites

- **Rocket League** (Epic or Steam version)
- **BakkesMod**: [Download here](https://www.bakkesmod.com/)
- **Python 3.12+**: [Download here](https://www.python.org/)
- **Visual Studio 2019+**: For building the BakkesMod plugin
- **CMake 3.15+**: For C++ build configuration

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dkdknight/RLbot.git
   cd RLbot
   ```

2. **Build the BakkesMod Plugin**:
   ```bash
   cd BakkesModPlugin
   mkdir build
   cd build
   cmake .. -G "Visual Studio 16 2019" -A x64
   cmake --build . --config Release
   ```
   
3. **Install the plugin**:
   Copy `RLBotBridge.dll` to `%APPDATA%\bakkesmod\bakkesmod\plugins\`

4. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Add your bot models**:
   Place your trained models in the `bots/` directory:
   ```
   bots/
   ├── Seer/
   │   └── Seer.pt
   └── Ripple/
       └── Ripple.pt
   ```

6. **Launch**:
   ```bash
   start_bot.bat
   ```

## Usage

### Running the Bot

1. **Start Rocket League** with BakkesMod loaded
2. **Load the plugin** (should auto-load if in plugins directory):
   ```
   plugin load rlbotbridge
   ```
3. **Run the Python bot manager**:
   ```bash
   python main.py
   ```
   or simply double-click `start_bot.bat`

4. **Select your bot** from the menu

5. **Enter a game** (Freeplay recommended for testing)

6. **Press F1** to enable bot control

7. **Press F1 again** to disable when done

### Configuration

- **Port**: Default 5000 (change in both C++ and Python if needed)
- **Frame Skip**: Default 8 frames (15 FPS) - edit in `main.py`
- **Model Path**: Automatically scans `bots/` directory

## Data Formats

### Observation Space (159 values)

The plugin extracts normalized game state:

| Component | Count | Description |
|-----------|-------|-------------|
| Ball State | 12 | Position (3), Velocity (3), Rotation (3), Angular Velocity (3) |
| Player Car | 47 | Position (3), Velocity (3), Rotation (3), Angular Velocity (3), Orientation Vectors (9), Boost (1), Flags (4), etc. |
| Other Cars | ~230 | Similar data for up to 5 other cars |
| Boost Pads | ~68 | State of 34 boost pads |
| Padding | Variable | Padded to 159 total |

All values are normalized to approximately [-1, 1] range for neural network compatibility.

### Action Space (8 values)

| Index | Control | Type | Range |
|-------|---------|------|-------|
| 0 | Throttle | Float | -1.0 to 1.0 |
| 1 | Steer | Float | -1.0 to 1.0 |
| 2 | Pitch | Float | -1.0 to 1.0 |
| 3 | Yaw | Float | -1.0 to 1.0 |
| 4 | Roll | Float | -1.0 to 1.0 |
| 5 | Jump | Bool | 0.0 or 1.0 |
| 6 | Boost | Bool | 0.0 or 1.0 |
| 7 | Handbrake | Bool | 0.0 or 1.0 |

## Model Requirements

Your PyTorch model should:

1. **Input**: Accept 159 observation values
2. **Output**: Produce 8 action values (or multi-discrete equivalent)
3. **LSTM Support**: Optionally maintain hidden state for recurrent models
4. **Format**: Saved as `.pt` file using `torch.save()`

### Example Model Structure

```python
import torch
import torch.nn as nn

class LSTMBot(nn.Module):
    def __init__(self, input_size=159, hidden_size=256, action_size=8):
        super(LSTMBot, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, action_size)
    
    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        action = self.fc(lstm_out[:, -1, :])
        return action, hidden
```

## Project Structure

```
RLbot/
├── BakkesModPlugin/          # C++ BakkesMod plugin
│   ├── RLBotBridge.h         # Plugin header
│   ├── RLBotBridge.cpp       # Plugin implementation
│   ├── CMakeLists.txt        # Build configuration
│   └── README.md             # Plugin documentation
├── bots/                     # Bot models directory
│   └── example_bot/          # Example bot structure
│       └── README.md         # Bot configuration guide
├── main.py                   # Python bot manager
├── start_bot.bat             # Windows launcher script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Technical Details

### Communication Protocol

1. Plugin waits for client connection on port 5000
2. On each physics tick (if bot enabled):
   - Plugin sends observation size (4 bytes, int32)
   - Plugin sends observations (159 × 4 bytes, float32 array)
   - Plugin receives actions (8 × 4 bytes, float32 array)
   - Plugin applies actions to car controller

### Frame Skip Implementation

The Python client implements frame skip (default 8 frames):
- Receives observations at 120Hz
- Computes new actions every 8th frame (15Hz)
- Reuses last action for intermediate frames
- Matches training conditions for Seer model

### LSTM State Management

For recurrent models:
- Hidden state is maintained between inference calls
- State persists across frames
- State resets when connection is lost or bot is disabled

## Troubleshooting

### Plugin Issues

**Plugin doesn't load:**
- Check BakkesMod console for errors
- Verify DLL is in `%APPDATA%\bakkesmod\bakkesmod\plugins\`
- Ensure Visual C++ Redistributable is installed

**No observations sent:**
- Check if bot is enabled (press F1)
- Verify you're in a game (not menu)
- Check BakkesMod console logs

### Python Issues

**Connection refused:**
- Ensure BakkesMod plugin is loaded
- Check if port 5000 is available
- Verify firewall isn't blocking connection

**Model loading errors:**
- Check model file format (.pt extension)
- Verify PyTorch version compatibility
- Try loading model separately to test

**Poor performance:**
- Check if CUDA is being used (if GPU available)
- Verify frame skip is set correctly
- Monitor CPU/GPU usage

## Performance Tips

1. **Use GPU**: If available, PyTorch will use CUDA automatically
2. **Frame Skip**: Adjust based on model requirements (Seer uses 8)
3. **Close Other Programs**: Free up CPU/GPU resources
4. **Freeplay Mode**: Best for testing with minimal overhead

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Known Limitations

- Single bot instance per game
- Local connection only (no network play)
- Windows only (BakkesMod limitation)
- No official RLBot compatibility

## Future Enhancements

- [ ] Multi-agent support (multiple bots)
- [ ] Enhanced observation space (all boost pads, etc.)
- [ ] Model hot-reloading
- [ ] GUI for bot selection
- [ ] Recording/replay system
- [ ] Performance metrics dashboard

## Credits

- **BakkesMod**: Essential modding framework
- **Seer/Ripple**: Inspiration for observation space design
- **RLBot Community**: Rocket League AI research

## License

This project is provided as-is for educational and research purposes. Use responsibly.

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review BakkesMod and PyTorch documentation

---

**Warning**: This is a custom implementation and may not work with all models or game modes. Always test in Freeplay first!
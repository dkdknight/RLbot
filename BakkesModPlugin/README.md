# BakkesMod Plugin - RLBot Bridge

This directory contains the C++ BakkesMod plugin that acts as a bridge between Rocket League and the Python bot manager.

## Overview

The plugin creates a TCP socket server on port 5000 that:
- Extracts game state observations (159 values) at 120Hz physics tick rate
- Sends observations to the Python client
- Receives action commands from the Python client
- Applies actions to the player's car via controller input injection

## Features

- **Socket Server**: TCP server listening on localhost:5000
- **Physics Tick Hook**: Captures game state at 120Hz
- **Observation Extraction**: Extracts 159 Seer-compatible observations
- **Action Application**: Applies 8-dimensional action vector to car controls
- **F1 Hotkey**: Toggle bot control on/off with F1 key
- **Thread-Safe**: Asynchronous socket communication

## Building the Plugin

### Prerequisites

1. **BakkesMod SDK**: Download from [BakkesMod](https://www.bakkesmod.com/)
2. **Visual Studio 2019+**: With C++ desktop development tools
3. **CMake 3.15+**: For build configuration

### Build Steps

1. Clone or download the BakkesMod SDK
2. Update `CMakeLists.txt` with your SDK path
3. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```
4. Generate Visual Studio project:
   ```bash
   cmake .. -G "Visual Studio 16 2019" -A x64
   ```
5. Build the plugin:
   ```bash
   cmake --build . --config Release
   ```

### Installation

1. Copy the compiled `RLBotBridge.dll` to your BakkesMod plugins folder:
   ```
   %APPDATA%\bakkesmod\bakkesmod\plugins\
   ```

2. Create/edit `%APPDATA%\bakkesmod\bakkesmod\plugins\RLBotBridge.set` (optional settings file)

3. Load the plugin in BakkesMod console:
   ```
   plugin load rlbotbridge
   ```

## Usage

1. Start Rocket League with BakkesMod
2. Load the RLBotBridge plugin (should auto-load if in plugins directory)
3. Run the Python bot manager (`python main.py`)
4. Enter a game (Freeplay recommended)
5. Press **F1** to enable bot control
6. Press **F1** again to disable bot control

## Commands

- `rlbot_toggle` - Toggle bot control on/off (bound to F1)

## Technical Details

### Observation Space (159 values)

The plugin extracts:
- **Ball**: Position (3), Velocity (3), Rotation (3), Angular Velocity (3)
- **Player Car**: Position (3), Velocity (3), Rotation (3), Angular Velocity (3), Forward/Up/Right vectors (9), Boost (1), Flags (4)
- **Other Data**: Padded to 159 total values for Seer compatibility

### Action Space (8 values)

- **Throttle** (float, -1 to 1): Forward/backward
- **Steer** (float, -1 to 1): Left/right steering
- **Pitch** (float, -1 to 1): Nose up/down
- **Yaw** (float, -1 to 1): Turn left/right in air
- **Roll** (float, -1 to 1): Roll left/right in air
- **Jump** (bool): Jump button
- **Boost** (bool): Boost button
- **Handbrake** (bool): Powerslide/handbrake

### Communication Protocol

1. Plugin sends observation size (4 bytes, int)
2. Plugin sends observations (159 * 4 bytes, float array)
3. Python client sends actions (8 * 4 bytes, float array)
4. Repeat at 120Hz (or lower based on frame skip)

## Notes

- The plugin runs at 120Hz (physics tick rate)
- Frame skipping should be handled on the Python side
- The socket server is single-client (one bot at a time)
- Connection is non-blocking for the game
- Hotkey can be rebound in BakkesMod settings

## Troubleshooting

**Plugin doesn't load:**
- Check BakkesMod console for error messages
- Verify DLL is in correct plugins directory
- Ensure all dependencies are present

**Connection fails:**
- Check if port 5000 is available
- Verify firewall settings
- Check BakkesMod console logs

**Bot doesn't respond:**
- Press F1 to enable bot control
- Check Python client is connected
- Verify observations are being sent (console logs)

## License

This plugin is provided as-is for educational and research purposes.

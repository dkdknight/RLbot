# Example Bot Configuration

This is an example bot directory. Place your trained PyTorch model here.

## Expected Files

- `*.pt` - Your trained model file (e.g., `Seer.pt`, `Ripple.pt`, `my_bot.pt`)

## Model Requirements

The model should:
- Accept 159 observation values as input
- Output 8 action values (or multi-discrete equivalent)
- Support LSTM hidden state management (if applicable)

## Model Format

The model can be saved as:
1. Complete model: `torch.save(model, 'bot.pt')`
2. State dict: `torch.save(model.state_dict(), 'bot.pt')`

The loader will attempt to handle both formats.

## Action Space

The bot expects 8 continuous actions:
1. **Throttle** (-1 to 1): Forward/backward
2. **Steer** (-1 to 1): Left/right steering
3. **Pitch** (-1 to 1): Nose up/down
4. **Yaw** (-1 to 1): Turn left/right in air
5. **Roll** (-1 to 1): Roll left/right in air
6. **Jump** (0 or 1): Jump button
7. **Boost** (0 or 1): Boost button
8. **Handbrake** (0 or 1): Powerslide/handbrake

## Observation Space

The bot receives 159 observations including:
- Ball state (position, velocity, rotation, angular velocity)
- Player car state (position, velocity, rotation, angular velocity, boost, flags)
- Other cars state (for multi-agent)
- Boost pad states
- Game state information

Values are normalized for neural network training.

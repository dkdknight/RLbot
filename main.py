import socket
import struct
import time
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


class LSTMBot(nn.Module):
    """
    Basic LSTM bot architecture compatible with Seer/Ripple models.
    This is a placeholder architecture - actual model should be loaded from .pt files.
    """
    def __init__(self, input_size=159, hidden_size=256, action_size=8):
        super(LSTMBot, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, action_size)
        
    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        lstm_out, hidden = self.lstm(x, hidden)
        action = self.fc(lstm_out[:, -1, :])
        return action, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))


class BotManager:
    # Action space constants
    ANALOG_ACTIONS_START = 0
    ANALOG_ACTIONS_END = 5
    BUTTON_ACTIONS_START = 5
    BUTTON_ACTIONS_END = 8
    
    def __init__(self, bot_path, host='localhost', port=5000):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
        # Frame skip for 15 FPS (120Hz / 8 = 15Hz)
        self.frame_skip = 8
        self.frame_counter = 0
        
        # Load bot model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = self.load_bot(bot_path)
        self.hidden_state = None
        self.last_action = np.zeros(8, dtype=np.float32)
        
        print(f"Bot loaded from: {bot_path}")
        
    def load_bot(self, bot_path):
        """Load the bot model from a .pt file"""
        try:
            # Try to load the entire model (with weights_only=False for full model loading)
            model = torch.load(bot_path, map_location=self.device, weights_only=False)
            
            # If it's a state dict, create a model and load it
            if isinstance(model, dict):
                bot_model = LSTMBot().to(self.device)
                bot_model.load_state_dict(model)
                model = bot_model
            
            model.eval()
            print("Model loaded successfully")
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using default LSTM architecture")
            model = LSTMBot().to(self.device)
            model.eval()
            return model
    
    def connect(self):
        """Connect to the BakkesMod plugin via TCP socket"""
        print(f"Connecting to BakkesMod plugin at {self.host}:{self.port}...")
        
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                self.connected = True
                print("Connected successfully!")
                return True
            except ConnectionRefusedError:
                print(f"Connection attempt {attempt + 1}/{max_retries} failed. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            except Exception as e:
                print(f"Connection error: {e}")
                time.sleep(retry_delay)
        
        print("Failed to connect to BakkesMod plugin")
        return False
    
    def receive_observations(self):
        """Receive observation data from BakkesMod plugin"""
        try:
            # Receive size first (4 bytes)
            size_data = self.socket.recv(4)
            if len(size_data) < 4:
                return None
            
            obs_size = struct.unpack('i', size_data)[0]
            
            # Receive observations (float array)
            obs_bytes = obs_size * 4  # 4 bytes per float
            data = b''
            while len(data) < obs_bytes:
                chunk = self.socket.recv(obs_bytes - len(data))
                if not chunk:
                    return None
                data += chunk
            
            observations = struct.unpack(f'{obs_size}f', data)
            return np.array(observations, dtype=np.float32)
            
        except Exception as e:
            print(f"Error receiving observations: {e}")
            self.connected = False
            return None
    
    def send_actions(self, actions):
        """Send action data to BakkesMod plugin"""
        try:
            # Pack actions as float array
            action_data = struct.pack(f'{len(actions)}f', *actions)
            self.socket.sendall(action_data)
            return True
        except Exception as e:
            print(f"Error sending actions: {e}")
            self.connected = False
            return False
    
    def multi_discrete_to_continuous(self, action_tensor):
        """
        Convert multi-discrete action format to continuous format expected by RL
        
        Seer uses multi-discrete action space with bins for each control.
        This converts to continuous values [-1, 1] for axes and booleans for buttons.
        
        Action space (example):
        - Throttle: 3 bins -> [-1, 0, 1]
        - Steer: 3 bins -> [-1, 0, 1]
        - Pitch: 3 bins -> [-1, 0, 1]
        - Yaw: 3 bins -> [-1, 0, 1]
        - Roll: 3 bins -> [-1, 0, 1]
        - Jump: 2 bins -> [0, 1]
        - Boost: 2 bins -> [0, 1]
        - Handbrake: 2 bins -> [0, 1]
        """
        # If the model outputs continuous values, clamp them
        if action_tensor.shape[-1] == 8:
            actions = action_tensor.detach().cpu().numpy().flatten()
            
            # Clamp analog inputs to [-1, 1]
            actions[self.ANALOG_ACTIONS_START:self.ANALOG_ACTIONS_END] = np.clip(
                actions[self.ANALOG_ACTIONS_START:self.ANALOG_ACTIONS_END], -1.0, 1.0
            )
            
            # Convert buttons to binary
            actions[self.BUTTON_ACTIONS_START:self.BUTTON_ACTIONS_END] = (
                actions[self.BUTTON_ACTIONS_START:self.BUTTON_ACTIONS_END] > 0.0
            ).astype(np.float32)
            
            return actions
        
        # Handle multi-discrete case (if needed)
        # This would require knowing the exact bin structure
        actions = action_tensor.detach().cpu().numpy().flatten()
        return actions
    
    def predict_action(self, observations):
        """Generate action from observations using the loaded model"""
        with torch.no_grad():
            # Convert observations to tensor
            obs_tensor = torch.from_numpy(observations).float().unsqueeze(0).unsqueeze(0)
            obs_tensor = obs_tensor.to(self.device)
            
            # Forward pass through model
            if hasattr(self.model, 'forward') and 'hidden' in self.model.forward.__code__.co_varnames:
                # LSTM model with hidden state
                action_tensor, self.hidden_state = self.model(obs_tensor, self.hidden_state)
            else:
                # Standard feedforward model
                action_tensor = self.model(obs_tensor)
            
            # Convert to continuous actions
            actions = self.multi_discrete_to_continuous(action_tensor)
            
            return actions
    
    def run(self):
        """Main loop for bot execution"""
        if not self.connect():
            return
        
        print("\nBot is running! Waiting for observations...")
        print("Press F1 in Rocket League to enable bot control")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.connected:
                # Receive observations from BakkesMod
                observations = self.receive_observations()
                
                if observations is None:
                    print("Connection lost")
                    break
                
                # Frame skip logic: only compute new actions every N frames
                self.frame_counter += 1
                if self.frame_counter >= self.frame_skip:
                    self.frame_counter = 0
                    
                    # Predict action using the model
                    actions = self.predict_action(observations)
                    self.last_action = actions
                else:
                    # Reuse last action
                    actions = self.last_action
                
                # Send actions back to BakkesMod
                if not self.send_actions(actions):
                    print("Failed to send actions")
                    break
                    
        except KeyboardInterrupt:
            print("\nStopping bot...")
        except socket.error as e:
            print(f"Socket error in main loop: {e}")
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.disconnect()
    
    def disconnect(self):
        """Close socket connection"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False
        print("Disconnected")


def scan_bots(bots_dir='bots'):
    """Scan the bots directory for available bot models"""
    bots_path = Path(bots_dir)
    
    if not bots_path.exists():
        print(f"Bots directory '{bots_dir}' not found. Creating it...")
        bots_path.mkdir(parents=True, exist_ok=True)
        print(f"Please place your bot models (.pt files) in '{bots_path.absolute()}'")
        return []
    
    # Find all .pt files in subdirectories
    bot_models = {}
    for subdir in bots_path.iterdir():
        if subdir.is_dir():
            for model_file in subdir.glob('*.pt'):
                bot_name = f"{subdir.name}/{model_file.name}"
                bot_models[bot_name] = model_file
    
    return bot_models


def select_bot(bot_models):
    """Interactive bot selection menu"""
    if not bot_models:
        print("\nNo bot models found!")
        print(f"Please place your bot models in the 'bots' directory.")
        print("Each bot should be in its own subdirectory with a .pt file.")
        print("\nExample structure:")
        print("  bots/")
        print("    Seer/")
        print("      Seer.pt")
        print("    Ripple/")
        print("      Ripple.pt")
        return None
    
    print("\n=== Available Bots ===")
    bot_list = list(bot_models.items())
    for idx, (name, path) in enumerate(bot_list, 1):
        print(f"{idx}. {name}")
    
    while True:
        try:
            choice = input("\nSelect a bot (enter number): ").strip()
            idx = int(choice) - 1
            
            if 0 <= idx < len(bot_list):
                selected_name, selected_path = bot_list[idx]
                print(f"\nSelected: {selected_name}")
                return selected_path
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None


def main():
    print("=" * 50)
    print("RLBot Bridge - Python Bot Manager")
    print("=" * 50)
    
    # Check for PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("ERROR: PyTorch is not installed!")
        print("Please install it with: pip install torch")
        return
    
    # Scan for available bots
    bot_models = scan_bots()
    
    # Let user select a bot
    bot_path = select_bot(bot_models)
    
    if bot_path is None:
        return
    
    # Create and run bot manager
    try:
        manager = BotManager(bot_path)
        manager.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

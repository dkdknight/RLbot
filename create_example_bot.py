"""
Example script showing how to create a compatible bot model for RLBot Bridge.

This script demonstrates:
1. Creating a PyTorch model with LSTM architecture
2. Correct input/output dimensions
3. Saving the model in compatible format
"""

import torch
import torch.nn as nn


class ExampleLSTMBot(nn.Module):
    """
    Example LSTM bot compatible with RLBot Bridge.
    
    This architecture:
    - Accepts 159 observations (Seer format)
    - Uses LSTM for sequential processing
    - Outputs 8 continuous actions
    - Maintains hidden state between calls
    """
    
    def __init__(self, input_size=159, hidden_size=256, num_layers=2, action_size=8):
        super(ExampleLSTMBot, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.action_size = action_size
        
        # LSTM layer(s)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        # Optional: Add a layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
    def forward(self, x, hidden=None):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional LSTM hidden state tuple (h, c)
        
        Returns:
            actions: Action tensor of shape (batch, action_size)
            hidden: Updated LSTM hidden state
        """
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Layer normalization
        normalized = self.layer_norm(last_output)
        
        # Action head
        actions = self.action_head(normalized)
        
        # Apply activation functions appropriate for each action
        # Throttle, Steer, Pitch, Yaw, Roll: tanh for [-1, 1] range
        actions[:, 0:5] = torch.tanh(actions[:, 0:5])
        
        # Jump, Boost, Handbrake: sigmoid for [0, 1] range
        actions[:, 5:8] = torch.sigmoid(actions[:, 5:8])
        
        return actions, hidden
    
    def init_hidden(self, batch_size, device='cpu'):
        """Initialize hidden state for LSTM"""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h, c)


class SimpleFeedforwardBot(nn.Module):
    """
    Example feedforward bot (no LSTM) compatible with RLBot Bridge.
    
    Simpler than LSTM but may not capture temporal patterns as well.
    """
    
    def __init__(self, input_size=159, hidden_size=512, action_size=8):
        super(SimpleFeedforwardBot, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
    def forward(self, x, hidden=None):
        """Forward pass (hidden state ignored for compatibility)"""
        # Flatten if necessary
        if len(x.shape) == 3:
            x = x[:, -1, :]  # Take last timestep
        
        actions = self.network(x)
        
        # Apply appropriate activations
        actions[:, 0:5] = torch.tanh(actions[:, 0:5])
        actions[:, 5:8] = torch.sigmoid(actions[:, 5:8])
        
        return actions, None  # Return None for hidden state


def create_and_save_example_bot(save_path='bots/example_bot/example_model.pt'):
    """
    Create and save an example bot model.
    
    This creates an untrained model just to demonstrate the format.
    You would normally train this model on Rocket League data first.
    """
    import os
    
    print("Creating example bot model...")
    
    # Create model
    model = ExampleLSTMBot(
        input_size=159,
        hidden_size=256,
        num_layers=2,
        action_size=8
    )
    
    # Set to eval mode
    model.eval()
    
    # Print model info
    print(f"\nModel Architecture:")
    print(f"  Input size: {model.input_size}")
    print(f"  Hidden size: {model.hidden_size}")
    print(f"  Num layers: {model.num_layers}")
    print(f"  Action size: {model.action_size}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    dummy_input = torch.randn(1, 1, 159)
    with torch.no_grad():
        actions, hidden = model(dummy_input)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {actions.shape}")
    print(f"  Sample actions: {actions[0].numpy()}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save model
    torch.save(model, save_path)
    print(f"\nModel saved to: {save_path}")
    
    # Verify loading
    print(f"Verifying model can be loaded...")
    loaded_model = torch.load(save_path, weights_only=False)
    loaded_model.eval()
    
    with torch.no_grad():
        test_actions, _ = loaded_model(dummy_input)
    
    print(f"✓ Model loaded and tested successfully!")
    
    return model


def main():
    """Main function"""
    print("=" * 60)
    print("RLBot Bridge - Example Bot Creator")
    print("=" * 60)
    print()
    print("This script creates example bot models compatible with")
    print("the RLBot Bridge system.")
    print()
    
    # Create LSTM bot
    print("Creating LSTM bot...")
    create_and_save_example_bot('bots/example_bot/lstm_bot.pt')
    
    print("\n" + "=" * 60)
    print("Creating feedforward bot...")
    model = SimpleFeedforwardBot()
    torch.save(model, 'bots/example_bot/feedforward_bot.pt')
    print(f"Feedforward bot saved!")
    
    print("\n" + "=" * 60)
    print("Example bots created successfully!")
    print()
    print("IMPORTANT: These are untrained models and will not perform well.")
    print("They are provided only as examples of the correct format.")
    print()
    print("To use these models:")
    print("1. Run 'python main.py'")
    print("2. Select one of the example bots")
    print("3. Test in Rocket League (behavior will be random)")
    print()
    print("To create a real bot:")
    print("1. Collect Rocket League gameplay data")
    print("2. Train your model with that data")
    print("3. Save trained model in same format")
    print("4. Place in bots/ directory")
    print("=" * 60)


if __name__ == "__main__":
    main()

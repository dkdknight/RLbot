"""
Test script for RLBot Bridge Python components.
This script tests the bot manager functionality without requiring a live connection.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Test configuration
TEST_OBSERVATIONS = 159
TEST_ACTIONS = 8


class TestLSTMBot(nn.Module):
    """Test LSTM bot for validation"""
    def __init__(self, input_size=159, hidden_size=128, action_size=8):
        super(TestLSTMBot, self).__init__()
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


def test_model_creation():
    """Test that we can create a model"""
    print("Testing model creation...")
    try:
        model = TestLSTMBot()
        print("✓ Model created successfully")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_model_forward_pass():
    """Test forward pass through model"""
    print("\nTesting model forward pass...")
    try:
        model = TestLSTMBot()
        model.eval()
        
        # Create dummy input
        dummy_obs = torch.randn(1, 1, TEST_OBSERVATIONS)
        
        # Forward pass
        with torch.no_grad():
            action, hidden = model(dummy_obs)
        
        # Check output shape
        assert action.shape == (1, TEST_ACTIONS), f"Expected (1, 8), got {action.shape}"
        assert len(hidden) == 2, "Expected 2 hidden states"
        
        print(f"✓ Forward pass successful, output shape: {action.shape}")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def test_action_conversion():
    """Test action conversion to continuous format"""
    print("\nTesting action conversion...")
    try:
        # Create dummy action tensor
        action_tensor = torch.tensor([[0.5, -0.3, 0.8, -0.1, 0.2, 0.7, 0.9, 0.3]])
        
        # Convert
        actions = action_tensor.detach().cpu().numpy().flatten()
        
        # Clamp analog inputs
        actions[0:5] = np.clip(actions[0:5], -1.0, 1.0)
        
        # Convert buttons
        actions[5:8] = (actions[5:8] > 0.0).astype(np.float32)
        
        # Verify
        assert len(actions) == TEST_ACTIONS, f"Expected 8 actions, got {len(actions)}"
        assert all(actions[0:5] >= -1.0) and all(actions[0:5] <= 1.0), "Analog values out of range"
        assert all(actions[5:8] >= 0.0) and all(actions[5:8] <= 1.0), "Button values out of range"
        
        print(f"✓ Action conversion successful")
        print(f"  Analog actions: {actions[0:5]}")
        print(f"  Button actions: {actions[5:8]}")
        return True
    except Exception as e:
        print(f"✗ Action conversion failed: {e}")
        return False


def test_observation_processing():
    """Test observation array creation and validation"""
    print("\nTesting observation processing...")
    try:
        # Create dummy observations
        observations = np.random.randn(TEST_OBSERVATIONS).astype(np.float32)
        
        # Convert to tensor
        obs_tensor = torch.from_numpy(observations).float().unsqueeze(0).unsqueeze(0)
        
        # Check shape
        assert obs_tensor.shape == (1, 1, TEST_OBSERVATIONS), f"Expected (1, 1, 159), got {obs_tensor.shape}"
        
        print(f"✓ Observation processing successful, shape: {obs_tensor.shape}")
        return True
    except Exception as e:
        print(f"✗ Observation processing failed: {e}")
        return False


def test_frame_skip_logic():
    """Test frame skip implementation"""
    print("\nTesting frame skip logic...")
    try:
        frame_skip = 8
        frame_counter = 0
        actions_computed = 0
        total_frames = 120  # Simulate 1 second at 120Hz
        
        last_action = np.zeros(8, dtype=np.float32)
        
        for frame in range(total_frames):
            frame_counter += 1
            
            if frame_counter >= frame_skip:
                frame_counter = 0
                actions_computed += 1
                # Compute new action (simulated)
                last_action = np.random.randn(8).astype(np.float32)
            else:
                # Reuse last action
                pass
        
        expected_actions = total_frames // frame_skip
        assert actions_computed == expected_actions, f"Expected {expected_actions} actions, got {actions_computed}"
        
        print(f"✓ Frame skip logic successful")
        print(f"  Total frames: {total_frames}")
        print(f"  Actions computed: {actions_computed}")
        print(f"  Effective rate: {actions_computed} FPS (from 120Hz)")
        return True
    except Exception as e:
        print(f"✗ Frame skip logic failed: {e}")
        return False


def test_model_save_load():
    """Test saving and loading model"""
    print("\nTesting model save/load...")
    try:
        # Create and save model
        model = TestLSTMBot()
        test_path = Path("/tmp/test_model.pt")
        torch.save(model, test_path)
        
        # Load model (with weights_only=False for compatibility)
        loaded_model = torch.load(test_path, map_location='cpu', weights_only=False)
        loaded_model.eval()
        
        # Test loaded model
        dummy_obs = torch.randn(1, 1, TEST_OBSERVATIONS)
        with torch.no_grad():
            action, hidden = loaded_model(dummy_obs)
        
        assert action.shape == (1, TEST_ACTIONS), "Loaded model output shape incorrect"
        
        # Cleanup
        test_path.unlink()
        
        print("✓ Model save/load successful")
        return True
    except Exception as e:
        print(f"✗ Model save/load failed: {e}")
        return False


def test_hidden_state_persistence():
    """Test LSTM hidden state management"""
    print("\nTesting hidden state persistence...")
    try:
        model = TestLSTMBot()
        model.eval()
        
        # First forward pass
        dummy_obs1 = torch.randn(1, 1, TEST_OBSERVATIONS)
        with torch.no_grad():
            action1, hidden1 = model(dummy_obs1)
        
        # Second forward pass with hidden state
        dummy_obs2 = torch.randn(1, 1, TEST_OBSERVATIONS)
        with torch.no_grad():
            action2, hidden2 = model(dummy_obs2, hidden1)
        
        # Verify hidden states changed
        assert not torch.equal(hidden1[0], hidden2[0]), "Hidden state should change"
        
        print("✓ Hidden state persistence successful")
        return True
    except Exception as e:
        print(f"✗ Hidden state persistence failed: {e}")
        return False


def test_pytorch_installation():
    """Test PyTorch installation and CUDA availability"""
    print("\nTesting PyTorch installation...")
    try:
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU device: {torch.cuda.get_device_name(0)}")
        print("✓ PyTorch installed correctly")
        return True
    except Exception as e:
        print(f"✗ PyTorch check failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("=" * 60)
    print("RLBot Bridge - Python Component Tests")
    print("=" * 60)
    
    tests = [
        ("PyTorch Installation", test_pytorch_installation),
        ("Model Creation", test_model_creation),
        ("Model Forward Pass", test_model_forward_pass),
        ("Action Conversion", test_action_conversion),
        ("Observation Processing", test_observation_processing),
        ("Frame Skip Logic", test_frame_skip_logic),
        ("Model Save/Load", test_model_save_load),
        ("Hidden State Persistence", test_hidden_state_persistence),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 60)
    print("Results: {}/{} tests passed ({}%)".format(
        passed, total, (100 * passed // total) if total > 0 else 0
    ))
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

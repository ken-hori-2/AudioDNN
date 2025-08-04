"""
Test script for U-Net model functionality.
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models import create_unet_model, SourceSeparator

def test_unet_model():
    """Test U-Net model creation and forward pass."""
    print("Testing U-Net model...")
    
    # Create model
    model = create_unet_model(
        model_type="unet",
        n_sources=4,
        n_channels=2,
        n_fft=4096
    )
    
    print(f"Model created successfully")
    print(f"Model type: {type(model).__name__}")
    
    # Test input shape (適切なサイズに調整)
    batch_size = 2
    channels = 2  # stereo audio
    freq_bins = 512  # より小さなサイズで開始
    time_frames = 128  # 2の累乗サイズ
    
    # Create test input (batch, channels, freq, time)
    test_input = torch.randn(batch_size, channels, freq_bins, time_frames)
    print(f"Test input shape: {test_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: {(batch_size, 4 * channels, freq_bins, time_frames)}")
    
    # Check output shape
    expected_shape = (batch_size, 4 * channels, freq_bins, time_frames)
    assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
    
    print("✅ U-Net model test passed!")
    return model

def test_source_separator():
    """Test SourceSeparator functionality."""
    print("\nTesting SourceSeparator...")
    
    # Create model
    model = create_unet_model(
        model_type="unet",
        n_sources=4,
        n_channels=2,
        n_fft=4096
    )
    
    # Create separator
    separator = SourceSeparator(
        model=model,
        n_fft=4096,
        hop_length=1024,
        source_names=["vocals", "drums", "bass", "other"],
        sample_rate=44100
    )
    
    print("SourceSeparator created successfully")
    print(f"Source names: {separator.source_names}")
    
    # Test audio input (3 seconds stereo)
    test_audio = torch.randn(1, 2, 44100 * 3)  # (batch, channels, time)
    print(f"Test audio shape: {test_audio.shape}")
    
    # Separate sources
    separator.eval()
    with torch.no_grad():
        separated = separator(test_audio)
    
    print(f"Separated sources: {list(separated.keys())}")
    for name, audio in separated.items():
        print(f"  {name}: {audio.shape}")
    
    # Check outputs
    expected_sources = {"vocals", "drums", "bass", "other"}
    assert set(separated.keys()) == expected_sources, f"Missing sources: {expected_sources - set(separated.keys())}"
    
    for name, audio in separated.items():
        assert audio.shape == test_audio.shape, f"Shape mismatch for {name}: {audio.shape} vs {test_audio.shape}"
    
    print("✅ SourceSeparator test passed!")
    return separator

def test_model_parameters():
    """Test model parameter count and memory usage."""
    print("\nTesting model parameters...")
    
    model = create_unet_model(
        model_type="unet",
        n_sources=4,
        n_channels=2,
        n_fft=4096
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / (1024**2):.2f} MB (float32)")
    
    # Test with different input sizes (適切なサイズ)
    test_shapes = [
        (1, 2, 256, 64),     # Small
        (1, 2, 512, 128),    # Medium  
        (1, 2, 1024, 256),   # Large
    ]
    
    model.eval()
    for shape in test_shapes:
        test_input = torch.randn(*shape)
        with torch.no_grad():
            output = model(test_input)
        print(f"Input {shape} -> Output {output.shape} ✅")
    
    print("✅ Model parameter test passed!")

def test_gpu_compatibility():
    """Test GPU compatibility if available."""
    print("\nTesting GPU compatibility...")
    
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} devices")
        device = torch.device("cuda")
        
        # Create model and move to GPU
        model = create_unet_model(
            model_type="unet",
            n_sources=4,
            n_channels=2,
            n_fft=4096
        ).to(device)
        
        # Test input
        test_input = torch.randn(1, 2, 512, 128).to(device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        
        print(f"GPU test: Input {test_input.shape} -> Output {output.shape} ✅")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        
    else:
        print("CUDA not available, skipping GPU test")
    
    print("✅ GPU compatibility test completed!")

if __name__ == "__main__":
    print("🎵 U-Net Music Source Separation - Model Tests")
    print("=" * 50)
    
    try:
        # Run tests
        test_unet_model()
        test_source_separator()
        test_model_parameters()
        test_gpu_compatibility()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        print("The U-Net model is ready for training and inference.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

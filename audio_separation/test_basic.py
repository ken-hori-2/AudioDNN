"""
Simple test script for basic functionality.
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from models import create_unet_model, SourceSeparator
        from utils import load_audio, save_audio, compute_stft, compute_istft
        from data import SpectrogramTransform
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_audio_utils():
    """Test audio utility functions."""
    print("\nTesting audio utilities...")
    
    from utils import compute_stft, compute_istft, peak_normalize
    
    # Create test audio (3 seconds stereo)
    test_audio = torch.randn(2, 44100 * 3)
    print(f"Test audio shape: {test_audio.shape}")
    
    # Test STFT/iSTFT
    stft_result = compute_stft(test_audio, n_fft=4096, hop_length=1024)
    print(f"STFT shape: {stft_result.shape}")
    
    reconstructed = compute_istft(stft_result, n_fft=4096, hop_length=1024, length=test_audio.shape[-1])
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Test reconstruction error
    reconstruction_error = torch.mean((test_audio - reconstructed) ** 2)
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    
    # Test normalization
    normalized = peak_normalize(test_audio, target_peak=0.8)
    peak_after = torch.max(torch.abs(normalized))
    print(f"Peak after normalization: {peak_after:.3f}")
    
    print("✅ Audio utilities test passed!")
    return True

def test_spectrogram_transform():
    """Test spectrogram transform."""
    print("\nTesting spectrogram transform...")
    
    from data import SpectrogramTransform
    
    # Create transform
    transform = SpectrogramTransform(
        n_fft=4096,
        hop_length=1024
    )
    
    # Test audio (3 seconds stereo)
    test_audio = torch.randn(2, 44100 * 3)
    print(f"Input audio shape: {test_audio.shape}")
    
    # Apply transform
    spectrogram = transform(test_audio)
    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Spectrogram dtype: {spectrogram.dtype}")
    
    # Check if it's complex (STFT should be complex)
    assert torch.is_complex(spectrogram), "Spectrogram should be complex (STFT)"
    
    # Check magnitude is positive
    magnitude = torch.abs(spectrogram)
    assert torch.all(magnitude >= 0), "Magnitude should be non-negative"
    
    print("✅ Spectrogram transform test passed!")
    return True

def test_simple_model():
    """Test simple model creation without forward pass."""
    print("\nTesting model creation...")
    
    from models import create_unet_model
    
    try:
        # Create model
        model = create_unet_model(
            model_type="unet",
            n_sources=4,
            n_channels=2,
            n_fft=4096
        )
        
        print(f"Model created successfully: {type(model).__name__}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Model size: {total_params * 4 / (1024**2):.2f} MB")
        
        print("✅ Model creation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_config_loading():
    """Test configuration file loading."""
    print("\nTesting configuration loading...")
    
    import yaml
    
    config_path = Path(__file__).parent / "configs" / "unet_config.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"Config loaded successfully")
        print(f"Model config: {config.get('model', {})}")
        print(f"Training config: {config.get('training', {})}")
        
        print("✅ Configuration loading test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_tensorboard_integration():
    """Test TensorBoard integration."""
    print("\nTesting TensorBoard integration...")
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        import tempfile
        import os
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = SummaryWriter(temp_dir)
            
            # Log some dummy data
            writer.add_scalar('test/loss', 1.0, 0)
            writer.add_scalar('test/accuracy', 0.5, 0)
            
            writer.close()
            
            # Check if log file was created
            log_files = list(Path(temp_dir).glob("**/*"))
            assert len(log_files) > 0, "No TensorBoard log files created"
        
        print("✅ TensorBoard integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ TensorBoard integration failed: {e}")
        return False

if __name__ == "__main__":
    print("🎵 U-Net Music Source Separation - Basic Tests")
    print("=" * 50)
    
    all_passed = True
    
    # Run basic tests
    tests = [
        test_imports,
        test_audio_utils,
        test_spectrogram_transform,
        test_simple_model,
        test_config_loading,
        test_tensorboard_integration,
    ]
    
    for test_func in tests:
        try:
            result = test_func()
            all_passed = all_passed and result
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 All basic tests passed!")
        print("The system is ready for training preparation.")
        print("\nNext steps:")
        print("1. Prepare MUSDB18 dataset")
        print("2. Test training script with small data")
        print("3. Run full training")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        sys.exit(1)

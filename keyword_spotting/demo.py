# Demo script for BC-ResNet keyword spotting
# Shows how to use the trained model for inference

import torch
import torchaudio
import numpy as np
from bcresnet import create_bcresnet
from data_utils import AudioPreprocessor, Padding, LABEL_DICT

def load_model(model_path: str, tau: float, device: torch.device):
    """Load trained BC-ResNet model"""
    model = create_bcresnet(tau=tau, num_classes=len(LABEL_DICT))
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def predict_audio(model, audio_path: str, device: torch.device):
    """Predict keyword from audio file"""
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Ensure single channel
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Pad/truncate to 1 second
    padding = Padding()
    waveform = padding(waveform)
    
    # Add batch dimension
    waveform = waveform.unsqueeze(0).to(device)
    
    # Preprocess (convert to spectrogram)
    preprocessor = AudioPreprocessor(noise_dir=None, device=device)
    labels = torch.zeros(1, dtype=torch.long)  # Dummy labels
    features = preprocessor(waveform, labels, augment=False, is_train=False)
    
    # Predict
    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1)
    
    # Convert to class name
    idx_to_label = {v: k for k, v in LABEL_DICT.items()}
    predicted_label = idx_to_label[predicted_class.item()]
    confidence = probabilities[0, predicted_class].item()
    
    return predicted_label, confidence, probabilities[0].cpu().numpy()

def demo():
    """Run demo with sample predictions"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test model creation
    print("Testing BC-ResNet model creation...")
    for tau in [1.0, 1.5, 2.0]:
        model = create_bcresnet(tau=tau)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"BC-ResNet-{tau}: {total_params:,} parameters")
    
    # Create dummy input to test forward pass
    print("\nTesting forward pass...")
    model = create_bcresnet(tau=1.0)
    model.eval()
    
    # Simulate log-mel spectrogram input: (batch, channels, freq, time)
    dummy_input = torch.randn(2, 1, 40, 101)  # 40 mel bins, 101 time frames
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output classes: {torch.argmax(output, dim=1)}")
    
    # Show class mappings
    print("\nKeyword classes:")
    for label, idx in LABEL_DICT.items():
        print(f"  {idx}: {label}")

if __name__ == "__main__":
    demo()

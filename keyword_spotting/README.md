# BC-ResNet: Keyword Spotting Implementation

PyTorch implementation of "Broadcasted Residual Learning for Efficient Keyword Spotting" by Kim et al. (2021).

## 📋 Overview

This implementation provides BC-ResNet for keyword spotting on the Google Speech Commands dataset, achieving state-of-the-art performance:
- **Speech Commands v1**: 98.0% accuracy
- **Speech Commands v2**: 98.7% accuracy

## 🏗️ Architecture

BC-ResNet introduces a novel broadcasted residual learning approach that:
- Uses 1D temporal convolutions for most residual functions
- Maintains 2D convolutions through broadcasted-residual connections
- Achieves high accuracy with reduced computational cost
- Scales efficiently for different resource constraints

### Key Components

1. **BC-ResBlock**: Core building block with 2D→1D broadcasted residual connections
2. **Sub-Spectral Normalization**: Frequency-aware normalization for spectrograms
3. **SpecAugment**: Data augmentation for robust training
4. **Multi-scale Architecture**: Configurable model sizes (τ = 1, 1.5, 2, 3, 6, 8)

## 🚀 Quick Start

### Installation

```bash
# Install requirements
pip install -r requirements.txt
```

### Demo

```bash
# Test model creation and forward pass
python demo.py
```

### Training

```bash
# Train BC-ResNet-1.0 on Speech Commands v2 (download dataset)
python train.py --tau 1.0 --version 2 --download

# Train BC-ResNet-8 with GPU
python train.py --tau 8 --gpu 0 --version 2

# Train with custom settings
python train.py --tau 2 --epochs 100 --batch_size 128 --lr 0.05
```

### Testing Pre-trained Model

```bash
# Test a saved model
python train.py --test_only --model_path ./checkpoints/best_model.pth --tau 1.0
```

## 📊 Expected Performance

| Model Size (τ) | Parameters | Speech Commands v1 | Speech Commands v2 |
|----------------|------------|-------------------|-------------------|
| BC-ResNet-1.0  | ~100K     | ~96.6%           | ~98.0%           |
| BC-ResNet-1.5  | ~150K     | -                | ~98.3%           |
| BC-ResNet-2.0  | ~200K     | -                | ~98.5%           |
| BC-ResNet-8.0  | ~500K     | -                | ~98.7%           |

## 📁 File Structure

```
audio_sep/
├── bcresnet.py          # BC-ResNet model implementation
├── data_utils.py        # Dataset loading and preprocessing
├── train.py             # Training script
├── demo.py              # Demo and testing utilities
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔧 Key Features

### Model Architecture
- **Broadcasted Residual Learning**: Efficient 1D/2D convolution combination
- **Sub-Spectral Normalization**: Improved frequency representation learning
- **Scalable Design**: Multiple model sizes for different resource constraints

### Data Processing
- **Log-Mel Spectrograms**: 40 mel bins, 16kHz sample rate
- **SpecAugment**: Frequency and time masking for robustness
- **Noise Augmentation**: Background noise injection
- **Time Shifting**: Temporal data augmentation

### Training Features
- **Cosine Annealing**: Learning rate scheduling
- **Mixed Precision**: Optional for faster training
- **Automatic Dataset Download**: Google Speech Commands v1/v2
- **Cross-validation**: Official train/valid/test splits

## 📖 Usage Examples

### Basic Model Creation

```python
from bcresnet import create_bcresnet

# Create different model sizes
model_small = create_bcresnet(tau=1.0)   # ~100K parameters
model_large = create_bcresnet(tau=8.0)   # ~500K parameters
```

### Custom Training

```python
from train import KeywordSpottingTrainer
import argparse

# Setup arguments
args = argparse.Namespace(
    tau=2.0,
    version=2,
    epochs=50,
    batch_size=64,
    lr=0.1,
    data_dir="./data",
    gpu=0
)

# Train model
trainer = KeywordSpottingTrainer(args)
test_accuracy = trainer.train()
```

### Inference

```python
from demo import load_model, predict_audio

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("./checkpoints/best_model.pth", tau=1.0, device=device)

# Predict on audio file
label, confidence, probabilities = predict_audio(model, "test_audio.wav", device)
print(f"Predicted: {label} (confidence: {confidence:.3f})")
```

## 🎯 Keyword Classes

The model recognizes 12 classes from Google Speech Commands:

1. `_silence_` - Background/silence
2. `_unknown_` - Unknown words
3. `down` - Command word
4. `go` - Command word
5. `left` - Command word
6. `no` - Command word
7. `off` - Command word
8. `on` - Command word
9. `right` - Command word
10. `stop` - Command word
11. `up` - Command word
12. `yes` - Command word

## 🔬 Technical Details

### Model Configuration
- **Input**: Log-mel spectrograms (1 × 40 × 101)
- **Sample Rate**: 16 kHz
- **Window**: 30ms (480 samples)
- **Hop Length**: 10ms (160 samples)
- **FFT Size**: 512
- **Mel Bins**: 40

### Training Configuration
- **Optimizer**: SGD with momentum (0.9)
- **Learning Rate**: 0.1 with cosine annealing
- **Weight Decay**: 1e-3
- **Batch Size**: 100
- **Epochs**: 200
- **Warmup**: 5 epochs

## 📚 References

```bibtex
@inproceedings{kim21l_interspeech,
  author={Byeonggeun Kim and Simyung Chang and Jinkyu Lee and Dooyong Sung},
  title={{Broadcasted Residual Learning for Efficient Keyword Spotting}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={4538--4542},
  doi={10.21437/Interspeech.2021-383}
}
```

## 📄 License

This implementation is for research and educational purposes. Please refer to the original paper and official implementation for commercial use guidelines.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for:
- Bug fixes
- Performance improvements
- Additional features
- Documentation improvements

## 💡 Tips

1. **Model Size Selection**: Start with τ=1.0 for quick experiments, use τ=2-8 for best performance
2. **Data Augmentation**: SpecAugment is automatically enabled for τ≥1.5
3. **Memory Usage**: Reduce batch size if encountering CUDA out of memory errors
4. **Training Time**: Expect ~2-4 hours training on modern GPU for 200 epochs
5. **Reproducibility**: Set random seeds for consistent results across runs

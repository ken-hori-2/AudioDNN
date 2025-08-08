# 🎵 AudioDNN: Deep Learning for Audio Processing

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/ken-hori-2/AudioDNN?style=flat-square)](https://github.com/ken-hori-2/AudioDNN)

*A comprehensive deep learning toolkit for audio processing and analysis*

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-projects) • [🎯 Examples](#-examples) • [🤝 Contributing](#-contributing)

</div>

---

## 🌟 Overview

AudioDNN is a cutting-edge collection of deep learning models designed for audio processing tasks. This repository implements state-of-the-art architectures for music source separation and keyword spotting, providing researchers and developers with powerful tools for audio AI applications.

### ✨ Key Features

- 🎼 **Music Source Separation**: U-Net based system for separating vocals, drums, bass, and other instruments
- 🎤 **Keyword Spotting**: BC-ResNet implementation for real-time speech command recognition
- 🔧 **Production Ready**: Optimized models with comprehensive testing and evaluation
- 📚 **Easy to Use**: Simple APIs and detailed documentation
- ⚡ **High Performance**: State-of-the-art accuracy with efficient inference

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/ken-hori-2/AudioDNN.git
cd AudioDNN

# Choose your project
cd audio_separation  # For music source separation
# OR
cd keyword_spotting  # For keyword spotting

# Install dependencies
pip install -r requirements.txt
```

### 🎵 Quick Demo

**Music Source Separation:**
```bash
cd audio_separation
python src/separate.py --input song.wav --output separated/
```

**Keyword Spotting:**
```bash
cd keyword_spotting
python demo.py --audio_file voice_command.wav
```

## 📖 Projects

<table>
<tr>
<td width="50%">

### 🎼 Audio Separation
[![Status](https://img.shields.io/badge/Status-Production%20Ready-green?style=flat-square)](audio_separation/)

**U-Net Music Source Separation**

- 🎯 **Performance**: 6-7 dB SDR
- 🏗️ **Architecture**: U-Net (10.9M parameters)
- 📊 **Outputs**: Vocals, Drums, Bass, Other
- 🎵 **Dataset**: MUSDB18

[📖 Read More →](audio_separation/)

</td>
<td width="50%">

### 🎤 Keyword Spotting
[![Status](https://img.shields.io/badge/Status-Complete-green?style=flat-square)](keyword_spotting/)

**BC-ResNet Speech Recognition**

- 🎯 **Accuracy**: 98.7% on Speech Commands v2
- 🏗️ **Architecture**: BC-ResNet (9K-500K parameters)
- 📊 **Classes**: 12 keywords + silence/unknown
- 🎵 **Dataset**: Google Speech Commands

[📖 Read More →](keyword_spotting/)

</td>
</tr>
</table>

## 🎯 Examples

### Music Source Separation Example

```python
from audio_separation.src.separate import AudioSeparator

# Initialize the separator
separator = AudioSeparator('path/to/checkpoint.pth')

# Separate audio sources
sources = separator.separate('mixed_song.wav')

# Access separated sources
vocals = sources['vocals']
drums = sources['drums']
bass = sources['bass']
other = sources['other']
```

### Keyword Spotting Example

```python
from keyword_spotting.bcresnet import BCResNet

# Load pre-trained model
model = BCResNet.from_pretrained('bcresnet_1.0')

# Recognize keyword
prediction = model.predict('hello_world.wav')
print(f"Detected keyword: {prediction}")
```

## 📊 Performance Benchmarks

| Model | Task | Metric | Score | Parameters |
|-------|------|--------|-------|------------|
| U-Net | Source Separation | SDR | 6.5 dB | 10.9M |
| BC-ResNet-1.0 | Keyword Spotting | Accuracy | 96.8% | 9K |
| BC-ResNet-8.0 | Keyword Spotting | Accuracy | 98.7% | 500K |

## 🛠️ Development

### Project Structure

```
AudioDNN/
├── 🎼 audio_separation/          # Music source separation
│   ├── src/                     # Source code
│   ├── configs/                 # Configuration files
│   ├── tests/                   # Unit tests
│   └── requirements.txt         # Dependencies
├── 🎤 keyword_spotting/          # Keyword spotting
│   ├── bcresnet.py             # Model implementation
│   ├── data_utils.py           # Data utilities
│   ├── train.py                # Training script
│   └── requirements.txt        # Dependencies
└── 📖 README.md                # This file
```

### Running Tests

```bash
# Audio separation tests
cd audio_separation
python -m pytest tests/

# Keyword spotting tests
cd keyword_spotting
python test_pipeline.py
```

## 🔬 Research & Papers

This implementation is based on the following research:

- **U-Net Source Separation**: [Jansson et al., 2017](https://arxiv.org/abs/1708.00065)
- **BC-ResNet**: [Kim et al., 2021](https://arxiv.org/abs/2106.04140)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The PyTorch team for the excellent framework
- MUSDB18 dataset creators
- Google Speech Commands dataset
- The research community for advancing audio AI

## 📞 Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/ken-hori-2/AudioDNN/issues)
- **Discussions**: [Join our community discussions](https://github.com/ken-hori-2/AudioDNN/discussions)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by [ken-hori-2](https://github.com/ken-hori-2)

</div>

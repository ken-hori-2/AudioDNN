<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# BC-ResNet Keyword Spotting Project

This is a PyTorch implementation of BC-ResNet (Broadcasted Residual Learning) for keyword spotting on the Google Speech Commands dataset.

## Code Structure Guidelines

### Architecture Implementation
- **bcresnet.py**: Contains the main BC-ResNet model implementation with SubSpectralNorm and BCResBlock components
- **data_utils.py**: Handles dataset loading, preprocessing, and augmentation utilities
- **train.py**: Main training script with KeywordSpottingTrainer class
- **demo.py**: Demonstration and inference utilities

### Coding Standards
- Follow PyTorch conventions and best practices
- Use type hints for function parameters and return values
- Include comprehensive docstrings for all classes and functions
- Maintain compatibility with PyTorch 1.7.1+ and Python 3.6+

### Model Architecture Notes
- BC-ResNet uses a unique broadcasted residual connection combining 1D and 2D convolutions
- Sub-spectral normalization is applied to frequency dimensions for better spectrogram learning
- Model scaling is controlled by the τ (tau) parameter, affecting channel dimensions

### Data Processing
- Audio samples are 1 second long at 16kHz sampling rate
- Log-mel spectrograms with 40 mel bins are used as input features
- SpecAugment (frequency and time masking) is applied for data augmentation
- Background noise injection and time shifting are used for robustness

### Training Considerations
- Uses SGD optimizer with cosine annealing learning rate schedule
- SpecAugment is enabled automatically for models with τ ≥ 1.5
- Model checkpointing saves the best validation accuracy model

### Performance Targets
- Speech Commands v1: ~98.0% accuracy
- Speech Commands v2: ~98.7% accuracy
- Efficient computation with fewer parameters than conventional CNNs

When suggesting code improvements or modifications:
1. Maintain the paper's architectural design principles
2. Ensure backward compatibility with existing checkpoints
3. Consider computational efficiency and mobile deployment
4. Follow the established data preprocessing pipeline
5. Preserve the multi-scale model design (τ parameter)

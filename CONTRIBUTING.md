# Contributing to AudioDNN

Thank you for your interest in contributing to AudioDNN! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

- Use the [GitHub Issues](https://github.com/ken-hori-2/AudioDNN/issues) to report bugs
- Search existing issues before creating a new one
- Include detailed information about the problem
- Provide steps to reproduce the issue

### Suggesting Features

- Open a [GitHub Discussion](https://github.com/ken-hori-2/AudioDNN/discussions) for feature requests
- Clearly describe the proposed feature and its benefits
- Consider the scope and feasibility of the feature

### Pull Requests

1. **Fork the repository**
   ```bash
   git fork https://github.com/ken-hori-2/AudioDNN.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the coding standards
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   # For audio separation
   cd audio_separation
   python -m pytest tests/
   
   # For keyword spotting
   cd keyword_spotting
   python test_pipeline.py
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add: brief description of your changes"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Provide a clear description of the changes
   - Reference any related issues
   - Include test results

## 📝 Coding Standards

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Maximum line length: 88 characters (Black formatter)

### Code Format

We use the following tools for code formatting:

```bash
# Install formatting tools
pip install black isort flake8

# Format code
black .
isort .
flake8 .
```

### Documentation

- Update README files when adding new features
- Add inline comments for complex logic
- Include examples in docstrings
- Update API documentation

### Testing

- Write unit tests for new functions
- Ensure all tests pass before submitting
- Aim for good test coverage
- Include integration tests for major features

## 🏗️ Project Structure

### Audio Separation Module

```
audio_separation/
├── src/
│   ├── models/          # Model implementations
│   ├── data/           # Data loading and preprocessing
│   ├── utils/          # Utility functions
│   ├── train.py        # Training script
│   ├── evaluate.py     # Evaluation script
│   └── separate.py     # Inference script
├── configs/            # Configuration files
├── tests/             # Unit tests
└── requirements.txt   # Dependencies
```

### Keyword Spotting Module

```
keyword_spotting/
├── bcresnet.py        # BC-ResNet model
├── data_utils.py      # Data utilities
├── train.py          # Training script
├── demo.py           # Demo script
├── test_pipeline.py  # Testing pipeline
└── requirements.txt  # Dependencies
```

## 🐛 Bug Reports

When reporting bugs, please include:

- **Environment**: OS, Python version, PyTorch version
- **Steps to reproduce**: Clear, numbered steps
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full error traceback
- **Minimal example**: Smallest code that reproduces the issue

## 💡 Feature Requests

For feature requests, please provide:

- **Use case**: Why is this feature needed?
- **Description**: Detailed explanation of the feature
- **Examples**: How would it be used?
- **Alternatives**: Any workarounds you've considered?

## 📋 Development Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Git

### Setup Development Environment

```bash
# Clone your fork
git clone https://github.com/your-username/AudioDNN.git
cd AudioDNN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r audio_separation/requirements.txt
pip install -r keyword_spotting/requirements.txt

# Install development tools
pip install black isort flake8 pytest
```

## 🔄 Release Process

1. Update version numbers
2. Update CHANGELOG.md
3. Create release branch
4. Run full test suite
5. Create pull request to main
6. Tag release after merge

## 📞 Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code contributions

## 🙏 Recognition

Contributors will be recognized in:

- README.md contributors section
- Release notes
- CONTRIBUTORS.md file

## 📄 License

By contributing to AudioDNN, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to AudioDNN! 🎵

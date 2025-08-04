"""
Setup script for U-Net Music Source Separation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="unet-music-separation",
    version="1.0.0",
    author="U-Net Source Separation Team",
    author_email="contact@example.com",
    description="PyTorch-based U-Net architecture for music source separation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/unet-music-separation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "musdb": ["musdb", "museval"],
        "dev": ["pytest", "black", "flake8", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "unet-separate=separate:main",
            "unet-train=train:main",
            "unet-evaluate=evaluate:main",
        ],
    },
    keywords="music source separation, deep learning, u-net, pytorch, audio processing",
    project_urls={
        "Bug Reports": "https://github.com/example/unet-music-separation/issues",
        "Source": "https://github.com/example/unet-music-separation",
        "Documentation": "https://github.com/example/unet-music-separation#readme",
    },
)

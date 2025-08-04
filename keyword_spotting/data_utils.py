# Data utilities for Speech Commands dataset processing
# Includes dataset loading, preprocessing, and augmentation

import os
import random
import shutil
import tarfile
from glob import glob
from typing import List, Tuple, Optional, Dict
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import requests
from torch.utils.data import Dataset

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable


# Google Speech Commands dataset configuration
LABEL_DICT = {
    "_silence_": 0,
    "_unknown_": 1,
    "down": 2,
    "go": 3,
    "left": 4,
    "no": 5,
    "off": 6,
    "on": 7,
    "right": 8,
    "stop": 9,
    "up": 10,
    "yes": 11,
}

# Sample count per class for balancing unknown and silence classes
SAMPLE_PER_CLS_V1 = [1854, 258, 257]  # train, valid, test
SAMPLE_PER_CLS_V2 = [3077, 371, 408]  # train, valid, test

SR = 16000  # Sample rate


class LogMelSpectrogram(nn.Module):
    """
    Log-Mel spectrogram transform for audio preprocessing.
    Converts raw audio to log-mel spectrograms used as input features.
    """
    def __init__(
        self, 
        device: torch.device, 
        sample_rate: int = SR, 
        hop_length: int = 160, 
        win_length: int = 480, 
        n_fft: int = 512, 
        n_mels: int = 40
    ):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=win_length,
            n_mels=n_mels,
        )
        self.device = device

    def forward(self, x):
        self.mel = self.mel.to(self.device)
        output = (self.mel(x) + 1e-6).log()
        return output


class Padding:
    """Zero pad audio to have 1 second length"""
    def __init__(self, output_len: int = SR):
        self.output_len = output_len

    def __call__(self, x):
        pad_len = self.output_len - x.shape[-1]
        if pad_len > 0:
            x = torch.cat([x, torch.zeros([x.shape[0], pad_len])], dim=-1)
        elif pad_len < 0:
            # Truncate if longer than 1 second
            x = x[:, :self.output_len]
            warnings.warn("Audio sample longer than 1 second, truncating")
        return x


def spec_augment(
    x, 
    frequency_masking_para: int = 20, 
    time_masking_para: int = 20, 
    frequency_mask_num: int = 2, 
    time_mask_num: int = 2
):
    """
    SpecAugment implementation for speech spectrograms.
    Applies frequency and time masking for data augmentation.
    
    Args:
        x: Input spectrogram of shape (batch, freq, time)
        frequency_masking_para: Maximum frequency mask size
        time_masking_para: Maximum time mask size
        frequency_mask_num: Number of frequency masks
        time_mask_num: Number of time masks
    """
    lenF, lenT = x.shape[1:3]
    
    # Frequency masking
    for _ in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        if f > 0:
            f0 = random.randint(0, max(0, lenF - f))
            x[:, f0:f0 + f, :] = 0
    
    # Time masking
    for _ in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        if t > 0:
            t0 = random.randint(0, max(0, lenT - t))
            x[:, :, t0:t0 + t] = 0
    
    return x


class AudioPreprocessor:
    """
    Audio preprocessing pipeline for keyword spotting.
    Handles noise injection, time shifting, and spectrogram conversion.
    """
    def __init__(
        self,
        noise_dir: Optional[str],
        device: torch.device,
        hop_length: int = 160,
        win_length: int = 480,
        n_fft: int = 512,
        n_mels: int = 40,
        specaug: bool = False,
        sample_rate: int = SR,
        frequency_masking_para: int = 7,
        time_masking_para: int = 20,
        frequency_mask_num: int = 2,
        time_mask_num: int = 2,
    ):
        # Load background noise files
        if noise_dir is None or not os.path.exists(noise_dir):
            self.background_noise = []
            print("Warning: No background noise directory found. Skipping noise augmentation.")
        else:
            noise_files = glob(os.path.join(noise_dir, "*.wav"))
            self.background_noise = []
            for file_name in noise_files:
                try:
                    noise, _ = torchaudio.load(file_name)
                    self.background_noise.append(noise)
                except Exception as e:
                    print(f"Warning: Could not load noise file {file_name}: {e}")
        
        # Feature extraction
        self.feature = LogMelSpectrogram(
            device,
            sample_rate=sample_rate,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            n_mels=n_mels,
        )
        
        self.sample_len = sample_rate
        self.specaug = specaug
        self.device = device
        
        if self.specaug:
            self.frequency_masking_para = frequency_masking_para
            self.time_masking_para = time_masking_para
            self.frequency_mask_num = frequency_mask_num
            self.time_mask_num = time_mask_num
            print(f"SpecAugment enabled - freq: {self.frequency_mask_num} masks with max {self.frequency_masking_para}")
            print(f"SpecAugment enabled - time: {self.time_mask_num} masks with max {self.time_masking_para}")

    def __call__(self, x, labels, augment: bool = True, noise_prob: float = 0.8, is_train: bool = True):
        """
        Apply preprocessing to audio batch.
        
        Args:
            x: Audio tensor of shape (batch, channels, time)
            labels: Label tensor
            augment: Whether to apply data augmentation
            noise_prob: Probability of adding noise
            is_train: Whether in training mode
        """
        assert len(x.shape) == 3
        
        if augment and len(self.background_noise) > 0:
            for idx in range(x.shape[0]):
                # Skip noise for non-silence classes with some probability
                if labels[idx] != 0 and (not is_train or random.random() > noise_prob):
                    continue
                
                # Noise amplitude: lower for keyword classes, higher for silence
                noise_amp = (
                    np.random.uniform(0, 0.1) if labels[idx] != 0 else np.random.uniform(0, 1)
                )
                
                # Select random noise
                noise = random.choice(self.background_noise).to(self.device)
                sample_loc = random.randint(0, max(0, noise.shape[-1] - self.sample_len))
                noise = noise_amp * noise[:, sample_loc:sample_loc + SR]
                
                if is_train:
                    # Time shifting for training
                    x_shift = int(np.random.uniform(-0.1, 0.1) * SR)
                    zero_padding = torch.zeros(1, abs(x_shift)).to(self.device)
                    
                    if x_shift < 0:
                        temp_x = torch.cat([zero_padding, x[idx, :, :x_shift]], dim=-1)
                    else:
                        temp_x = torch.cat([x[idx, :, x_shift:], zero_padding], dim=-1)
                    x[idx] = temp_x + noise
                else:
                    # Just add noise for validation/test
                    x[idx] = x[idx] + noise
                    
                # Clamp to valid range
                x[idx] = torch.clamp(x[idx], -1.0, 1.0)

        # Convert to log-mel spectrogram
        x = self.feature(x)
        
        # Apply SpecAugment if enabled
        if self.specaug and augment:
            for i in range(x.shape[0]):
                x[i] = spec_augment(
                    x[i],
                    self.frequency_masking_para,
                    self.time_masking_para,
                    self.frequency_mask_num,
                    self.time_mask_num,
                )
        
        return x


def scan_audio_files(root_dir: str, version: int) -> Tuple[List[str], List[int]]:
    """
    Scan directory for audio files and return paths with labels.
    
    Args:
        root_dir: Root directory containing class folders
        version: Dataset version (1 or 2) for sample balancing
        
    Returns:
        Tuple of (audio_paths, labels)
    """
    sample_per_cls = SAMPLE_PER_CLS_V1 if version == 1 else SAMPLE_PER_CLS_V2
    audio_paths, labels = [], []
    
    for path, _, files in sorted(os.walk(root_dir, followlinks=True)):
        random.shuffle(files)
        for idx, filename in enumerate(files):
            if not filename.endswith(".wav"):
                continue
                
            # Extract dataset split and class name from path
            path_parts = path.split(os.sep)
            if len(path_parts) < 2:
                continue
                
            dataset, class_name = path_parts[-2:]
            
            # Balance unknown and silence classes
            if class_name in ("_unknown_", "_silence_"):
                if "train" in dataset and idx >= sample_per_cls[0]:
                    break
                if "valid" in dataset and idx >= sample_per_cls[1]:
                    break
                if "test" in dataset and idx >= sample_per_cls[2]:
                    break
                    
            if class_name not in LABEL_DICT:
                print(f"Warning: Unknown class {class_name}, skipping")
                continue
                
            audio_paths.append(os.path.join(path, filename))
            labels.append(LABEL_DICT[class_name])
            
    return audio_paths, labels


class SpeechCommandsDataset(Dataset):
    """
    Google Speech Commands dataset for keyword spotting.
    
    Args:
        root_dir: Root directory containing audio files
        version: Dataset version (1 or 2)
        transform: Optional transform to apply to audio
    """
    def __init__(self, root_dir: str, version: int, transform=None):
        self.transform = transform
        self.data_list, self.labels = scan_audio_files(root_dir, version)
        print(f"Loaded {len(self.data_list)} audio files from {root_dir}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_path = self.data_list[idx]
        
        try:
            sample, _ = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return zero tensor as fallback
            sample = torch.zeros(1, SR)
            
        if self.transform:
            sample = self.transform(sample)
            
        label = self.labels[idx]
        return sample, label


def download_dataset(save_dir: str, url: str):
    """Download and extract dataset from URL"""
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    filename = os.path.basename(url)
    filepath = os.path.join(save_dir, filename)
    
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1048576  # 1MB
    
    with open(filepath, "wb") as f:
        for data in tqdm(response.iter_content(block_size), 
                        total=total_size//block_size, 
                        unit='MB'):
            f.write(data)
    
    print(f"Extracting {filepath}...")
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(save_dir)
    
    # Remove the tar file to save space
    os.remove(filepath)
    print("Dataset download and extraction completed!")


def create_12class_dataset(source_dir: str, target_dir: str):
    """
    Create 12-class dataset from full Speech Commands dataset.
    Groups non-target words into '_unknown_' class.
    """
    if os.path.exists(target_dir):
        print(f"Target directory {target_dir} already exists, skipping creation")
        return
        
    os.makedirs(target_dir)
    os.makedirs(os.path.join(target_dir, "_unknown_"))
    
    # 10 target words + silence + unknown = 12 classes
    target_words = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
    
    for class_dir in glob(os.path.join(source_dir, "*")):
        class_name = os.path.basename(class_dir)
        
        if class_name in target_words:
            # Copy target word directories as-is
            target_class_dir = os.path.join(target_dir, class_name)
            shutil.copytree(class_dir, target_class_dir)
            print(f"Copied {class_dir} to {target_class_dir}")
        elif class_name != "_background_noise_":
            # Move non-target words to _unknown_ directory
            for file_path in glob(os.path.join(class_dir, "*.wav")):
                filename = os.path.basename(file_path)
                target_file = os.path.join(target_dir, "_unknown_", f"{class_name}_{filename}")
                shutil.copy2(file_path, target_file)
            print(f"Moved {class_name} files to _unknown_")


def create_silence_samples(output_dir: str, num_samples: int):
    """Create silent audio samples for the _silence_ class"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i in range(num_samples):
        silence_path = os.path.join(output_dir, f"{i:06d}.wav")
        if not os.path.exists(silence_path):
            zeros = torch.zeros([1, SR])  # 1 second of silence
            torchaudio.save(silence_path, zeros, SR)


def split_dataset(source_dir: str, target_dir: str, valid_list_file: str, test_list_file: str):
    """
    Split dataset into train/valid/test based on official split files.
    
    Args:
        source_dir: Source directory with all audio files
        target_dir: Target directory for split dataset
        valid_list_file: File containing validation split filenames
        test_list_file: File containing test split filenames
    """
    # Read split files
    with open(valid_list_file, "r") as f:
        valid_names = [item.strip() for item in f.readlines()]
    with open(test_list_file, "r") as f:
        test_names = [item.strip() for item in f.readlines()]
    
    # Create target directories
    split_dirs = ["train", "valid", "test"]
    for split_dir in split_dirs:
        os.makedirs(os.path.join(target_dir, split_dir), exist_ok=True)
    
    # Process all audio files
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if not filename.endswith(".wav") or "_background_noise_" in root:
                continue
                
            # Determine class and relative path
            class_name = os.path.basename(root)
            rel_path = os.path.join(class_name, filename)
            
            # Determine split
            if rel_path in valid_names:
                split = "valid"
            elif rel_path in test_names:
                split = "test"
            else:
                split = "train"
            
            # Create class directory in split
            split_class_dir = os.path.join(target_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            
            # Copy file
            src_path = os.path.join(root, filename)
            dst_path = os.path.join(split_class_dir, filename)
            shutil.copy2(src_path, dst_path)
    
    print(f"Dataset split completed: {target_dir}")


def prepare_speech_commands_dataset(data_dir: str, version: int = 2, download: bool = False):
    """
    Prepare Speech Commands dataset for training.
    
    Args:
        data_dir: Directory to store/load dataset
        version: Dataset version (1 or 2)
        download: Whether to download dataset if not present
        
    Returns:
        Dictionary with paths to train/valid/test directories
    """
    version_str = "v0.01" if version == 1 else "v0.02"
    base_dir = os.path.join(data_dir, f"speech_commands_{version_str}")
    
    urls = {
        1: {
            "main": "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz",
            "test": "http://download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz"
        },
        2: {
            "main": "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz", 
            "test": "http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"
        }
    }
    
    # Download if requested and not exists
    if download or not os.path.exists(base_dir):
        if download:
            print(f"Downloading Speech Commands dataset v{version}...")
            download_dataset(base_dir, urls[version]["main"])
            
            # Download test set separately
            test_dir = base_dir.replace("commands_", "commands_test_set_")
            download_dataset(test_dir, urls[version]["test"])
        else:
            raise ValueError(f"Dataset not found at {base_dir}. Set download=True to download.")
    
    # Create split dataset
    split_dir = f"{base_dir}_split"
    if not os.path.exists(split_dir):
        print("Creating dataset splits...")
        split_dataset(
            base_dir,
            split_dir,
            os.path.join(base_dir, "validation_list.txt"),
            os.path.join(base_dir, "testing_list.txt")
        )
    
    # Create 12-class datasets
    sample_per_cls = SAMPLE_PER_CLS_V1 if version == 1 else SAMPLE_PER_CLS_V2
    
    dataset_paths = {}
    for i, split_name in enumerate(["train", "valid", "test"]):
        split_12class_dir = os.path.join(base_dir, f"{split_name}_12class")
        
        if not os.path.exists(split_12class_dir):
            print(f"Creating 12-class {split_name} dataset...")
            create_12class_dataset(os.path.join(split_dir, split_name), split_12class_dir)
            
            # Add silence samples
            silence_dir = os.path.join(split_12class_dir, "_silence_")
            create_silence_samples(silence_dir, sample_per_cls[i])
        
        dataset_paths[split_name] = split_12class_dir
    
    # Add noise directory path
    dataset_paths["noise"] = os.path.join(base_dir, "_background_noise_")
    
    return dataset_paths


if __name__ == "__main__":
    # Test dataset preparation
    data_dir = "./data"
    paths = prepare_speech_commands_dataset(data_dir, version=2, download=False)
    print("Dataset paths:", paths)
    
    # Test dataset loading
    transform = Padding()  # Use our custom Padding class
    
    dataset = SpeechCommandsDataset(paths["train"], version=2, transform=transform)
    print(f"Dataset size: {len(dataset)}")
    
    sample, label = dataset[0]
    print(f"Sample shape: {sample.shape}, Label: {label}")

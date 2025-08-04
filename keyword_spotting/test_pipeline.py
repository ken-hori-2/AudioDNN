# Quick test script for BC-ResNet training pipeline
# Tests the complete training workflow with minimal data and epochs

import os
import torch
from train import KeywordSpottingTrainer
import argparse

def quick_test():
    """Run a quick test of the training pipeline"""
    print("Running BC-ResNet training pipeline test...")
    
    # Setup test arguments
    args = argparse.Namespace(
        data_dir="./data",
        version=2,
        download=False,  # Don't download large dataset for quick test
        tau=1.0,
        epochs=2,  # Very few epochs for quick test
        batch_size=4,  # Small batch size
        lr=0.1,
        num_workers=0,  # Avoid multiprocessing issues
        gpu=-1,  # Use CPU for compatibility
        save_dir="./test_checkpoints",
        test_only=False,
        model_path=None
    )
    
    print(f"Test configuration:")
    print(f"  Model: BC-ResNet-{args.tau}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {'CPU' if args.gpu == -1 else f'GPU {args.gpu}'}")
    
    try:
        # Create dummy data directory structure for testing
        create_dummy_dataset()
        
        # Run training
        trainer = KeywordSpottingTrainer(args)
        trainer.train()
        
        print("\n✅ Training pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Training pipeline test failed: {e}")
        return False

def create_dummy_dataset():
    """Create minimal dummy dataset for testing"""
    print("Creating dummy dataset for testing...")
    
    base_dir = "./data/speech_commands_v0.02"
    
    # Create main directory and split files
    os.makedirs(base_dir, exist_ok=True)
    
    # Create validation and test split files
    with open(os.path.join(base_dir, "validation_list.txt"), "w") as f:
        f.write("yes/dummy_000.wav\n")
        f.write("no/dummy_000.wav\n")
    
    with open(os.path.join(base_dir, "testing_list.txt"), "w") as f:
        f.write("_unknown_/dummy_000.wav\n")
    
    # Create directory structure
    for split in ["train_12class", "valid_12class", "test_12class"]:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Create a few class directories with dummy audio files
        for class_name in ["_silence_", "_unknown_", "yes", "no"]:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Create dummy wav files (silence)
            num_files = 2 if split == "train_12class" else 1
            for i in range(num_files):
                dummy_audio = torch.zeros(1, 16000)  # 1 second of silence
                file_path = os.path.join(class_dir, f"dummy_{i:03d}.wav")
                if not os.path.exists(file_path):
                    import torchaudio
                    torchaudio.save(file_path, dummy_audio, 16000)
    
    # Create noise directory
    noise_dir = os.path.join(base_dir, "_background_noise_")
    os.makedirs(noise_dir, exist_ok=True)
    noise_file = os.path.join(noise_dir, "white_noise.wav")
    if not os.path.exists(noise_file):
        noise_audio = torch.randn(1, 16000) * 0.1  # Low amplitude noise
        torchaudio.save(noise_file, noise_audio, 16000)
    
    print("Dummy dataset created successfully!")

if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n🎉 BC-ResNet implementation is ready for training!")
        print("\nNext steps:")
        print("1. Download Speech Commands dataset: python train.py --download --version 2")
        print("2. Train BC-ResNet model: python train.py --tau 1.0 --version 2")
        print("3. For best results, use: python train.py --tau 8 --version 2 --gpu 0")
    else:
        print("\n⚠️  Please check the error messages above and fix any issues.")

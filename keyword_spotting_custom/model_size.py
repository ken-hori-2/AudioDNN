import torch
# from sound_classifier_pytorch_1 import SoundClassifierNet
from sound_classifier_6class import SoundClassifierNet

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # モデルのインスタンス化（50クラス分類）
    # model = SoundClassifierNet(num_classes=50)
    model = SoundClassifierNet(num_classes=6)
    
    # パラメータ数の計算
    total_params = count_parameters(model)
    
    # モデルサイズの表示
    print(f"モデルの総パラメータ数: {total_params:,}")
    print(f"モデルのサイズ（MB）: {total_params * 4 / (1024 * 1024):.2f}")

if __name__ == "__main__":
    main() 
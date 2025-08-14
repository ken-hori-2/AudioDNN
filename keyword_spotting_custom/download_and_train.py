import os
import subprocess
from sound_classifier_pytorch_acc66 import SoundClassifier

def download_esc50():
    """ESC-50データセットのダウンロード"""
    if not os.path.exists("../ESC-50-master"):
        print("ESC-50データセットをダウンロード中...")
        subprocess.run(["git", "clone", "https://github.com/karolpiczak/ESC-50.git", "ESC-50-master"])
        print("ダウンロード完了")
    else:
        print("ESC-50データセットは既に存在します")

def main():
    # データセットのダウンロード
    download_esc50()
    
    # 分類器のインスタンス化
    classifier = SoundClassifier()
    
    # データセットのパス
    data_dir = "ESC-50-master/audio"
    meta_file = "ESC-50-master/meta/esc50.csv"
    
    # モデルの学習
    print("モデルの学習を開始します...")
    classifier.train(
        data_dir=data_dir,
        meta_file=meta_file,
        epochs=50,
        batch_size=32,
        learning_rate=0.001
    )
    
    # テスト用の音声ファイルで予測
    test_file = "ESC-50-master/audio/1-100032-A-0.wav"
    predicted_label = classifier.predict(test_file)
    print(f"テスト音声の予測結果: {predicted_label}")

if __name__ == "__main__":
    main() 
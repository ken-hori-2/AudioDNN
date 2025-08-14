import matplotlib.pyplot as plt
import numpy as np
import os
from sound_classifier_5class import SoundClassifier, TARGET_CLASSES
import librosa
import librosa.display
import sounddevice as sd
import wave
import torch

import time
import threading
import queue

# 予測間隔
prediction_interval = 1 # 3  # 予測間隔（秒）
input_diration = 2 # 3

def record_audio(duration=input_diration, sample_rate=22050):
    """音声を録音する"""
    print(f"{duration}秒間の録音を開始します...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("録音が完了しました。")
    return recording

def save_audio(recording, filename, sample_rate=22050):
    """録音した音声をWAVファイルとして保存する"""
    # 音声データを16ビット整数に変換
    recording = (recording * 32767).astype(np.int16)
    
    # WAVファイルとして保存
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # モノラル
        wf.setsampwidth(2)  # 16ビット
        wf.setframerate(sample_rate)
        wf.writeframes(recording.tobytes())
    
    print(f"音声を {filename} に保存しました。")

def visualize_prediction(audio_file, classifier, threshold=0.7):
    """予測結果を可視化する"""
    # 予測
    prediction, probability = classifier.predict(audio_file, threshold)
    
    # モデルから確率分布を取得
    feature = classifier.load_audio(audio_file)
    if feature is None:
        return
    
    # 特徴量の正規化と形状の変更
    feature = (feature - np.mean(feature)) / np.std(feature)
    feature = feature.reshape(1, feature.shape[0], feature.shape[1], 1)
    
    # 予測
    classifier.model.eval()
    with torch.no_grad():
        feature = torch.FloatTensor(feature).to(classifier.device)
        outputs = classifier.model(feature)
        probabilities = torch.softmax(outputs, dim=1)
        probs = probabilities.cpu().numpy()[0]
    
    # クラス名を取得
    class_names = classifier.label_encoder.classes_
    
    # 図の作成
    fig = plt.figure(figsize=(10, 6)) # 15, 10))
    
    # 1. 波形とスペクトログラム
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=1)
    y, sr = librosa.load(audio_file, sr=classifier.sample_rate)
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title('音声波形')
    ax1.set_xlabel('時間 (秒)')
    ax1.set_ylabel('振幅')
    
    # 2. メルスペクトログラム
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=1)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax2)
    ax2.set_title('メルスペクトログラム')
    
    # 3. 予測結果の円グラフ
    ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=1, rowspan=1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_names)))
    ax3.pie(probs, labels=class_names, autopct='%1.1f%%', startangle=90, colors=colors)
    ax3.axis('equal')
    ax3.set_title('予測確率 (円グラフ)')
    
    # 4. 予測結果の棒グラフ
    ax4 = plt.subplot2grid((3, 3), (2, 1), colspan=1, rowspan=1)
    bars = ax4.bar(class_names, probs, color=colors)
    ax4.set_ylim(0, 1)
    ax4.set_title('予測確率 (棒グラフ)')
    ax4.set_xticklabels(class_names, rotation=45, ha='right')
    
    # 5. 予測結果のテキスト表示
    ax5 = plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=3)
    ax5.axis('off')
    result_text = f"予測結果: {prediction}\n確率: {probability:.2%}"
    if prediction == "Unknown":
        result_text += "\n\n(閾値 {:.0%} 未満のため、Unknownと判定)"
    ax5.text(0.5, 0.5, result_text, fontsize=14, ha='center', va='center', 
             bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=1'))
    
    # plt.tight_layout()
    # plt.savefig('prediction_result.png', dpi=300)
    # plt.show()
    
    # グラフを更新
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=300)
    # グラフを表示
    plt.show(block=False)
    plt.pause(prediction_interval)
    # time.sleep(prediction_interval)
    plt.close(fig)
    
    return prediction, probability

# def main():
#     # 分類器のインスタンス化
#     classifier = SoundClassifier()
    
#     # 保存されたモデルの読み込み
#     classifier.load_model('best_model_5class_90.pth')
#     # classifier.load_model('best_model_5class_original.pth')
    
#     while True:
#         print("\n1: 音声ファイルを予測")
#         print("2: 音声を録音して予測")
#         print("3: 終了")
#         choice = input("選択してください (1/2/3): ")
        
#         if choice == "1":
#             # 音声ファイルのパスを入力
#             file_path = input("音声ファイルのパスを入力してください: ")
#             if os.path.exists(file_path):
#                 visualize_prediction(file_path, classifier)
#             else:
#                 print("ファイルが存在しません。")
        
#         elif choice == "2":
#             # 録音
#             recording = record_audio()
            
#             # 一時的なWAVファイルとして保存
#             temp_file = "temp_recording.wav"
#             save_audio(recording, temp_file)
            
#             # 予測と可視化
#             visualize_prediction(temp_file, classifier)
            
#             # 一時ファイルの削除
#             os.remove(temp_file)
        
#         elif choice == "3":
#             print("プログラムを終了します。")
#             break
        
#         else:
#             print("無効な選択です。")



def main():
    # 分類器のインスタンス化
    classifier = SoundClassifier()
    
    # 保存されたモデルの読み込み
    classifier.load_model('best_model_5class_90.pth')
    
    while True:
        print("音声分類予測可視化プログラムを開始します...")
        print("自動的にマイクから音声を録音し、予測を行います。")
        print("終了するには 'q' を入力してください。")

        # 録音
        recording = record_audio()
        
        # 一時的なWAVファイルとして保存
        temp_file = "temp_recording.wav"
        save_audio(recording, temp_file)
        
        # 予測と可視化
        visualize_prediction(temp_file, classifier)
        
        # 一時ファイルの削除
        os.remove(temp_file)

if __name__ == "__main__":
    main() 
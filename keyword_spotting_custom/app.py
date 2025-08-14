import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sounddevice as sd
import wave
import torch
import os
import time
import threading
import queue
from io import BytesIO
import base64
from sound_classifier_5class import SoundClassifier, TARGET_CLASSES

# ページ設定
st.set_page_config(
    page_title="Real-time Audio Classification System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        text-shadow: none;
    }
    .status-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .recording {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        border: none;
        color: white;
        animation: pulse 2s infinite;
    }
    .stopped {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        border: none;
        color: white;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        animation: slideIn 0.5s ease-out;
    }
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1.5rem 0;
    }
    .metric {
        text-align: center;
        padding: 1.5rem;
        background: rgba(255,255,255,0.15);
        border-radius: 15px;
        margin: 0 0.5rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .metric h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .metric p {
        margin: 0;
        font-size: 2rem;
        font-weight: bold;
    }
    .stats-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
    }
    .live-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: #ff4757;
        border-radius: 50%;
        margin-right: 8px;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.3; }
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class AudioRecorder:
    def __init__(self, sample_rate=22050, duration=2):
        self.sample_rate = sample_rate
        self.duration = duration
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
    def record_audio(self):
        """音声を録音する"""
        recording = sd.rec(int(self.duration * self.sample_rate), 
                          samplerate=self.sample_rate, channels=1)
        sd.wait()
        return recording
    
    def save_audio(self, recording, filename):
        """録音した音声をWAVファイルとして保存する"""
        recording = (recording * 32767).astype(np.int16)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(recording.tobytes())

def create_visualization(audio_file, classifier, prediction, probability):
    """予測結果の可視化を作成"""
    # 音声データの読み込み
    y, sr = librosa.load(audio_file, sr=classifier.sample_rate)
    
    # 特徴量の取得
    feature = classifier.load_audio(audio_file)
    if feature is None:
        return None
    
    # 特徴量の正規化と形状の変更
    feature = (feature - np.mean(feature)) / np.std(feature)
    feature = feature.reshape(1, feature.shape[0], feature.shape[1], 1)
    
    # 予測確率の取得
    classifier.model.eval()
    with torch.no_grad():
        feature = torch.FloatTensor(feature).to(classifier.device)
        outputs = classifier.model(feature)
        probabilities = torch.softmax(outputs, dim=1)
        probs = probabilities.cpu().numpy()[0]
    
    # クラス名を取得
    class_names = classifier.label_encoder.classes_
    
    # 図の作成
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Real-time Audio Analysis Results', fontsize=18, fontweight='bold', color='#2c3e50')
    
    # 1. 波形
    librosa.display.waveshow(y, sr=sr, ax=ax1, color='#3498db')
    ax1.set_title('Audio Waveform', fontweight='bold', color='#2c3e50')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # 2. メルスペクトログラム
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax2, cmap='viridis')
    ax2.set_title('Mel Spectrogram', fontweight='bold', color='#2c3e50')
    fig.colorbar(img, ax=ax2, format='%+2.0f dB')
    ax2.set_facecolor('#f8f9fa')
    
    # 3. 予測確率の棒グラフ
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_names)))
    bars = ax3.bar(class_names, probs, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    ax3.set_ylim(0, 1)
    ax3.set_title('Class Prediction Probabilities', fontweight='bold', color='#2c3e50')
    ax3.set_xticklabels(class_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#f8f9fa')
    
    # 確率値を棒グラフの上に表示
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.1%}', ha='center', va='bottom', fontweight='bold', color='#2c3e50')
    
    # 4. 予測結果の円グラフ
    wedges, texts, autotexts = ax4.pie(probs, labels=class_names, autopct='%1.1f%%', 
                                       startangle=90, colors=colors, textprops={'fontsize': 10})
    ax4.set_title('Prediction Distribution', fontweight='bold', color='#2c3e50')
    
    # 円グラフのテキストを白色に
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    return fig

def main():
    # ヘッダー
    st.markdown('<h1 class="main-header">🎵 Real-time Audio Classification System</h1>', unsafe_allow_html=True)
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # 録音設定
        st.subheader("Recording Settings")
        duration = st.slider("Recording Duration (seconds)", 1.0, 5.0, 2.0, 0.5)
        sample_rate = st.selectbox("Sample Rate", [16000, 22050, 44100], index=1)
        detection_count = st.slider("Number of Detections", 1, 20, 5, 1)
        
        # 分類設定
        st.subheader("Classification Settings")
        threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.7, 0.05)
        
        # モデル情報
        st.subheader("📊 Model Information")
        st.write("**Target Classes:**")
        class_descriptions = {
            'dog': '🐕 Dog Barking',
            'cat': '🐱 Cat Meowing', 
            'chirping_birds': '🐦 Bird Chirping',
            'finger_snap': '👆 Finger Snap',
            'human_voice': '👤 Human Voice'
        }
        for i, class_name in enumerate(TARGET_CLASSES, 1):
            st.write(f"{i}. {class_descriptions.get(class_name, class_name)}")
    
    # メインコンテンツ
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎤 Recording Control")
        
        # 録音状態の表示
        if 'recording_status' not in st.session_state:
            st.session_state.recording_status = False
        
        status_class = "recording" if st.session_state.recording_status else "stopped"
        status_text = "🔴 LIVE Recording..." if st.session_state.recording_status else "⏹️ Stopped"
        
        st.markdown(f"""
        <div class="status-box {status_class}">
            {status_text}
        </div>
        """, unsafe_allow_html=True)
        
        # 制御ボタン
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🎙️ Start Recording", type="primary", use_container_width=True):
                st.session_state.recording_status = True
                st.rerun()
        
        with col_btn2:
            if st.button("⏹️ Stop Recording", use_container_width=True):
                st.session_state.recording_status = False
                st.rerun()
    
    with col2:
        st.subheader("📈 Statistics")
        
        # 統計情報の表示
        if 'total_predictions' not in st.session_state:
            st.session_state.total_predictions = 0
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        if 'current_result' not in st.session_state:
            st.session_state.current_result = None
        
        st.markdown("""
        <div class="stats-card">
            <h4>Total Predictions</h4>
            <p style="font-size: 2rem; margin: 0;">{}</p>
        </div>
        """.format(st.session_state.total_predictions), unsafe_allow_html=True)
        
        if st.session_state.prediction_history:
            recent_accuracy = sum(1 for p in st.session_state.prediction_history[-10:] 
                                if p != "Unknown") / min(10, len(st.session_state.prediction_history))
            st.markdown("""
            <div class="stats-card">
                <h4>Recent Accuracy (Last 10)</h4>
                <p style="font-size: 2rem; margin: 0;">{:.1%}</p>
            </div>
            """.format(recent_accuracy), unsafe_allow_html=True)
    
    # 分類器の初期化
    if 'classifier' not in st.session_state:
        with st.spinner("Loading model..."):
            st.session_state.classifier = SoundClassifier()
            st.session_state.classifier.load_model('best_model_5class_90.pth')
    
    # 録音器の初期化
    if 'recorder' not in st.session_state:
        st.session_state.recorder = AudioRecorder(sample_rate, duration)
    else:
        # 設定が変更された場合は録音器を更新
        if (st.session_state.recorder.sample_rate != sample_rate or 
            st.session_state.recorder.duration != duration):
            st.session_state.recorder = AudioRecorder(sample_rate, duration)
    
    # リアルタイム録音と分類
    if st.session_state.recording_status:
        st.subheader("🔍 Live Analysis")
        
        # プログレスバー
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 結果表示エリア
        result_container = st.container()
        
        # 録音と分析のループ
        for i in range(detection_count):  # 10回の録音サイクル
            if not st.session_state.recording_status:
                break
                
            status_text.text(f"🎙️ Recording... ({i+1}/{detection_count})")
            progress_bar.progress((i + 1) / detection_count)
            
            # 録音
            recording = st.session_state.recorder.record_audio()
            
            # 一時ファイルに保存
            temp_file = f"temp_recording_{i}.wav"
            st.session_state.recorder.save_audio(recording, temp_file)
            
            # 分類
            prediction, probability = st.session_state.classifier.predict(temp_file, threshold)
            
            # 履歴に追加
            st.session_state.total_predictions += 1
            st.session_state.prediction_history.append(prediction)
            
            # 結果表示
            with result_container:
                if prediction != "Unknown":
                    # クラス名の日本語表示
                    class_display_names = {
                        'dog': '🐕 Dog Barking',
                        'cat': '🐱 Cat Meowing',
                        'chirping_birds': '🐦 Bird Chirping', 
                        'finger_snap': '👆 Finger Snap',
                        'human_voice': '👤 Human Voice'
                    }
                    display_name = class_display_names.get(prediction, prediction)
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>🎯 Detection Result</h3>
                        <div class="metric-container">
                            <div class="metric">
                                <h4>Classification</h4>
                                <p>{display_name}</p>
                            </div>
                            <div class="metric">
                                <h4>Confidence</h4>
                                <p>{probability:.1%}</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 可視化
                    fig = create_visualization(temp_file, st.session_state.classifier, prediction, probability)
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)
                else:
                    st.warning(f"🔍 No audio detected (Confidence: {probability:.1%})")
            
            # 一時ファイルの削除
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            # 3秒間待機
            time.sleep(3)
        
        progress_bar.empty()
        status_text.empty()
    
    # 履歴表示
    if st.session_state.prediction_history:
        st.subheader("📊 Prediction History")
        
        # 履歴の可視化
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # クラス別の出現回数をカウント
        class_counts = {}
        for pred in st.session_state.prediction_history:
            if pred != "Unknown":
                class_counts[pred] = class_counts.get(pred, 0) + 1
        
        if class_counts:
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
            
            bars = ax.bar(classes, counts, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
            ax.set_title('Detected Audio Distribution', fontweight='bold', color='#2c3e50', fontsize=16)
            ax.set_ylabel('Detection Count', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')
            
            # 数値を棒グラフの上に表示
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       str(count), ha='center', va='bottom', fontweight='bold', color='#2c3e50')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    # フッター
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🎵 Real-time Audio Classification System | Powered by Streamlit & PyTorch
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
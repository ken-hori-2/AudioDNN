import streamlit as st
import os
import torch
import yt_dlp
from demucs.pretrained import get_model
from demucs.apply import apply_model
import soundfile as sf
import numpy as np
import librosa
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import tempfile
from pathlib import Path
import base64

# ページ設定
st.set_page_config(
    page_title="AudioSep - AI音源分離ツール",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# デフォルトのURL（サカナクション）
DEFAULT_URL = "https://www.youtube.com/shorts/GtiIMqCy_3k"

# カスタムCSS - モダンで美しいデザイン
st.markdown("""
<style>
    /* 全体のフォント設定 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* メインヘッダー */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    
    /* サブヘッダー */
    .sub-header {
        font-size: 1.25rem;
        color: #667eea;
        margin-bottom: 1.5rem;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    /* サイドバーヘッダー */
    .sidebar-header {
        font-size: 1.1rem;
        color: #667eea;
        margin-bottom: 1rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* カードデザイン */
    .feature-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        border-color: #667eea;
    }
    
    .feature-card h3 {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.75rem;
    }
    
    .feature-card p {
        color: #64748b;
        line-height: 1.6;
        margin: 0;
    }
    
    /* 使用方法ボックス */
    .usage-box {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0;
    }
    
    .usage-box h3 {
        color: #1e293b;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .usage-box ol {
        color: #475569;
        line-height: 1.8;
        padding-left: 1.5rem;
    }
    
    .usage-box li {
        margin-bottom: 0.5rem;
    }
    
    /* ステータスボックス */
    .status-box {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: none;
        font-weight: 500;
    }
    
    .success-box {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        color: #166534;
        border-left: 4px solid #22c55e;
    }
    
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: #1e40af;
        border-left: 4px solid #3b82f6;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        border-left: 4px solid #f59e0b;
    }
    
    /* ボタンスタイル */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(102, 126, 234, 0.4);
    }
    
    /* 入力フィールド */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* チェックボックス */
    .stCheckbox > label {
        font-weight: 500;
        color: #374151;
    }
    
    /* プログレスバー */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
    }
    
    /* タブ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }
    
    /* 音声プレイヤー */
    .stAudio {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* ダウンロードリンク */
    .download-link {
        display: inline-block;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
        margin-top: 0.5rem;
    }
    
    .download-link:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.3);
    }
    
    /* アニメーション */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* レスポンシブデザイン */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .feature-card {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def download_youtube_audio(url, output_path="downloads"):
    """YouTubeから音声をダウンロードする関数"""
    os.makedirs(output_path, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_file = os.path.join(output_path, f"{info['title']}.mp3")
    
    return audio_file, info['title']

def separate_audio(audio_path, output_dir="separated", use_cuda=False):
    """音声ファイルからボーカルと楽器を分離する関数"""
    os.makedirs(output_dir, exist_ok=True)
    
    # モデルを読み込む
    model = get_model('htdemucs')
    
    if use_cuda and torch.cuda.is_available():
        model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    
    # 音声を読み込む
    wav, sr = librosa.load(audio_path, sr=model.samplerate)
    
    # ステレオに変換
    if len(wav.shape) == 1:
        wav = np.stack([wav, wav])
    elif wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)
    
    wav_tensor = torch.from_numpy(wav).float()
    
    # 音声を分離
    sources = apply_model(model, wav_tensor.unsqueeze(0), device=device, split=True)[0]
    
    # 分離された音声を保存
    sources_dict = {}
    for source, name in zip(sources, model.sources):
        source_path = os.path.join(output_dir, f"{os.path.basename(audio_path).split('.')[0]}_{name}.wav")
        sf.write(source_path, source.T, model.samplerate)
        sources_dict[name] = source_path
    
    return sources_dict

def create_waveform_plot(audio_path, title="波形", color="#667eea"):
    """音声ファイルの波形をプロット"""
    try:
        # 音声を読み込み
        y, sr = librosa.load(audio_path, sr=None)
        
        # 時間軸を作成
        time = np.linspace(0, len(y) / sr, len(y))
        
        # プロット作成
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time,
            y=y,
            mode='lines',
            name=title,
            line=dict(color=color, width=1.5),
            fill='tonexty',
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)'
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=16, color='#1e293b', family='Inter')
            ),
            xaxis_title="時間 (秒)",
            yaxis_title="振幅",
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'),
            xaxis=dict(
                gridcolor='#e2e8f0',
                zeroline=False,
                showline=True,
                linecolor='#cbd5e1'
            ),
            yaxis=dict(
                gridcolor='#e2e8f0',
                zeroline=False,
                showline=True,
                linecolor='#cbd5e1'
            )
        )
        
        return fig
    except Exception as e:
        st.error(f"波形の作成中にエラーが発生しました: {str(e)}")
        return None

def create_spectrogram_plot(audio_path, title="スペクトログラム"):
    """音声ファイルのスペクトログラムをプロット"""
    try:
        # 音声を読み込み
        y, sr = librosa.load(audio_path, sr=None)
        
        # スペクトログラムを計算
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        # プロット作成
        fig = go.Figure(data=go.Heatmap(
            z=D,
            colorscale='Viridis',
            xaxis='x',
            yaxis='y'
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=16, color='#1e293b', family='Inter')
            ),
            xaxis_title="時間 (フレーム)",
            yaxis_title="周波数 (Hz)",
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter')
        )
        
        return fig
    except Exception as e:
        st.error(f"スペクトログラムの作成中にエラーが発生しました: {str(e)}")
        return None

def get_audio_download_link(file_path, link_text="ダウンロード"):
    """音声ファイルのダウンロードリンクを生成"""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    ext = file_path.split(".")[-1]
    filename = os.path.basename(file_path)
    href = f'<a href="data:audio/{ext};base64,{b64}" download="{filename}" class="download-link">{link_text}</a>'
    return href

def main():
    # ヘッダー
    st.markdown('<h1 class="main-header animate-fade-in">🎵 AudioSep</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.3rem; color: #64748b; margin-bottom: 3rem; font-weight: 400;">AI音源分離ツール</p>', unsafe_allow_html=True)
    
    # サイドバー
    with st.sidebar:
        st.markdown('<h3 class="sidebar-header">⚙️ 設定</h3>', unsafe_allow_html=True)
        
        # YouTube URL入力
        url = st.text_input(
            "YouTube URL",
            value=DEFAULT_URL,
            help="分離したいYouTube動画のURLを入力してください"
        )
        
        # GPU使用設定
        use_cuda = st.checkbox(
            "GPUを使用 (CUDA)",
            value=torch.cuda.is_available(),
            help="GPUが利用可能な場合、処理速度が向上します"
        )
        
        # 処理開始ボタン
        process_button = st.button(
            "🎵 音源分離を開始",
            type="primary",
            use_container_width=True
        )
        
        # 情報表示
        st.markdown("---")
        st.markdown("### 📊 処理情報")
        if torch.cuda.is_available():
            st.markdown('<div class="status-box success-box">✅ GPU (CUDA) が利用可能です</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box warning-box">⚠️ GPU (CUDA) が利用できません。CPUで処理します。</div>', unsafe_allow_html=True)
    
    # メインコンテンツ
    if process_button:
        if not url:
            st.error("YouTube URLを入力してください。")
            return
        
        # プログレスバー
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # ステップ1: YouTube音声ダウンロード
            status_text.text("📥 YouTube音声をダウンロード中...")
            progress_bar.progress(20)
            
            audio_path, title = download_youtube_audio(url)
            
            status_text.text("✅ ダウンロード完了")
            progress_bar.progress(40)
            
            # ダウンロード情報表示
            with st.expander("📥 ダウンロード情報", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**タイトル:** {title}")
                with col2:
                    st.info(f"**ファイル:** {os.path.basename(audio_path)}")
                
                # 元音声の波形表示
                st.markdown("### 🎵 元音声の波形")
                waveform_fig = create_waveform_plot(audio_path, "元音声の波形")
                if waveform_fig:
                    st.plotly_chart(waveform_fig, use_container_width=True)
                
                # 元音声の再生
                st.audio(audio_path, format="audio/mp3")
            
            # ステップ2: 音源分離
            status_text.text("🎛️ 音源分離中... (時間がかかる場合があります)")
            progress_bar.progress(60)
            
            separated_files = separate_audio(audio_path, use_cuda=use_cuda)
            
            status_text.text("✅ 音源分離完了")
            progress_bar.progress(100)
            
            # 分離結果表示
            st.markdown('<h2 class="sub-header">🎵 分離結果</h2>', unsafe_allow_html=True)
            
            # 4つの音源を表示
            sources = ['vocals', 'drums', 'bass', 'other']
            source_names = ['ボーカル', 'ドラム', 'ベース', 'その他']
            source_colors = ['#ef4444', '#10b981', '#3b82f6', '#8b5cf6']
            
            # タブで分離結果を表示
            tabs = st.tabs(source_names)
            
            for i, (source, name, color) in enumerate(zip(sources, source_names, source_colors)):
                with tabs[i]:
                    if source in separated_files:
                        file_path = separated_files[source]
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # 波形表示
                            waveform_fig = create_waveform_plot(file_path, f"{name}の波形", color)
                            if waveform_fig:
                                st.plotly_chart(waveform_fig, use_container_width=True)
                            
                            # スペクトログラム表示
                            spec_fig = create_spectrogram_plot(file_path, f"{name}のスペクトログラム")
                            if spec_fig:
                                st.plotly_chart(spec_fig, use_container_width=True)
                        
                        with col2:
                            # 音声再生
                            st.markdown(f"### 🔊 {name}")
                            st.audio(file_path, format="audio/wav")
                            
                            # ダウンロードリンク
                            st.markdown(get_audio_download_link(file_path, f"{name}をダウンロード"), unsafe_allow_html=True)
            
            # 完了メッセージ
            st.success("🎉 音源分離が完了しました！")
            
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            st.exception(e)
    
    else:
        # 初期画面
        st.markdown("""
        <div class="usage-box animate-fade-in">
            <h3>🚀 使い方</h3>
            <ol>
                <li>左側のサイドバーでYouTube URLを入力（デフォルトでサカナクションの楽曲が設定されています）</li>
                <li>必要に応じてGPU使用設定を調整</li>
                <li>「音源分離を開始」ボタンをクリック</li>
                <li>処理完了後、分離された音源を確認・ダウンロード</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # 機能説明
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card animate-fade-in">
                <h3>🎬 YouTube音声ダウンロード</h3>
                <p>任意のYouTube動画から高品質な音声を自動取得</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card animate-fade-in">
                <h3>🎤 AI音源分離</h3>
                <p>Demucs AIモデルでボーカル・ドラム・ベース・その他に分離</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card animate-fade-in">
                <h3>📊 可視化</h3>
                <p>波形・スペクトログラムで音声を視覚的に確認</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
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

# Page configuration
st.set_page_config(
    page_title="AudioSep - AI Audio Separation Tool",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default URL (Sakanaction)
DEFAULT_URL = "https://www.youtube.com/shorts/GtiIMqCy_3k"

# Custom CSS - Modern and beautiful design
st.markdown("""
<style>
    /* Global font settings */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main header */
    .main-header {
        font-size: 4.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 20%, #f093fb 40%, #f5576c 60%, #4facfe 80%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.04em;
        line-height: 1;
        text-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    
    .main-subtitle {
        font-size: 1.6rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 3.5rem;
        font-weight: 400;
        letter-spacing: 0.03em;
    }
    
    /* Sub header */
    .sub-header {
        font-size: 1.8rem;
        color: #1e293b;
        margin-bottom: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    /* Sidebar header */
    .sidebar-header {
        font-size: 1.4rem;
        color: #1e293b;
        margin-bottom: 2rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 1rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #e2e8f0;
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 2px solid #e2e8f0;
        border-radius: 24px;
        padding: 3rem 2.5rem;
        text-align: center;
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.1);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c);
        transform: scaleX(0);
        transition: transform 0.4s ease;
    }
    
    .feature-card:hover::before {
        transform: scaleX(1);
    }
    
    .feature-card:hover {
        transform: translateY(-12px) scale(1.03);
        box-shadow: 0 32px 64px -12px rgba(0, 0, 0, 0.3);
        border-color: #667eea;
    }
    
    .feature-card h3 {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1.5rem;
        letter-spacing: -0.02em;
    }
    
    .feature-card p {
        color: #64748b;
        line-height: 1.8;
        margin: 0;
        font-size: 1.1rem;
    }
    
    /* Usage box */
    .usage-box {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 2px solid #cbd5e1;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 3rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .usage-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.15) 50%, transparent 70%);
        transform: translateX(-100%);
        transition: transform 0.8s ease;
    }
    
    .usage-box:hover::before {
        transform: translateX(100%);
    }
    
    .usage-box h3 {
        color: #1e293b;
        font-weight: 700;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        font-size: 1.5rem;
    }
    
    .usage-box ol {
        color: #475569;
        line-height: 2;
        padding-left: 2rem;
        font-size: 1.15rem;
    }
    
    .usage-box li {
        margin-bottom: 1rem;
        position: relative;
    }
    
    .usage-box li::marker {
        font-weight: 700;
        color: #667eea;
        font-size: 1.2rem;
    }
    
    /* Status boxes */
    .status-box {
        padding: 1.5rem 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        border: none;
        font-weight: 600;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        box-shadow: 0 6px 12px -2px rgba(0, 0, 0, 0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        color: #166534;
        border-left: 6px solid #22c55e;
    }
    
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: #1e40af;
        border-left: 6px solid #3b82f6;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        border-left: 6px solid #f59e0b;
    }
    
    /* Button styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 20px;
        padding: 1.25rem 2.5rem;
        font-weight: 700;
        font-size: 1.2rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 12px 24px -6px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.6s ease;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.03);
        box-shadow: 0 24px 32px -8px rgba(102, 126, 234, 0.5);
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 16px;
        border: 2px solid #e2e8f0;
        transition: all 0.4s ease;
        padding: 1rem 1.25rem;
        font-size: 1.1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 6px rgba(102, 126, 234, 0.1);
        transform: scale(1.02);
    }
    
    /* Checkbox */
    .stCheckbox > label {
        font-weight: 600;
        color: #374151;
        font-size: 1.1rem;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        border-radius: 16px;
        height: 12px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        background: #f8fafc;
        padding: 12px;
        border-radius: 16px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* Audio player */
    .stAudio {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 6px 12px -2px rgba(0, 0, 0, 0.1);
    }
    
    /* Download link */
    .download-link {
        display: inline-block;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 16px;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.4s ease;
        margin-top: 1.5rem;
        box-shadow: 0 6px 12px -2px rgba(16, 185, 129, 0.3);
    }
    
    .download-link:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px -4px rgba(16, 185, 129, 0.4);
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(40px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-40px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.08);
        }
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
    }
    
    .animate-fade-in {
        animation: fadeInUp 1s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .animate-slide-in {
        animation: slideInLeft 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .animate-pulse {
        animation: pulse 2.5s infinite;
    }
    
    .animate-float {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 3rem;
        }
        
        .feature-card {
            padding: 2rem;
        }
        
        .usage-box {
            padding: 2rem;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8, #6b46c1, #e879f9);
    }
</style>
""", unsafe_allow_html=True)

def download_youtube_audio(url, output_path="downloads"):
    """Download audio from YouTube"""
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
    """Separate vocals and instruments from audio file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = get_model('htdemucs')
    
    if use_cuda and torch.cuda.is_available():
        model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Load audio
    wav, sr = librosa.load(audio_path, sr=model.samplerate)
    
    # Convert to stereo
    if len(wav.shape) == 1:
        wav = np.stack([wav, wav])
    elif wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)
    
    wav_tensor = torch.from_numpy(wav).float()
    
    # Separate audio
    sources = apply_model(model, wav_tensor.unsqueeze(0), device=device, split=True)[0]
    
    # Save separated audio
    sources_dict = {}
    for source, name in zip(sources, model.sources):
        source_path = os.path.join(output_dir, f"{os.path.basename(audio_path).split('.')[0]}_{name}.wav")
        sf.write(source_path, source.T, model.samplerate)
        sources_dict[name] = source_path
    
    return sources_dict

def create_waveform_plot(audio_path, title="Waveform", color="#667eea"):
    """Create waveform plot for audio file"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Create time axis
        time = np.linspace(0, len(y) / sr, len(y))
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time,
            y=y,
            mode='lines',
            name=title,
            line=dict(color=color, width=2.5),
            fill='tonexty',
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)'
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, color='#1e293b', family='Inter')
            ),
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            height=400,
            margin=dict(l=70, r=70, t=70, b=70),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'),
            xaxis=dict(
                gridcolor='#e2e8f0',
                zeroline=False,
                showline=True,
                linecolor='#cbd5e1',
                linewidth=1.5
            ),
            yaxis=dict(
                gridcolor='#e2e8f0',
                zeroline=False,
                showline=True,
                linecolor='#cbd5e1',
                linewidth=1.5
            ),
            showlegend=False
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating waveform: {str(e)}")
        return None

def create_spectrogram_plot(audio_path, title="Spectrogram"):
    """Create spectrogram plot for audio file"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Calculate spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        # Create plot
        fig = go.Figure(data=go.Heatmap(
            z=D,
            colorscale='Viridis',
            xaxis='x',
            yaxis='y'
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, color='#1e293b', family='Inter')
            ),
            xaxis_title="Time (frames)",
            yaxis_title="Frequency (Hz)",
            height=400,
            margin=dict(l=70, r=70, t=70, b=70),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter')
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating spectrogram: {str(e)}")
        return None

def get_audio_download_link(file_path, link_text="Download"):
    """Generate download link for audio file"""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    ext = file_path.split(".")[-1]
    filename = os.path.basename(file_path)
    href = f'<a href="data:audio/{ext};base64,{b64}" download="{filename}" class="download-link">{link_text}</a>'
    return href

def main():
    # Header
    st.markdown('<h1 class="main-header animate-fade-in">🎵 AudioSep</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle animate-fade-in">AI-Powered Audio Separation Tool</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h3 class="sidebar-header">⚙️ Settings</h3>', unsafe_allow_html=True)
        
        # YouTube URL input
        url = st.text_input(
            "YouTube URL",
            value=DEFAULT_URL,
            help="Enter YouTube URL to separate audio"
        )
        
        # GPU usage setting
        use_cuda = st.checkbox(
            "Use GPU (CUDA)",
            value=torch.cuda.is_available(),
            help="Enable GPU acceleration for faster processing"
        )
        
        # Process start button
        process_button = st.button(
            "🎵 Start Audio Separation",
            type="primary",
            use_container_width=True
        )
        
        # Information display
        st.markdown("---")
        st.markdown("### 📊 System Information")
        if torch.cuda.is_available():
            st.markdown('<div class="status-box success-box">✅ GPU (CUDA) Available</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box warning-box">⚠️ GPU (CUDA) Not Available</div>', unsafe_allow_html=True)
    
    # Main content
    if process_button:
        if not url:
            st.error("Please enter a YouTube URL.")
            return
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: YouTube audio download
            status_text.text("📥 Downloading YouTube audio...")
            progress_bar.progress(20)
            
            audio_path, title = download_youtube_audio(url)
            
            status_text.text("✅ Download Complete")
            progress_bar.progress(40)
            
            # Download information display
            with st.expander("📥 Download Information", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Title:** {title}")
                with col2:
                    st.info(f"**File:** {os.path.basename(audio_path)}")
                
                # Original audio waveform display
                st.markdown("### 🎵 Original Audio Waveform")
                waveform_fig = create_waveform_plot(audio_path, "Original Audio Waveform")
                if waveform_fig:
                    st.plotly_chart(waveform_fig, use_container_width=True)
                
                # Original audio playback
                st.audio(audio_path, format="audio/mp3")
            
            # Step 2: Audio separation
            status_text.text("🎛️ Separating audio sources... (This may take a while)")
            progress_bar.progress(60)
            
            separated_files = separate_audio(audio_path, use_cuda=use_cuda)
            
            status_text.text("✅ Audio Separation Complete")
            progress_bar.progress(100)
            
            # Separation results display
            st.markdown('<h2 class="sub-header">🎵 Separation Results</h2>', unsafe_allow_html=True)
            
            # Display 4 audio sources
            sources = ['vocals', 'drums', 'bass', 'other']
            source_names = ['Vocals', 'Drums', 'Bass', 'Other']
            source_colors = ['#ef4444', '#10b981', '#3b82f6', '#8b5cf6']
            
            # Display separation results in tabs
            tabs = st.tabs(source_names)
            
            for i, (source, name, color) in enumerate(zip(sources, source_names, source_colors)):
                with tabs[i]:
                    if source in separated_files:
                        file_path = separated_files[source]
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Waveform display
                            waveform_fig = create_waveform_plot(file_path, f"{name} Waveform", color)
                            if waveform_fig:
                                st.plotly_chart(waveform_fig, use_container_width=True)
                            
                            # Spectrogram display
                            spec_fig = create_spectrogram_plot(file_path, f"{name} Spectrogram")
                            if spec_fig:
                                st.plotly_chart(spec_fig, use_container_width=True)
                        
                        with col2:
                            # Audio playback
                            st.markdown(f"### 🔊 {name}")
                            st.audio(file_path, format="audio/wav")
                            
                            # Download link
                            st.markdown(get_audio_download_link(file_path, f"Download {name}"), unsafe_allow_html=True)
            
            # Completion message
            st.success("🎉 Audio separation completed successfully!")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
    
    else:
        # Initial screen
        st.markdown("""
        <div class="usage-box animate-fade-in">
            <h3>🚀 How to Use</h3>
            <ol>
                <li>Enter YouTube URL in the sidebar (Sakanaction song is set as default)</li>
                <li>Adjust GPU settings if needed for faster processing</li>
                <li>Click "Start Audio Separation" button to begin</li>
                <li>Wait for processing and download separated audio sources</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature explanation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card animate-fade-in">
                <h3>🎬 YouTube Audio Download</h3>
                <p>Automatically extract high-quality audio from any YouTube video with advanced processing</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card animate-fade-in">
                <h3>🎤 AI Audio Separation</h3>
                <p>Separate vocals, drums, bass, and other instruments using state-of-the-art Demucs AI</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card animate-fade-in">
                <h3>📊 Audio Visualization</h3>
                <p>Visualize audio with interactive waveforms and spectrograms for detailed analysis</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
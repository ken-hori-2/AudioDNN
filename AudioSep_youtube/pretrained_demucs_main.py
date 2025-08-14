import os
import argparse
import torch
import yt_dlp
from demucs.pretrained import get_model
from demucs.apply import apply_model
import soundfile as sf
import numpy as np
import librosa

# デフォルトのURL
# DEFAULT_URL = "https://www.youtube.com/shorts/B4GfdMC5iZg" # ロマンチシズム
# DEFAULT_URL = "https://www.youtube.com/shorts/eFIvdm64lPM" # ライラック
# DEFAULT_URL = "https://www.youtube.com/shorts/VP6p_301xA8" # 僕のこと
# DEFAULT_URL = "https://www.youtube.com/shorts/_1H24i09r14" # ヒカリへ
DEFAULT_URL = "https://www.youtube.com/shorts/GtiIMqCy_3k" # Music

def download_youtube_audio(url, output_path="downloads"):
    """
    YouTubeから音声をダウンロードする関数
    
    Args:
        url (str): YouTubeのURL
        output_path (str): 出力ディレクトリ
        
    Returns:
        str: ダウンロードした音声ファイルのパス
    """
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_path, exist_ok=True)
    
    # yt-dlpのオプション設定
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False,
    }
    
    # 音声をダウンロード
    print(f"Downloading audio from: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_file = os.path.join(output_path, f"{info['title']}.mp3")
        print(f"Downloaded: {info['title']}")
    
    return audio_file

def separate_audio(audio_path, output_dir="separated", use_cuda=False):
    """
    音声ファイルからボーカルと楽器を分離する関数
    
    Args:
        audio_path (str): 音声ファイルのパス
        output_dir (str): 出力ディレクトリ
        use_cuda (bool): CUDAを使用するかどうか
        
    Returns:
        dict: 分離された音声ファイルのパス
    """
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # モデルを読み込む
    print("Loading model...")
    model = get_model('htdemucs')
    
    # CUDAが利用可能な場合のみGPUを使用
    if use_cuda and torch.cuda.is_available():
        print("Using GPU for audio separation")
        model.cuda()
        device = 'cuda'
    else:
        print("Using CPU for audio separation (this may be slow)")
        device = 'cpu'
    
    # 音声を読み込む
    print(f"Loading audio: {audio_path}")
    wav, sr = librosa.load(audio_path, sr=model.samplerate)
    
    # ステレオに変換（モノラルの場合）
    if len(wav.shape) == 1:
        wav = np.stack([wav, wav])
    elif wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)
    
    # NumPyの配列をPyTorchのテンソルに変換
    wav_tensor = torch.from_numpy(wav).float()
    
    # 音声を分離
    print("Separating audio...")
    sources = apply_model(model, wav_tensor.unsqueeze(0), device=device, split=True)[0]
    
    # 分離された音声を保存
    sources_dict = {}
    for source, name in zip(sources, model.sources):
        source_path = os.path.join(output_dir, f"{os.path.basename(audio_path).split('.')[0]}_{name}.wav")
        sf.write(source_path, source.T, model.samplerate)
        sources_dict[name] = source_path
        print(f"Saved {name} to {source_path}")
    
    return sources_dict

def main():
    parser = argparse.ArgumentParser(description="YouTube音源からボーカルと楽器を分離するツール")
    parser.add_argument("--url", default=DEFAULT_URL, help="YouTubeのURL")
    parser.add_argument("--output", "-o", default="separated", help="出力ディレクトリ")
    parser.add_argument("--download-dir", "-d", default="downloads", help="ダウンロードディレクトリ")
    parser.add_argument("--use-cuda", action="store_true", help="CUDAを使用する（利用可能な場合）")
    
    args = parser.parse_args()
    
    # YouTubeから音声をダウンロード
    audio_path = download_youtube_audio(args.url, args.download_dir)
    
    # 音声を分離
    separated_files = separate_audio(audio_path, args.output, args.use_cuda)
    
    print("\n分離が完了しました！")
    print(f"ボーカル: {separated_files.get('vocals', 'Not found')}")
    print(f"ドラム: {separated_files.get('drums', 'Not found')}")
    print(f"ベース: {separated_files.get('bass', 'Not found')}")
    print(f"その他: {separated_files.get('other', 'Not found')}")

if __name__ == "__main__":
    main()

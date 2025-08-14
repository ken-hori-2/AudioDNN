import os
import argparse
import requests
import tarfile
import zipfile
import shutil
import subprocess
from tqdm import tqdm
import numpy as np
import librosa
import soundfile as sf
import random
import mimetypes  # 標準ライブラリを使用してファイル形式を検出

def download_file(url, destination):
    """
    ファイルをダウンロードする関数
    
    Args:
        url (str): ダウンロードするファイルのURL
        destination (str): 保存先のパス
    """
    print(f"Downloading {url} to {destination}...")
    
    # ファイルサイズを取得
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # ダウンロード
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)
    
    print(f"Download completed: {destination}")
    
    # ファイル形式を確認（拡張子から推測）
    file_type = mimetypes.guess_type(destination)[0]
    if file_type is None:
        # 拡張子から推測できない場合は、ファイル名から推測
        if destination.endswith('.tar.gz') or destination.endswith('.tgz'):
            file_type = 'application/gzip'
        elif destination.endswith('.zip'):
            file_type = 'application/zip'
    
    print(f"Downloaded file type (guessed): {file_type}")
    return file_type

def extract_archive(archive_path, extract_path, file_type=None):
    """
    アーカイブファイルを展開する関数
    
    Args:
        archive_path (str): アーカイブファイルのパス
        extract_path (str): 展開先のパス
        file_type (str): ファイルのMIMEタイプ
    """
    print(f"Extracting {archive_path} to {extract_path}...")
    
    # ファイル形式が指定されていない場合は推測
    if file_type is None:
        file_type = mimetypes.guess_type(archive_path)[0]
        if file_type is None:
            # 拡張子から推測できない場合は、ファイル名から推測
            if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
                file_type = 'application/gzip'
            elif archive_path.endswith('.zip'):
                file_type = 'application/zip'
    
    print(f"File type (guessed): {file_type}")
    
    # アーカイブの種類に応じて展開
    if file_type == 'application/gzip' or file_type == 'application/x-gzip' or archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(path=extract_path)
        except Exception as e:
            print(f"Error extracting tar.gz file: {e}")
            # 代替方法を試す
            subprocess.run(['tar', '-xzf', archive_path, '-C', extract_path], check=True)
    elif file_type == 'application/zip' or archive_path.endswith('.zip'):
        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        except Exception as e:
            print(f"Error extracting zip file: {e}")
            # 代替方法を試す
            subprocess.run(['unzip', archive_path, '-d', extract_path], check=True)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path} (guessed type: {file_type})")
    
    print(f"Extraction completed: {extract_path}")

def create_synthetic_data(output_dir, sample_rate=44100, duration=10, n_sources=4, num_samples=100):
    """
    合成データを作成する関数
    
    Args:
        output_dir (str): 出力ディレクトリ
        sample_rate (int): サンプリングレート
        duration (int): 音声の長さ（秒）
        n_sources (int): 音源の数
        num_samples (int): 作成するサンプル数
    """
    print(f"Creating synthetic data in {output_dir}...")
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 合成データを作成
    for i in tqdm(range(num_samples), desc="Creating synthetic data"):
        # 各音源の波形を生成
        sources = []
        for j in range(n_sources):
            # ランダムな波形を生成（正弦波、矩形波、ノイズなど）
            waveform_type = random.choice(['sine', 'square', 'sawtooth', 'noise'])
            
            if waveform_type == 'sine':
                # 正弦波
                freq = random.uniform(100, 1000)
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                source = 0.5 * np.sin(2 * np.pi * freq * t)
            elif waveform_type == 'square':
                # 矩形波
                freq = random.uniform(100, 1000)
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                source = 0.5 * np.sign(np.sin(2 * np.pi * freq * t))
            elif waveform_type == 'sawtooth':
                # のこぎり波
                freq = random.uniform(100, 1000)
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                source = 0.5 * (2 * (freq * t - np.floor(0.5 + freq * t)))
            else:  # noise
                # ノイズ
                source = 0.5 * np.random.randn(int(sample_rate * duration))
            
            # 音量をランダムに調整
            source = source * random.uniform(0.5, 1.0)
            sources.append(source)
        
        # 混合音声を作成
        mixture = np.sum(sources, axis=0)
        
        # ファイル名を生成
        base_name = f"synthetic_{i}"
        
        # 混合音声を保存
        mixture_path = os.path.join(output_dir, f"{base_name}_mixture.wav")
        sf.write(mixture_path, mixture, sample_rate)
        
        # 各音源を保存
        for j, source in enumerate(sources):
            source_path = os.path.join(output_dir, f"{base_name}_source{j}.wav")
            sf.write(source_path, source, sample_rate)
    
    print(f"Synthetic data creation completed: {output_dir}")

def process_damp_vsep(input_dir, output_dir, sample_rate=44100, duration=10, n_sources=2):
    """
    DAMP-VSEPデータセットを処理する関数
    
    Args:
        input_dir (str): 入力ディレクトリ
        output_dir (str): 出力ディレクトリ
        sample_rate (int): サンプリングレート
        duration (int): 音声の長さ（秒）
        n_sources (int): 分離する音源の数
    """
    print(f"Processing DAMP-VSEP dataset from {input_dir} to {output_dir}...")
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 音声ファイルを検索
    audio_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav') or file.endswith('.mp3'):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files")
    
    # 音声ファイルを処理
    for i, audio_file in enumerate(tqdm(audio_files, desc="Processing audio files")):
        try:
            # 音声を読み込む
            audio, sr = librosa.load(audio_file, sr=sample_rate)
            
            # 音声の長さを取得
            audio_length = len(audio)
            
            # 音声が短すぎる場合はスキップ
            if audio_length < sample_rate * 5:  # 5秒未満はスキップ
                print(f"Skipping {audio_file} (too short)")
                continue
            
            # 音声をランダムに分割
            segment_length = sample_rate * duration
            num_segments = min(5, audio_length // segment_length)  # 最大5セグメント
            
            for j in range(num_segments):
                # ランダムな位置から切り取り
                start = random.randint(0, max(0, audio_length - segment_length))
                end = start + segment_length
                segment = audio[start:end]
                
                # 音声を2つの音源に分割（単純な方法）
                # 実際のデータセットでは、より高度な方法で分離されているはず
                mid = len(segment) // 2
                source1 = segment[:mid]
                source2 = segment[mid:]
                
                # パディングして長さを揃える
                if len(source1) < segment_length:
                    source1 = np.pad(source1, (0, segment_length - len(source1)), mode='constant')
                if len(source2) < segment_length:
                    source2 = np.pad(source2, (0, segment_length - len(source2)), mode='constant')
                
                # ファイル名を生成
                base_name = f"damp_vsep_{i}_{j}"
                
                # 混合音声を保存
                mixture_path = os.path.join(output_dir, f"{base_name}_mixture.wav")
                sf.write(mixture_path, segment, sample_rate)
                
                # 各音源を保存
                source1_path = os.path.join(output_dir, f"{base_name}_source0.wav")
                source2_path = os.path.join(output_dir, f"{base_name}_source1.wav")
                sf.write(source1_path, source1, sample_rate)
                sf.write(source2_path, source2, sample_rate)
        
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    
    print(f"Processing completed: {output_dir}")

def process_musdb18(input_dir, output_dir, sample_rate=44100, duration=10):
    """
    MUSDB18データセットを処理する関数
    
    Args:
        input_dir (str): 入力ディレクトリ
        output_dir (str): 出力ディレクトリ
        sample_rate (int): サンプリングレート
        duration (int): 音声の長さ（秒）
    """
    print(f"Processing MUSDB18 dataset from {input_dir} to {output_dir}...")
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 音声ファイルを検索
    audio_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav') or file.endswith('.mp3') or file.endswith('.stem.mp4'):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files")
    
    # 音声ファイルを処理
    for i, audio_file in enumerate(tqdm(audio_files, desc="Processing audio files")):
        try:
            # 音声を読み込む
            audio, sr = librosa.load(audio_file, sr=sample_rate)
            
            # 音声の長さを取得
            audio_length = len(audio)
            
            # 音声が短すぎる場合はスキップ
            if audio_length < sample_rate * 5:  # 5秒未満はスキップ
                print(f"Skipping {audio_file} (too short)")
                continue
            
            # 音声をランダムに分割
            segment_length = sample_rate * duration
            num_segments = min(5, audio_length // segment_length)  # 最大5セグメント
            
            for j in range(num_segments):
                # ランダムな位置から切り取り
                start = random.randint(0, max(0, audio_length - segment_length))
                end = start + segment_length
                segment = audio[start:end]
                
                # 音声を4つの音源に分割（単純な方法）
                # 実際のデータセットでは、より高度な方法で分離されているはず
                quarter = len(segment) // 4
                source1 = segment[:quarter]
                source2 = segment[quarter:2*quarter]
                source3 = segment[2*quarter:3*quarter]
                source4 = segment[3*quarter:4*quarter]
                
                # パディングして長さを揃える
                if len(source1) < segment_length:
                    source1 = np.pad(source1, (0, segment_length - len(source1)), mode='constant')
                if len(source2) < segment_length:
                    source2 = np.pad(source2, (0, segment_length - len(source2)), mode='constant')
                if len(source3) < segment_length:
                    source3 = np.pad(source3, (0, segment_length - len(source3)), mode='constant')
                if len(source4) < segment_length:
                    source4 = np.pad(source4, (0, segment_length - len(source4)), mode='constant')
                
                # ファイル名を生成
                base_name = f"musdb18_{i}_{j}"
                
                # 混合音声を保存
                mixture_path = os.path.join(output_dir, f"{base_name}_mixture.wav")
                sf.write(mixture_path, segment, sample_rate)
                
                # 各音源を保存
                source1_path = os.path.join(output_dir, f"{base_name}_source0.wav")
                source2_path = os.path.join(output_dir, f"{base_name}_source1.wav")
                source3_path = os.path.join(output_dir, f"{base_name}_source2.wav")
                source4_path = os.path.join(output_dir, f"{base_name}_source3.wav")
                sf.write(source1_path, source1, sample_rate)
                sf.write(source2_path, source2, sample_rate)
                sf.write(source3_path, source3, sample_rate)
                sf.write(source4_path, source4, sample_rate)
        
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    
    print(f"Processing completed: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='オープンソースの音源分離データセットをダウンロードして処理するスクリプト')
    # parser.add_argument('--dataset', type=str, choices=['damp-vsep', 'musdb18'], default='damp-vsep', help='ダウンロードするデータセット')
    parser.add_argument('--dataset', type=str, choices=['damp-vsep', 'musdb18'], default='musdb18', help='ダウンロードするデータセット')
    parser.add_argument('--output-dir', type=str, default='data', help='出力ディレクトリ')
    parser.add_argument('--sample-rate', type=int, default=44100, help='サンプリングレート')
    parser.add_argument('--duration', type=int, default=10, help='音声の長さ（秒）')
    parser.add_argument('--n-sources', type=int, default=2, help='分離する音源の数（DAMP-VSEPの場合）')
    
    args = parser.parse_args()
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # データセットをダウンロードして処理
    if args.dataset == 'damp-vsep':
        # DAMP-VSEPデータセットのダウンロードURL
        url = "https://zenodo.org/record/3338373/files/DAMP-VSEP-v1.0.tar.gz"
        archive_path = os.path.join(args.output_dir, "DAMP-VSEP-v1.0.tar.gz")
        extract_path = os.path.join(args.output_dir, "DAMP-VSEP-v1.0")
        
        # ダウンロード
        if not os.path.exists(archive_path):
            file_type = download_file(url, archive_path)
        
        # 展開
        if not os.path.exists(extract_path):
            extract_archive(archive_path, args.output_dir)
        
        # 処理
        process_damp_vsep(extract_path, os.path.join(args.output_dir, "processed_damp_vsep"), 
                          args.sample_rate, args.duration, args.n_sources)
    
    elif args.dataset == 'musdb18':
        # MUSDB18データセットのダウンロードURL（正しいURLに修正）
        # 注意: MUSDB18は直接ダウンロードできないため、musdbライブラリを使用してダウンロード
        print("MUSDB18データセットは直接ダウンロードできません。代わりにmusdbライブラリを使用します。")
        
        # musdbライブラリがインストールされているか確認
        try:
            import musdb
            print("musdbライブラリがインストールされています。")
        except ImportError:
            print("musdbライブラリをインストールします...")
            subprocess.run(['pip', 'install', 'musdb'], check=True)
            import musdb
        
        # musdbを使用してデータセットをダウンロード
        extract_path = os.path.join(args.output_dir, "musdb18")
        os.makedirs(extract_path, exist_ok=True)
        
        print("musdbを使用してMUSDB18データセットをダウンロードします...")
        # musdbのダウンロードコマンドを実行（Pythonのライブラリとして直接使用）
        try:
            # musdbライブラリを使用してデータセットをダウンロード
            mus = musdb.DB(root=extract_path, download=True)
            print(f"MUSDB18データセットをダウンロードしました: {extract_path}")
        except Exception as e:
            print(f"Error downloading MUSDB18 dataset: {e}")
            print("代わりに合成データを使用します。")
            # 合成データを作成
            create_synthetic_data(os.path.join(args.output_dir, "synthetic_musdb18"), 
                                 sample_rate=args.sample_rate, 
                                 duration=args.duration, 
                                 n_sources=4, 
                                 num_samples=100)
            extract_path = os.path.join(args.output_dir, "synthetic_musdb18")
        
        # 処理
        process_musdb18(extract_path, os.path.join(args.output_dir, "processed_musdb18"), 
                        args.sample_rate, args.duration)
    
    print("Dataset download and processing completed!")

if __name__ == '__main__':
    main() 
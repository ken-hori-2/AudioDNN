import os
import argparse
import subprocess
import glob
import time
from tqdm import tqdm

def convert_musdb18_to_wav(input_dir='data/musdb18/train', output_dir='data/musdb18_wav'):
    """
    MUSDB18データセットをWAV形式に変換する関数
    
    Args:
        input_dir (str): 入力ディレクトリ（MUSDB18データセットのディレクトリ）
        output_dir (str): 出力ディレクトリ
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # すべての.stem.mp4ファイルを取得
    stem_files = glob.glob(os.path.join(input_dir, '*.stem.mp4'))
    print(f"Found {len(stem_files)} stem files")
    
    # 各stemファイルを処理
    for stem_file in tqdm(stem_files, desc="Converting stem files"):
        try:
            base_name = os.path.splitext(os.path.basename(stem_file))[0]
            base_name = base_name.replace('.stem', '')
            
            # 出力ファイルのパス
            mixture_path = os.path.join(output_dir, f"{base_name}_mixture.wav")
            source_paths = [
                os.path.join(output_dir, f"{base_name}_source{i}.wav")
                for i in range(4)
            ]
            
            # 各ソースを個別に抽出
            for i in range(4):
                cmd = [
                    'ffmpeg', '-i', stem_file,
                    '-map', f'0:{i+1}',
                    '-acodec', 'pcm_s16le',
                    '-ar', '44100',
                    source_paths[i]
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            
            # ミックスを作成（4つのソースを結合）
            cmd = [
                'ffmpeg',
                '-i', source_paths[0],
                '-i', source_paths[1],
                '-i', source_paths[2],
                '-i', source_paths[3],
                '-filter_complex', '[0:a][1:a][2:a][3:a]amix=inputs=4:duration=longest',
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                mixture_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # 出力ファイルの存在とサイズを確認
            if not os.path.exists(mixture_path) or os.path.getsize(mixture_path) == 0:
                print(f"Error: Failed to create mixture file for {base_name}")
                continue
                
            for source_path in source_paths:
                if not os.path.exists(source_path) or os.path.getsize(source_path) == 0:
                    print(f"Error: Failed to create source file for {base_name}")
                    continue
            
            # システムの負荷を軽減するために少し待機
            time.sleep(0.1)
            
        except subprocess.CalledProcessError as e:
            print(f"Error converting {stem_file}: {e.stderr.decode()}")
        except Exception as e:
            print(f"Error processing {stem_file}: {str(e)}")
    
    print(f"Conversion completed. Output directory: {os.path.abspath(output_dir)}")

def main():
    parser = argparse.ArgumentParser(description='MUSDB18データセットをWAV形式に変換')
    parser.add_argument('--input-dir', type=str, default='data/musdb18/train', help='入力ディレクトリ（MUSDB18データセットのディレクトリ）')
    parser.add_argument('--output-dir', type=str, default='data/musdb18_wav', help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # 入力ディレクトリのパスを絶対パスに変換
    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.input_dir)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
    
    # 変換を実行
    convert_musdb18_to_wav(input_dir, output_dir)

if __name__ == '__main__':
    main() 
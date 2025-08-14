import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from AudioSep_youtube.main import ImprovedAudioSeparationModel, compute_loss

class AudioDataset(Dataset):
    """音声データセット"""
    def __init__(self, data_dir, sample_rate=44100, duration=10):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.segment_length = sample_rate * duration
        
        # データファイルのリストを取得
        self.mixture_files = []
        self.source_files = []
        
        # データディレクトリの構造を探索
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('_mixture.wav'):
                    mixture_path = os.path.join(root, file)
                    
                    # 空のファイルをスキップ
                    if os.path.getsize(mixture_path) == 0:
                        print(f"Skipping empty file: {mixture_path}")
                        continue
                    
                    base_name = file.replace('_mixture.wav', '')
                    
                    # 対応する音源ファイルを探す
                    source_paths = []
                    all_sources_found = True
                    
                    for i in range(4):
                        source_name = f"{base_name}_source{i}.wav"
                        source_path = os.path.join(root, source_name)
                        
                        if os.path.exists(source_path) and os.path.getsize(source_path) > 0:
                            source_paths.append(source_path)
                        else:
                            all_sources_found = False
                            print(f"Missing or empty source file: {source_path}")
                            break
                    
                    # すべての音源ファイルが見つかった場合のみ追加
                    if all_sources_found and len(source_paths) == 4:
                        self.mixture_files.append(mixture_path)
                        self.source_files.append(source_paths)
        
        print(f"Found {len(self.mixture_files)} valid audio pairs")
        
        # データセットが空の場合
        if len(self.mixture_files) == 0:
            print(f"Warning: No valid audio pairs found in {data_dir}")
            print("Please make sure the dataset is properly downloaded and formatted.")
            print("Expected format: {base_name}_mixture.wav and {base_name}_source{i}.wav")
            raise ValueError(f"No valid audio pairs found in {data_dir}. Please check the dataset.")
    
    def __len__(self):
        return len(self.mixture_files)
    
    def __getitem__(self, idx):
        """データセットからアイテムを取得"""
        try:
            # 混合音声を読み込む
            mixture_path = self.mixture_files[idx]
            mixture, sr = librosa.load(mixture_path, sr=self.sample_rate)
            
            # 音源を読み込む
            sources = []
            for source_path in self.source_files[idx]:
                source, sr = librosa.load(source_path, sr=self.sample_rate)
                sources.append(source)
            
            # 音声の長さを統一
            target_length = self.segment_length
            
            # 混合音声の長さを調整
            if len(mixture) > target_length:
                # ランダムな位置から切り取り
                start = np.random.randint(0, len(mixture) - target_length)
                mixture = mixture[start:start + target_length]
            elif len(mixture) < target_length:
                # パディング
                pad_length = target_length - len(mixture)
                mixture = np.pad(mixture, (0, pad_length), mode='constant')
            
            # 各音源の長さを調整
            adjusted_sources = []
            for source in sources:
                if len(source) > target_length:
                    # ランダムな位置から切り取り
                    start = np.random.randint(0, len(source) - target_length)
                    adjusted_source = source[start:start + target_length]
                elif len(source) < target_length:
                    # パディング
                    pad_length = target_length - len(source)
                    adjusted_source = np.pad(source, (0, pad_length), mode='constant')
                else:
                    adjusted_source = source
                
                adjusted_sources.append(adjusted_source)
            
            # テンソルに変換
            mixture = torch.from_numpy(mixture).float()
            sources = torch.from_numpy(np.array(adjusted_sources)).float()
            
            # チャンネル次元を追加（[1, time]の形状に）
            mixture = mixture.unsqueeze(0)
            
            # サイズを確認して必要に応じて調整
            if mixture.shape[1] != target_length:
                if mixture.shape[1] > target_length:
                    mixture = mixture[:, :, :target_length]
                else:
                    pad_length = target_length - mixture.shape[1]
                    mixture = torch.nn.functional.pad(mixture, (0, pad_length), mode='constant', value=0)
            
            if sources.shape[1] != target_length:
                if sources.shape[1] > target_length:
                    sources = sources[:, :, :target_length]
                else:
                    pad_length = target_length - sources.shape[1]
                    sources = torch.nn.functional.pad(sources, (0, pad_length), mode='constant', value=0)
            
            # サイズを確認
            assert mixture.shape[1] == target_length, f"Mixture shape mismatch: {mixture.shape[1]} != {target_length}"
            assert sources.shape[1] == target_length, f"Sources shape mismatch: {sources.shape[1]} != {target_length}"
            
            return mixture, sources
        
        except Exception as e:
            print(f"Error loading audio files: {str(e)}")
            # エラーが発生した場合は、ランダムな音声を生成して返す
            mixture = torch.randn(1, self.segment_length)
            sources = torch.randn(4, self.segment_length)
            return mixture, sources

def train_model(model, train_loader, val_loader, device, num_epochs=100, learning_rate=0.001, save_dir='models'):
    """モデルを学習する関数"""
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(save_dir, exist_ok=True)
    
    # オプティマイザを設定
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学習率スケジューラを設定
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # モデルをデバイスに移動
    model = model.to(device)
    
    # 学習ループ
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        try:
            # 学習モード
            model.train()
            train_loss = 0.0
            
            for batch_idx, (mixtures, sources) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
                try:
                    # データをデバイスに移動
                    mixtures = mixtures.to(device)  # [batch, 1, time]
                    sources = sources.to(device)    # [batch, 4, time]
                    
                    # 勾配をゼロに初期化
                    optimizer.zero_grad()
                    
                    # 順伝播
                    outputs = model(mixtures)  # [batch, 4, time]
                    
                    # 出力とターゲットのサイズを確認
                    if outputs.shape != sources.shape:
                        print(f"Shape mismatch: outputs {outputs.shape}, sources {sources.shape}")
                        # サイズを合わせる
                        min_length = min(outputs.shape[2], sources.shape[2])
                        outputs = outputs[:, :, :min_length]
                        sources = sources[:, :, :min_length]
                        
                        # サイズが一致しない場合は警告を表示
                        if outputs.shape != sources.shape:
                            print(f"Warning: Size mismatch after adjustment: outputs {outputs.shape}, sources {sources.shape}")
                            # さらにサイズを調整
                            if outputs.shape[2] > sources.shape[2]:
                                outputs = outputs[:, :, :sources.shape[2]]
                            else:
                                pad_length = sources.shape[2] - outputs.shape[2]
                                outputs = torch.nn.functional.pad(outputs, (0, pad_length), mode='constant', value=0)
                    
                    # 損失を計算
                    loss = compute_loss(outputs, sources)
                    
                    # 逆伝播
                    loss.backward()
                    optimizer.step()
                    
                    # 損失を累積
                    train_loss += loss.item()
                    
                    # メモリを解放
                    del mixtures, sources, outputs, loss
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    continue
            
            # 平均損失を計算
            train_loss /= len(train_loader)
            
            # 検証モード
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for mixtures, sources in val_loader:
                    try:
                        # データをデバイスに移動
                        mixtures = mixtures.to(device)  # [batch, 1, time]
                        sources = sources.to(device)    # [batch, 4, time]
                        
                        # 順伝播
                        outputs = model(mixtures)  # [batch, 4, time]
                        
                        # 出力とターゲットのサイズを確認
                        if outputs.shape != sources.shape:
                            # サイズを合わせる
                            min_length = min(outputs.shape[2], sources.shape[2])
                            outputs = outputs[:, :, :min_length]
                            sources = sources[:, :, :min_length]
                            
                            # サイズが一致しない場合は警告を表示
                            if outputs.shape != sources.shape:
                                print(f"Warning: Size mismatch after adjustment: outputs {outputs.shape}, sources {sources.shape}")
                                # さらにサイズを調整
                                if outputs.shape[2] > sources.shape[2]:
                                    outputs = outputs[:, :, :sources.shape[2]]
                                else:
                                    pad_length = sources.shape[2] - outputs.shape[2]
                                    outputs = torch.nn.functional.pad(outputs, (0, pad_length), mode='constant', value=0)
                        
                        # 損失を計算
                        loss = compute_loss(outputs, sources)
                        
                        # 損失を累積
                        val_loss += loss.item()
                        
                        # メモリを解放
                        del mixtures, sources, outputs, loss
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                    
                    except Exception as e:
                        print(f"Error in validation batch: {str(e)}")
                        continue
            
            # 平均損失を計算
            val_loss /= len(val_loader)
            
            # 学習率を更新
            scheduler.step(val_loss)
            
            # 結果を表示
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # モデルを保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                print(f'Model saved: {os.path.join(save_dir, "best_model.pth")}')
            
            # 定期的にモデルを保存
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
        
        except Exception as e:
            print(f"Error in epoch {epoch+1}: {str(e)}")
            continue
    
    return model

def main():
    parser = argparse.ArgumentParser(description='改善された音源分離モデルの学習')
    parser.add_argument('--data-dir', type=str, default='data/musdb18_wav', help='データディレクトリ')
    parser.add_argument('--output-dir', type=str, default='models', help='モデルを保存するディレクトリ')
    parser.add_argument('--batch-size', type=int, default=2, help='バッチサイズ')
    parser.add_argument('--num-epochs', type=int, default=100, help='エポック数')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='学習率')
    parser.add_argument('--sample-rate', type=int, default=44100, help='サンプリングレート')
    parser.add_argument('--duration', type=int, default=3, help='音声の長さ（秒）')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='使用するデバイス')
    
    args = parser.parse_args()
    
    # データセットを作成
    print('Loading dataset...')
    dataset = AudioDataset(
        args.data_dir,
        sample_rate=args.sample_rate,
        duration=args.duration
    )
    
    # データセットを分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # データローダーを作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # モデルを作成
    model = ImprovedAudioSeparationModel(n_sources=4)
    
    # モデルを学習
    print('Training model...')
    train_model(
        model,
        train_loader,
        val_loader,
        args.device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_dir=args.output_dir
    )
    
    print('Training completed!')

if __name__ == '__main__':
    main() 
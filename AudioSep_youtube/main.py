import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import yt_dlp
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import datetime

# デフォルトのURL
# DEFAULT_URL = "https://www.youtube.com/shorts/eFIvdm64lPM"
DEFAULT_URL = "https://www.youtube.com/shorts/VP6p_301xA8"

class STFT(nn.Module):
    """短時間フーリエ変換"""
    def __init__(self, n_fft=2048, hop_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)
    
    def forward(self, x):
        # 入力テンソルの次元を確認
        if len(x.shape) == 3:  # [batch, channel, time]
            batch_size = x.shape[0]
            # バッチとチャンネルを結合
            x = x.reshape(-1, x.shape[2])
        
        # STFTを実行
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            return_complex=True
        )
        
        # 振幅と位相に分離
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # 元のバッチ次元を復元
        if len(x.shape) == 2:  # 元が3次元だった場合
            magnitude = magnitude.reshape(batch_size, -1, magnitude.shape[1], magnitude.shape[2])
            phase = phase.reshape(batch_size, -1, phase.shape[1], phase.shape[2])
        
        return magnitude, phase

class ISTFT(nn.Module):
    """逆短時間フーリエ変換"""
    def __init__(self, n_fft=2048, hop_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)
    
    def forward(self, magnitude, phase):
        # 入力テンソルの次元を確認
        if len(magnitude.shape) == 4:  # [batch, channel, freq, time]
            batch_size = magnitude.shape[0]
            # バッチとチャンネルを結合
            magnitude = magnitude.reshape(-1, magnitude.shape[2], magnitude.shape[3])
            phase = phase.reshape(-1, phase.shape[2], phase.shape[3])
        
        # サイズを確認して調整
        if magnitude.shape != phase.shape:
            # 小さい方のサイズに合わせる
            min_freq = min(magnitude.shape[1], phase.shape[1])
            min_time = min(magnitude.shape[2], phase.shape[2])
            magnitude = magnitude[:, :min_freq, :min_time]
            phase = phase[:, :min_freq, :min_time]
        
        # 複素数に変換
        complex_spec = magnitude * torch.exp(1j * phase)
        
        # ISTFTを実行
        istft = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(magnitude.device)
        )
        
        # 元のバッチ次元を復元
        if len(magnitude.shape) == 3:  # 元が4次元だった場合
            istft = istft.reshape(batch_size, -1, istft.shape[1])
        
        return istft

class AttentionBlock(nn.Module):
    """アテンションブロック"""
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.MultiheadAttention(channels, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        # チャンネル次元を最後に移動
        x = x.transpose(1, 2)
        # アテンションの適用
        attn_out, _ = self.attention(x, x, x)
        # 正規化と残差接続
        x = self.norm(x + attn_out)
        # チャンネル次元を元に戻す
        return x.transpose(1, 2)

class ImprovedAudioSeparationModel(nn.Module):
    """改善された音源分離モデル"""
    def __init__(self, n_sources=4):
        super().__init__()
        self.n_sources = n_sources
        
        # STFT/ISTFT層
        self.stft = STFT()
        self.istft = ISTFT()
        
        # エンコーダー部分
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2)
            )
        ])
        
        # アテンションブロック
        self.attention = AttentionBlock(128)
        
        # デコーダー部分
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(128, 64, kernel_size=7, stride=2, padding=3, output_padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose1d(32, 16, kernel_size=7, stride=2, padding=3, output_padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose1d(16, n_sources, kernel_size=7, stride=2, padding=3, output_padding=1),
                nn.Tanh()
            )
        ])
        
        # 出力層
        self.output_layer = nn.Sequential(
            nn.Conv1d(n_sources, n_sources, kernel_size=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # 入力の形状を保存
        original_shape = x.shape
        
        # STFTを適用
        magnitude, phase = self.stft(x)
        original_magnitude_shape = magnitude.shape
        
        # エンコーダー
        x = magnitude
        skip_connections = []
        
        # 入力テンソルのサイズを調整
        if len(x.shape) == 4:  # [batch, channel, freq, time]
            batch_size, channels, freq, time = x.shape
            x = x.reshape(batch_size * channels, 1, freq * time)
        
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
        
        # アテンションブロック
        x = self.attention(x)
        
        # デコーダー
        for i, decoder in enumerate(self.decoder):
            # スキップ接続のチャンネル数を調整
            skip = skip_connections[-(i+1)]
            
            # 入力テンソルのチャンネル数を確認
            print(f"Layer {i}: x shape: {x.shape}, skip shape: {skip.shape}")
            
            # チャンネル数を調整
            target_channels = decoder[0].in_channels - skip.shape[1]
            if x.shape[1] > target_channels:
                x = x[:, :target_channels, :]
            elif x.shape[1] < target_channels:
                pad_channels = target_channels - x.shape[1]
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_channels), mode='replicate')
            
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        # 出力層
        x = self.output_layer(x)
        
        # ISTFTを適用
        # phaseの形状を確認
        print(f"Phase shape: {phase.shape}")
        print(f"X shape before ISTFT: {x.shape}")
        
        # 出力を元のSTFTの形状に変換
        batch_size = phase.shape[0]
        n_sources = x.shape[1]
        freq = phase.shape[2]
        time = phase.shape[3]
        
        # xを適切な形状に変換
        # まず、時間次元を調整
        target_size = freq * time
        if x.shape[2] > target_size:
            x = x[:, :, :target_size]
        elif x.shape[2] < target_size:
            pad_size = target_size - x.shape[2]
            x = torch.nn.functional.pad(x, (0, pad_size), mode='constant', value=0)
        
        # 次に、形状を変換
        x = x.reshape(batch_size, n_sources, freq, time)
        
        # 各ソースに対してISTFTを適用
        output = []
        for i in range(n_sources):
            # 現在のソースの複素スペクトログラムを計算
            source_spec = x[:, i:i+1] * torch.exp(1j * phase)
            
            # ISTFTを実行
            source_audio = torch.istft(
                source_spec.squeeze(1),
                n_fft=self.istft.n_fft,
                hop_length=self.istft.hop_length,
                window=self.istft.window.to(x.device)
            )
            output.append(source_audio)
        
        # 出力を結合
        x = torch.stack(output, dim=1)
        
        # 入力と同じサイズに調整
        if x.shape != original_shape:
            # 時間次元を調整
            if x.shape[2] > original_shape[2]:
                x = x[:, :, :original_shape[2]]
            elif x.shape[2] < original_shape[2]:
                pad_length = original_shape[2] - x.shape[2]
                x = torch.nn.functional.pad(x, (0, pad_length), mode='constant', value=0)
        
        return x

def spectral_loss(outputs, targets):
    """スペクトル損失の計算"""
    outputs_mag, _ = torch.stft(outputs, n_fft=2048, hop_length=512, window=torch.hann_window(2048).to(outputs.device))
    targets_mag, _ = torch.stft(targets, n_fft=2048, hop_length=512, window=torch.hann_window(2048).to(targets.device))
    return F.l1_loss(torch.abs(outputs_mag), torch.abs(targets_mag))

def phase_loss(outputs, targets):
    """位相損失の計算"""
    outputs_phase = torch.angle(torch.stft(outputs, n_fft=2048, hop_length=512, window=torch.hann_window(2048).to(outputs.device)))
    targets_phase = torch.angle(torch.stft(targets, n_fft=2048, hop_length=512, window=torch.hann_window(2048).to(targets.device)))
    return F.l1_loss(outputs_phase, targets_phase)

def compute_loss(outputs, targets):
    """総合損失の計算"""
    # L1損失
    l1_loss = F.l1_loss(outputs, targets)
    
    # スペクトル損失
    spec_loss = spectral_loss(outputs, targets)
    
    # 位相損失
    phase_loss_val = phase_loss(outputs, targets)
    
    # 総合損失
    total_loss = l1_loss + 0.5 * spec_loss + 0.3 * phase_loss_val
    
    return total_loss

def load_model(model_path='models/best_model.pth', device='cuda' if torch.cuda.is_available() else 'cpu'):
    """学習済みモデルをロードする関数"""
    model = ImprovedAudioSeparationModel(n_sources=4)
    
    # モデルの状態辞書を読み込む
    state_dict = torch.load(model_path, map_location=device)
    
    # 予期しないキーを削除
    if 'stft.window' in state_dict:
        del state_dict['stft.window']
    if 'istft.window' in state_dict:
        del state_dict['istft.window']
    
    # 不足しているキーを追加
    if 'output_layer.0.weight' not in state_dict:
        state_dict['output_layer.0.weight'] = model.output_layer[0].weight
    if 'output_layer.0.bias' not in state_dict:
        state_dict['output_layer.0.bias'] = model.output_layer[0].bias
    
    # モデルに状態辞書を読み込む
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    return model

def separate_audio(audio_path, model, output_dir="separated", device='cuda' if torch.cuda.is_available() else 'cpu'):
    """音声を分離する関数"""
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 音声の読み込み
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Error loading audio file: {str(e)}")
        # ファイル名を安全な形式に変更
        safe_path = os.path.join(os.path.dirname(audio_path), "safe_audio.wav")
        # 音声をコピー
        import shutil
        shutil.copy2(audio_path, safe_path)
        # 音声を読み込み
        waveform, sample_rate = torchaudio.load(safe_path)
    
    # モノラルに変換（ステレオの場合）
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 音声をチャンクに分割して処理
    chunk_size = 44100 * 5  # 5秒ごとに処理
    overlap = 44100  # 1秒のオーバーラップ
    
    # 音声の長さを取得
    audio_length = waveform.shape[1]
    print(f"Audio length: {audio_length} samples ({audio_length/sample_rate:.2f} seconds)")
    
    # 出力用のテンソルを初期化
    separated_audio = torch.zeros((4, audio_length), device=device)
    count = torch.zeros(audio_length, device=device)
    
    # チャンクごとに処理
    for start in tqdm(range(0, audio_length, chunk_size - overlap), desc="Separating audio"):
        end = min(start + chunk_size, audio_length)
        chunk = waveform[:, start:end]
        
        # チャンクが短すぎる場合はパディング
        if chunk.shape[1] < chunk_size:
            pad_size = chunk_size - chunk.shape[1]
            chunk = F.pad(chunk, (0, pad_size))
        
        # モデルで分離
        with torch.no_grad():
            separated = model(chunk.unsqueeze(0).to(device))
        
        # オーバーラップ部分を重み付けして加算
        if start > 0:
            # オーバーラップ部分の重み付け
            weight = torch.linspace(0, 1, overlap, device=device)
            separated_audio[:, start:start+overlap] += separated[0, :, :overlap] * weight
            count[start:start+overlap] += weight
            separated_audio[:, start+overlap:end] += separated[0, :, overlap:end-start]
            count[start+overlap:end] += 1
        else:
            separated_audio[:, start:end] += separated[0, :, :end-start]
            count[start:end] += 1
    
    # 重み付け平均
    separated_audio = separated_audio / (count.unsqueeze(0) + 1e-8)
    
    # 分離された音声を保存
    sources_dict = {}
    source_names = ['vocals', 'drums', 'bass', 'other']
    
    # ファイル名から特殊文字を削除し、先頭の空白も削除
    base_filename = os.path.basename(audio_path).split('.')[0]
    safe_filename = ''.join(c for c in base_filename if c.isalnum() or c in ' -_')
    safe_filename = safe_filename.strip()
    
    # ファイル名が空の場合はデフォルト名を使用
    if not safe_filename:
        safe_filename = "separated_audio"
    
    for i, name in enumerate(source_names):
        source_path = os.path.join(output_dir, f"{safe_filename}_{name}.wav")
        sf.write(source_path, separated_audio[i].cpu().numpy(), sample_rate)
        sources_dict[name] = source_path
        print(f"Saved {name} to {source_path}")
    
    return sources_dict

def download_youtube_audio(url, output_dir="downloads"):
    """YouTubeから音声をダウンロードする関数"""
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # yt-dlpのオプションを設定
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
    }
    
    # 音声をダウンロード
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info['title']
        
        # ファイル名から特殊文字、日本語、スペースを削除
        safe_title = ''.join(c for c in title if c.isascii() and c.isalnum() or c in '-_')
        safe_title = safe_title.strip()
        
        # ファイル名が空の場合はデフォルト名を使用
        if not safe_title:
            safe_title = "youtube_audio"
        
        # ファイル名を短くする
        if len(safe_title) > 50:
            safe_title = safe_title[:50]
        
        # ファイル名にタイムスタンプを追加
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = f"{safe_title}_{timestamp}"
        
        audio_path = os.path.join(output_dir, f"{safe_title}.wav")
    
    return audio_path

def main():
    parser = argparse.ArgumentParser(description="改善された音源分離デモ")
    parser.add_argument("--url", default=DEFAULT_URL, help="YouTubeのURL")
    parser.add_argument("--output", "-o", default="separated", help="出力ディレクトリ")
    parser.add_argument("--model", "-m", default="models/model_epoch_100.pth", help="モデルのパス")
    parser.add_argument("--device", "-d", default='cuda' if torch.cuda.is_available() else 'cpu', help="使用するデバイス")
    
    args = parser.parse_args()
    
    # モデルのロード
    print(f"Loading model from {args.model}")
    model = load_model(args.model, args.device)
    
    # YouTubeから音声をダウンロード
    # audio_path = download_youtube_audio(args.url)
    audio_path = "downloads/僕のこと.mp3"
    
    # ファイルの存在を確認
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        return
    
    # 音声を分離
    separated_files = separate_audio(audio_path, model, args.output, args.device)
    
    print("\n分離が完了しました！")
    print(f"ボーカル: {separated_files['vocals']}")
    print(f"ドラム: {separated_files['drums']}")
    print(f"ベース: {separated_files['bass']}")
    print(f"その他: {separated_files['other']}")

if __name__ == "__main__":
    main() 
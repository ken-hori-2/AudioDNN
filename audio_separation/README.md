# U-Net 音源分離システム (Music Source Separation)

PyTorchベースのU-Netアーキテクチャを使用した音楽音源分離システムです。ステレオ音楽から**vocals（ボーカル）**、**drums（ドラム）**、**bass（ベース）**、**other（その他楽器）**の4つのソースを分離します。

## 🎯 プロジェクト概要

- **アーキテクチャ**: U-Net (Jansson et al., 2017)
- **フレームワーク**: PyTorch 2.7+
- **データセット**: MUSDB18
- **目標性能**: SDR 6-7 dB (Spleeter-U-Netレベル)
- **入力**: ステレオ音楽 (44.1kHz)
- **出力**: 4ソース分離音声ファイル

## 📋 システム仕様

### モデル詳細
- **パラメータ数**: 10,919,624 (約10.9M)
- **モデルサイズ**: 41.66 MB
- **エンコーダー**: 6層畳み込み + ダウンサンプリング
- **デコーダー**: 6層転置畳み込み + アップサンプリング
- **スキップ接続**: エンコーダーからデコーダーへ
- **活性化関数**: LeakyReLU (エンコーダー), ReLU (デコーダー)
- **正規化**: Batch Normalization

### オーディオ処理
- **STFT設定**: n_fft=4096, hop_length=1024
- **ウィンドウ**: Hann window (75% overlap)
- **サンプリング**: 44.1kHz ステレオ
- **チャンク処理**: 6秒セグメント (訓練時)

### 性能
- **リアルタイム処理**: 12-46倍速 (CPUでも動作)
- **メモリ効率**: バッチ処理対応
- **精度**: SDR ≈ 6-7 dB (目標値)

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# リポジトリクローン
git clone <repository-url>
cd keyword_spot

# 仮想環境作成・有効化
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 依存関係インストール
pip install -r requirements.txt
```

### 2. 基本テスト実行

```bash
# 基本機能テスト
python test_basic.py

# トレーニング機能テスト
python test_training.py

# 統合テスト
python test_integration.py
```

### 3. 音源分離の実行

```python
import torch
from src.models import create_unet_model
from src.models.separator import SourceSeparator

# モデル作成
model = create_unet_model(
    model_type="unet",
    n_sources=4,
    n_channels=2,
    n_fft=4096
)

# 分離器作成
separator = SourceSeparator(
    model=model,
    n_fft=4096,
    hop_length=1024,
    source_names=['vocals', 'drums', 'bass', 'other'],
    sample_rate=44100
)

# ファイル分離
output_paths = separator.separate_file(
    input_path="input.wav",
    output_dir="separated_audio"
)
```

## 📁 プロジェクト構成

```
keyword_spot/
├── README.md                  # このファイル
├── requirements.txt           # 依存関係
├── .venv/                    # 仮想環境
├── .github/
│   └── copilot-instructions.md  # GitHub Copilot設定
├── src/                      # ソースコード
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py           # U-Netモデル
│   │   └── separator.py      # 音源分離パイプライン
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── audio.py          # 音声処理ユーティリティ
│   │   └── metrics.py        # 評価メトリクス
│   ├── train.py              # 訓練スクリプト
│   ├── evaluate.py           # 評価スクリプト
│   └── separate.py           # 分離実行スクリプト
├── config/
│   └── default.yaml          # デフォルト設定
├── tests/                    # テストファイル
│   ├── test_basic.py         # 基本機能テスト
│   ├── test_training.py      # 訓練テスト
│   └── test_integration.py   # 統合テスト
└── data/                     # データセット (ユーザーが準備)
    └── musdb18/              # MUSDB18データセット
```

## 🧪 テストスイート

### 1. 基本機能テスト (`test_basic.py`)

システムの基本機能をテストします：

```bash
python test_basic.py
```

**テスト項目:**
- ✅ インポートテスト
- ✅ 音声ユーティリティ (STFT/iSTFT)
- ✅ スペクトログラム変換
- ✅ モデル作成
- ✅ 設定ファイル読み込み
- ✅ TensorBoard統合

### 2. 訓練機能テスト (`test_training.py`)

訓練ループと関連機能をテストします：

```bash
python test_training.py
```

**テスト項目:**
- ✅ 合成データでの訓練ループ
- ✅ SourceSeparator基本動作
- ✅ モデル保存・読み込み
- ✅ 勾配計算
- ✅ メモリ使用量

### 3. 統合テスト (`test_integration.py`)

完全なパイプラインの統合テストです：

```bash
python test_integration.py
```

**テスト項目:**
- ✅ 完全な分離パイプライン
- ✅ ファイルベース分離
- ✅ モデル永続化
- ✅ パフォーマンス測定

### テスト結果例

```
🎉 All integration tests passed!
✅ U-Net source separation system is fully functional!

📋 System Summary:
  • Model: U-Net with 10.9M parameters
  • Sources: vocals, drums, bass, other
  • Input: Stereo audio at 44.1kHz
  • Features: File processing, chunking, real-time capable
  • Ready for MUSDB18 training and evaluation
```

## 🎵 使用方法

### コマンドライン実行

#### 1. 音源分離

```bash
python src/separate.py \
    --input input_music.wav \
    --output separated_output/ \
    --model_path models/unet_trained.pth
```

#### 2. モデル訓練

```bash
python src/train.py \
    --config config/default.yaml \
    --data_path data/musdb18 \
    --output_dir experiments/unet_experiment
```

#### 3. モデル評価

```bash
python src/evaluate.py \
    --model_path models/unet_trained.pth \
    --test_data data/musdb18/test \
    --output_dir evaluation_results
```

### Python API

#### 基本的な分離

```python
import torch
import torchaudio
from src.models import create_unet_model
from src.models.separator import SourceSeparator

# 音声読み込み
waveform, sr = torchaudio.load("input.wav")

# モデル・分離器作成
model = create_unet_model("unet", n_sources=4, n_channels=2, n_fft=4096)
separator = SourceSeparator(model=model, sample_rate=sr)

# 分離実行
separated = separator(waveform.unsqueeze(0))  # バッチ次元追加

# 結果保存
for source_name, audio in separated.items():
    torchaudio.save(f"{source_name}.wav", audio.squeeze(0), sr)
```

#### カスタム設定での分離

```python
from src.models.separator import SourceSeparator

# カスタム設定
separator = SourceSeparator(
    model=model,
    n_fft=2048,           # より速い処理
    hop_length=512,
    source_names=['vocals', 'instruments'],  # 2クラス分離
    sample_rate=22050     # 低サンプリングレート
)

# ファイル分離（チャンク処理対応）
output_paths = separator.separate_file(
    input_path="long_music.wav",
    output_dir="output/",
    chunk_duration=10.0,  # 10秒チャンク
    overlap=0.25          # 25%オーバーラップ
)
```

## ⚙️ 設定

### `config/default.yaml`

```yaml
model:
  type: "unet"
  n_fft: 4096
  hop_length: 1024
  n_sources: 4
  n_channels: 2
  sample_rate: 44100
  source_names: ["vocals", "drums", "bass", "other"]
  conv_filters: [16, 32, 64, 128, 256, 512]

training:
  batch_size: 8
  learning_rate: 0.001
  num_epochs: 100
  patience: 10
  gradient_clip_val: 1.0
  optimizer: "adam"
  weight_decay: 0.0001
  scheduler: "plateau"
  loss_function: "l1"
  use_amp: true
  val_interval: 1
  val_patience: 5

data:
  chunk_duration: 6.0
  overlap: 0.25
  augment: true
  normalize: true
```

## 📊 MUSDB18データセット準備

### 1. データセット取得

```bash
# MUSDB18をダウンロード（約4GB）
# https://sigsep.github.io/datasets/musdb.html

# データを配置
mkdir -p data/musdb18
# MUSDB18ファイルをdata/musdb18/に配置
```

### 2. 追加依存関係（オプション）

```bash
pip install musdb museval
```

### 3. データセット検証

```python
import musdb

# データセット読み込みテスト
mus = musdb.DB(root="data/musdb18", is_wav=True)
print(f"Training tracks: {len(mus.train_tracks)}")
print(f"Test tracks: {len(mus.test_tracks)}")
```

## 🔧 開発・カスタマイズ

### 新しいモデルアーキテクチャ追加

```python
# src/models/custom_model.py
import torch.nn as nn

class CustomUNet(nn.Module):
    def __init__(self, n_sources=4, n_channels=2):
        super().__init__()
        # カスタムアーキテクチャ実装
        
    def forward(self, x):
        # フォワードパス実装
        return output

# src/models/__init__.py に追加
def create_custom_model(**kwargs):
    return CustomUNet(**kwargs)
```

### カスタム損失関数

```python
# src/utils/losses.py
import torch.nn as nn

class SpectralLoss(nn.Module):
    def forward(self, pred, target):
        # カスタム損失計算
        return loss
```

### 新しい評価メトリクス

```python
# src/utils/metrics.py
def custom_metric(pred, target):
    """カスタム評価メトリクス"""
    return metric_value
```

## 🐛 トラブルシューティング

### よくある問題

#### 1. メモリエラー
```python
# バッチサイズを小さくする
config['training']['batch_size'] = 4

# またはチャンク処理を使用
separator.separate_file(input_path, output_dir, chunk_duration=5.0)
```

#### 2. CUDA利用可能だがCPU版がインストールされている
```bash
# GPU版PyTorchに変更
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. 音声ファイル読み込みエラー
```python
# 対応形式: WAV, FLAC, MP3 (torchaudioによる)
# ffmpegが必要な場合があります
```

### デバッグモード

```python
# 詳細ログ有効化
import logging
logging.basicConfig(level=logging.DEBUG)

# テンソル形状確認
def debug_shapes(tensor, name):
    print(f"{name}: {tensor.shape}")
```

## 📈 パフォーマンス最適化

### 1. 高速化設定

```python
# Mixed Precision Training有効化
config['training']['use_amp'] = True

# DataLoader並列化
config['training']['num_workers'] = 4

# PyTorchコンパイル (PyTorch 2.0+)
model = torch.compile(model)
```

### 2. メモリ最適化

```python
# 勾配蓄積
config['training']['accumulate_grad_batches'] = 4

# チェックポイント有効化
config['training']['gradient_checkpointing'] = True
```

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🙏 謝辞

- **Jansson et al. (2017)**: "Singing voice separation with deep u-net convolutional networks"
- **MUSDB18データセット**: Rafii et al.
- **Spleeter**: Deezer Research
- **PyTorch**: Meta AI

## 📮 サポート・貢献

- バグレポート: GitHub Issues
- 機能要求: GitHub Discussions
- 貢献: Pull Requests歓迎

---

**🎵 高品質な音源分離をお楽しみください！**

#### 評価パラメータ

- `--model`: 学習済みモデルパス
- `--musdb-root`: MUSDB18データセットパス
- `--test-dir`: テストデータディレクトリ
- `--subset`: MUSDB18サブセット（test/train）
- `--output-dir`: 分離音声保存ディレクトリ
- `--save-estimates`: 分離音声保存フラグ
- `--results-file`: 結果JSON保存パス

## 設定ファイル

`configs/unet_config.yaml`で学習パラメータを調整できます：

```yaml
model:
  model_type: "spleeter_unet"
  n_sources: 4
  source_names: ["vocals", "drums", "bass", "other"]
  # ... その他のパラメータ

training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 0.001
  # ... その他のパラメータ
```

## 期待される性能

MUSDB18テストセットでの目標性能（SDR）：

- **全体平均**: 6-7 dB
- **Vocals**: 6.8 dB
- **Drums**: 6.0 dB  
- **Bass**: 5.2 dB
- **Other**: 4.9 dB

※ Spleeter U-Net（Hennequin et al., JOSS 2020）と同等

## データ形式

### MUSDB18形式
- ステレオ音声（44.1kHz）
- HDF5形式またはWAV形式
- 4音源（vocals, drums, bass, other）

### カスタムデータ形式
```
train/
├── song1_mixture.wav
├── song1_vocals.wav
├── song1_drums.wav
├── song1_bass.wav
├── song1_other.wav
├── song2_mixture.wav
└── ...
```

## GPU使用量

- **学習**: 約6-8GB VRAM（バッチサイズ8）
- **推論**: 約2-3GB VRAM（10秒チャンク）

メモリ不足の場合：
- バッチサイズを減らす（`batch_size: 4`）
- チャンクサイズを短縮（`--chunk-duration 5.0`）

## トラブルシューティング

### よくある問題

1. **CUDA out of memory**
   - バッチサイズまたはチャンクサイズを減らす
   - `mixed_precision: true`を有効化

2. **musdb/museval import エラー**
   - `pip install musdb museval`でインストール
   - または`--test-dir`でカスタム評価

3. **音声読み込みエラー**
   - ffmpegがインストールされているか確認
   - サポート形式: WAV, MP3, FLAC, M4A

### ログ確認

```bash
# TensorBoard起動
tensorboard --logdir ../checkpoints/runs

# ブラウザで http://localhost:6006 を開く
```

## 参考文献

- Jansson et al. "Singing Voice Separation with Deep U-Net Convolutional Networks" (2017)
- Hennequin et al. "Spleeter: a fast and efficient music source separation tool" (JOSS 2020)
- MUSDB18: https://sigsep.github.io/datasets/musdb.html

## ライセンス

MIT License

## 貢献

Issue、Pull Requestを歓迎します。

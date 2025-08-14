# 🎵 AudioSep - AI音源分離ツール

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**YouTube音声からボーカル・楽器を自動分離するAIツール**

[🚀 クイックスタート](#クイックスタート) • [📖 機能](#機能) • [🔧 インストール](#インストール) • [💡 使い方](#使い方) • [🎯 実装バージョン](#実装バージョン) • [🌐 Webアプリ](#webアプリ) • [🎬 デモ](#デモ)

</div>

---

## ✨ 機能

- 🎬 **YouTube音声ダウンロード** - 任意のYouTube動画から音声を自動取得
- 🎤 **高品質音源分離** - ボーカル、ドラム、ベース、その他楽器を分離
- 🤖 **2つの実装方式** - 事前学習済みモデル or 独自学習モデル
- 📊 **学習機能** - 独自の音源分離モデルを学習可能
- 📦 **データセット対応** - MUSDB18、DAMP-VSEP等の標準データセット
- ⚡ **GPU対応** - CUDAを使用した高速処理
- 🌐 **Webアプリ** - Streamlitによる美しいWebインターフェース

## 🎬 デモ

AudioSepの実際の動作を確認できるデモ動画をご覧ください：

<div align="center">

![AudioSep Demo Preview](assets/demo.gif)

*AudioSepのプレビュー - より詳しく見るには下のボタンをクリック*

[![AudioSep Demo](https://img.shields.io/badge/🎬-デモ動画を見る-blue?style=for-the-badge&logo=youtube)](https://github.com/user-attachments/assets/04fdc669-24c3-49f3-bc68-faada70230f9)

*AudioSepのデモンストレーション*

**🎵 デモ内容：**
- YouTube音声の自動ダウンロード
- リアルタイム音源分離処理
- Webアプリの美しいUI操作
- 分離結果の確認とダウンロード

</div>

## 🌐 Webアプリ

### 🚀 Streamlit Webアプリの起動

```bash
# 依存関係をインストール
pip install -r requirements.txt

# Webアプリを起動
streamlit run app.py
```

**Webアプリの特徴：**
- 🎨 **モダンなUI** - 美しいグラデーションとレスポンシブデザイン
- 📊 **リアルタイム可視化** - 波形・スペクトログラムの動的表示
- 🎵 **音声プレビュー** - 分離前後の音声をブラウザで再生
- 📥 **ワンクリックダウンロード** - 分離結果を簡単にダウンロード
- ⚡ **プログレス表示** - 処理状況をリアルタイムで確認
- 🎛️ **GPU設定** - サイドバーでGPU使用設定を簡単切り替え

### 📱 Webアプリの使い方

1. **URL入力** - サイドバーでYouTube URLを入力（デフォルトでサカナクションの楽曲）
2. **設定調整** - GPU使用設定を必要に応じて変更
3. **処理開始** - 「音源分離を開始」ボタンをクリック
4. **結果確認** - タブで分離された音源を確認・ダウンロード

### 🎨 可視化機能

- **波形表示** - 時間軸での音声の振幅変化
- **スペクトログラム** - 時間-周波数領域での音声特性
- **カラーコーディング** - 各音源を異なる色で表示
- **インタラクティブ** - Plotlyによるズーム・パン機能

## 🚀 クイックスタート

### 1. インストール

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/AudioSep.git
cd AudioSep

# 依存関係をインストール
pip install -r requirements.txt

# FFmpegをインストール（音声変換に必要）
# macOS
brew install ffmpeg

# Ubuntu
sudo apt-get install ffmpeg

# Windows
# https://ffmpeg.org/download.html からダウンロード
```

### 2. 簡単な音源分離

```bash
# Webアプリ（推奨）
streamlit run app.py

# コマンドライン版
python pretrained_demucs_main.py
python main.py --url "https://www.youtube.com/watch?v=example"
```

## 🎯 実装バージョン

### 1. **Demucs事前学習済みモデル** (`pretrained_demucs_main.py`)
- ✅ **推奨** - 高品質で安定した分離結果
- 🚀 **即座に使用可能** - 学習不要
- 🎵 **4音源分離** - ボーカル、ドラム、ベース、その他

```bash
python pretrained_demucs_main.py --url "YOUR_YOUTUBE_URL"
```

### 2. **PyTorch独自モデル** (`main.py`)
- 🧠 **学習可能** - 独自のデータで学習
- 🎛️ **カスタマイズ可能** - モデル構造を変更可能
- 📈 **研究用途** - 音源分離アルゴリズムの研究
- 🎬 **YouTube音声対応** - 自動ダウンロード機能付き
- 📁 **自動出力** - separatedフォルダに結果を保存

```bash
# 事前学習済みモデルを使用
python main.py --model "models/model_epoch_100.pth"

# 学習から開始
python train.py --create-synthetic
```

## 💡 使い方

### 📥 YouTube音声のダウンロードと分離

#### Demucs事前学習済みモデル

```bash
# 基本的な使い方
python pretrained_demucs_main.py

# オプション付き
python pretrained_demucs_main.py \
  --url "https://www.youtube.com/watch?v=example" \
  --output "my_separated_audio" \
  --download-dir "my_downloads" \
  --use-cuda
```

#### PyTorch独自モデル（main.py）

```bash
# YouTube音声を自動ダウンロードして分離
python main.py --url "https://www.youtube.com/watch?v=example"

# ダウンロード済みの音声ファイルを分離
# main.py内で audio_path を変更して使用
python main.py --model "models/model_epoch_100.pth"

# カスタム出力ディレクトリを指定
python main.py --output "my_separated_results"
```

**main.pyの特徴：**
- 🎬 **YouTube音声自動ダウンロード** - `download_youtube_audio()`関数で自動取得
- 📁 **downloadsフォルダ** - ダウンロードした音声を保存
- 🎵 **音源分離処理** - `separate_audio()`関数で4音源に分離
- 📂 **separatedフォルダ** - 分離結果を自動保存
- ⚡ **チャンク処理** - 長い音声を5秒ごとに分割して処理
- 🔄 **オーバーラップ処理** - 滑らかな音声結合

### 🎓 モデルの学習

#### 合成データで学習

```bash
# 合成データを作成して学習
python train.py --create-synthetic \
  --data-dir "data" \
  --output-dir "models" \
  --batch-size 32 \
  --num-epochs 200 \
  --learning-rate 0.0005
```

#### オープンソースデータセットで学習

```bash
# データセットをダウンロード
python download_dataset.py --dataset musdb18 --output-dir "data"

# MUSDB18データセットを変換
python convert_musdb18.py --input-dir "data/musdb18" --output-dir "data/processed_musdb18"

# 学習を実行
python train.py --data-dir "data/processed_musdb18" --output-dir "models"
```

### 📊 利用可能なデータセット

| データセット | 説明 | 音源数 | ライセンス |
|-------------|------|--------|-----------|
| **MUSDB18** | 音楽源分離の標準データセット | 4音源 | CC-BY 4.0 |
| **DAMP-VSEP** | ボーカル分離専用データセット | 2音源 | CC-BY 4.0 |

## 🔧 コマンドラインオプション

### 共通オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--url` | YouTubeのURL | サンプルURL |
| `--output, -o` | 出力ディレクトリ | `separated` |
| `--download-dir, -d` | ダウンロードディレクトリ | `downloads` |
| `--use-cuda` | GPUを使用 | `False` |

### main.py専用オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--model, -m` | 学習済みモデルのパス | `models/model_epoch_100.pth` |
| `--device, -d` | 使用デバイス | `cuda` (利用可能な場合) |

### 学習オプション (`train.py`)

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--data-dir` | データディレクトリ | `data` |
| `--output-dir` | モデル保存ディレクトリ | `models` |
| `--batch-size` | バッチサイズ | `16` |
| `--num-epochs` | エポック数 | `100` |
| `--learning-rate` | 学習率 | `0.001` |
| `--create-synthetic` | 合成データを作成 | `False` |

### データセット変換オプション (`convert_musdb18.py`)

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--input-dir` | MUSDB18データセットの入力ディレクトリ | `data/musdb18` |
| `--output-dir` | 変換後の出力ディレクトリ | `data/processed_musdb18` |

## 📁 プロジェクト構造

```
AudioSep/
├── 📄 main.py                      # PyTorch独自モデル（メイン）
├── 📄 pretrained_demucs_main.py    # Demucs事前学習済みモデル
├── 📄 train.py                     # モデル学習スクリプト
├── 📄 download_dataset.py          # データセットダウンロード
├── 📄 convert_musdb18.py           # MUSDB18データセット変換
├── 📄 requirements.txt             # 依存関係
├── 📁 src/                         # ソースコード
│   ├── 📁 u-net/                   # U-Net実装
│   ├── 📁 waveunet/                # Wave-U-Net実装
│   └── 📄 hybrid_demucs_tutorial.py
├── 📁 Meta_src/                    # Meta/Facebook実装
├── 📁 models/                      # 学習済みモデル
├── 📁 data/                        # データセット
├── 📁 downloads/                   # ダウンロード音声
└── 📁 separated/                   # 分離結果
```

## 🎵 出力形式

分離された音声ファイルは以下の形式で保存されます：

```
separated/
├── 🎤 {filename}_vocals.wav      # ボーカル
├── 🥁 {filename}_drums.wav       # ドラム
├── 🎸 {filename}_bass.wav        # ベース
└── 🎹 {filename}_other.wav       # その他の楽器
```

**main.pyの処理フロー：**
1. 📥 **YouTube音声ダウンロード** → `downloads/` フォルダ
2. 🎵 **音源分離処理** → チャンク分割・オーバーラップ処理
3. 📂 **結果保存** → `separated/` フォルダ

## 🛠️ 技術仕様

### 依存関係

- **Python**: 3.8以上
- **PyTorch**: 2.0.1
- **torchaudio**: 2.0.1
- **librosa**: 0.10.1
- **yt-dlp**: 2023.11.16
- **FFmpeg**: 音声変換用

### 対応フォーマット

- **入力**: MP3, WAV, M4A, FLAC
- **出力**: WAV (44.1kHz, 16bit)
- **サンプリングレート**: 44.1kHz (デフォルト)

## 🚨 注意事項

- ⚖️ **著作権**: YouTubeの利用規約と著作権法を遵守してください
- 🎯 **教育目的**: このツールは教育・研究目的で作成されています
- 💾 **ストレージ**: 音声ファイルは大きな容量を必要とします
- 🔧 **GPU推奨**: 学習にはGPUの使用を強く推奨します

## 🤝 貢献

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 🙏 謝辞

- [Demucs](https://github.com/facebookresearch/demucs) - Meta/Facebook Research
- [MUSDB18](https://sigsep.github.io/datasets/musdb.html) - 音楽源分離データセット
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTubeダウンローダー

---

<div align="center">

**🎵 AudioSepで音楽の新しい可能性を発見しよう！**

[⭐ Star this repo](https://github.com/yourusername/AudioSep) • [🐛 Report issues](https://github.com/yourusername/AudioSep/issues)

</div> 
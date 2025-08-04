<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# GitHub Copilot Instructions

このプロジェクトは、PyTorchを使用したU-Netベースの音源分離システムです。以下の指針に従ってコードを生成してください：

## プロジェクト概要
- **目的**: 音楽音声からボーカル、ドラム、ベース、その他の楽器を分離
- **フレームワーク**: PyTorch
- **アーキテクチャ**: U-Net (Jansson et al., 2017)
- **データセット**: MUSDB18
- **目標性能**: SDR 6-7 dB (Spleeterレベル)

## コーディング規約
- Python 3.8+ 対応
- Type hints を使用
- Black フォーマッター準拠
- Docstrings は Google スタイル
- テンソル操作は PyTorch ネイティブ関数を優先

## アーキテクチャ指針
- **入力**: スペクトログラム (F x T x C)
- **出力**: ソフトマスク (F x T x C x Sources)
- **損失**: L1 損失（マスクされたスペクトログラムとターゲット間）
- **STFT設定**: n_fft=4096, hop_length=1024
- **サンプリングレート**: 44100 Hz

## モデル設計原則
1. エンコーダー: 6層の畳み込み＋ダウンサンプリング
2. デコーダー: 6層の転置畳み込み＋アップサンプリング
3. スキップ接続: エンコーダーからデコーダーへ
4. 活性化: エンコーダーはLeakyReLU、デコーダーはReLU
5. 正規化: Batch Normalization
6. ドロップアウト: 0.5（デコーダーの深い層）

## データ処理パターン
- STFT: Hann window, 75% overlap
- 正規化: 周波数ビンごとの平均・標準偏差
- セグメント化: 6秒チャンク（学習時）
- データ拡張: ランダムゲイン、位相シフト

## 学習設定
- オプティマイザー: Adam (lr=1e-3)
- バッチサイズ: 8
- スケジューラー: ReduceLROnPlateau
- 早期停止: patience=10

## ファイル命名規則
- モデル: `{model_name}_{timestamp}.pth`
- 設定: `{experiment_name}_config.yaml`
- ログ: `{experiment_name}_{timestamp}.log`

## エラーハンドリング
- GPU メモリ不足時の graceful degradation
- オーディオファイル読み込みエラーの適切な処理
- MUSDB18 データセット不在時の明確なエラーメッセージ

## パフォーマンス最適化
- Mixed precision training 対応
- DataLoader の num_workers 最適化
- STFT 計算の効率化
- メモリ使用量の監視

## テスト・評価
- SDR, SAR, SIR, ISR メトリクス
- MUSDB18 公式評価プロトコル準拠
- 音声品質の主観評価指標も考慮

import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tqdm import tqdm
import random

# 対象とする5クラス
TARGET_CLASSES = [
    'dog',             # 犬の鳴き声
    'cat',             # 猫の鳴き声
    'chirping_birds',  # 鳥のさえずり
    'finger_snap',     # 指パッチン
    'human_voice'      # 人の声（自分の声を含む）
]

class MelSpectrogramTransform:
    def __init__(self, p=0.5, is_finger_snap=False):
        self.p = p
        self.is_finger_snap = is_finger_snap
        
    def time_masking(self, mel_spec):
        if random.random() < self.p:
            time_steps = mel_spec.shape[1]
            # 指パッチンの場合はマスク幅をさらに小さく
            if self.is_finger_snap:
                mask_width = random.randint(1, time_steps // 16)
            else:
                mask_width = random.randint(1, time_steps // 8)
            mask_start = random.randint(0, time_steps - mask_width)
            mel_spec[:, mask_start:mask_start + mask_width] = 0
        return mel_spec
    
    def frequency_masking(self, mel_spec):
        if random.random() < self.p:
            freq_steps = mel_spec.shape[0]
            # 指パッチンの場合はマスク幅をさらに小さく
            if self.is_finger_snap:
                mask_width = random.randint(1, freq_steps // 16)
            else:
                mask_width = random.randint(1, freq_steps // 8)
            mask_start = random.randint(0, freq_steps - mask_width)
            mel_spec[mask_start:mask_start + mask_width, :] = 0
        return mel_spec
    
    def add_noise(self, mel_spec):
        if random.random() < self.p:
            # 指パッチンの場合はノイズ強度をさらに下げる
            if self.is_finger_snap:
                noise = np.random.normal(0, 0.01, mel_spec.shape)
            else:
                noise = np.random.normal(0, 0.03, mel_spec.shape)
            mel_spec = mel_spec + noise
        return mel_spec
    
    def __call__(self, mel_spec):
        mel_spec = self.time_masking(mel_spec)
        mel_spec = self.frequency_masking(mel_spec)
        mel_spec = self.add_noise(mel_spec)
        return mel_spec

class AudioDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform is not None:
            feature = self.transform(feature.numpy())
            feature = torch.FloatTensor(feature)
        
        return feature, label

class SoundClassifierNet(nn.Module):
    def __init__(self, num_classes=6):
        super(SoundClassifierNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 高周波数特徴を捉えるための追加の畳み込み層
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # 追加の畳み込み層
        x = self.adaptive_pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class SoundClassifier:
    def __init__(self, sample_rate=22050, duration=3):
        self.sample_rate = sample_rate
        self.duration = duration
        self.model = None
        self.label_encoder = LabelEncoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_audio(self, file_path):
        """音声ファイルを読み込んで前処理を行う"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # 音声の長さを統一
            target_length = int(self.sample_rate * self.duration)
            
            # 音声が短すぎる場合（1サンプル未満）はエラーを返す
            if len(audio) < 1:
                print(f"警告: {file_path} の音声が短すぎます（{len(audio)}サンプル）")
                return None
            
            # 音声が短すぎる場合は、パディングを追加
            if len(audio) < 256:  # 最小FFTサイズ
                audio = np.pad(audio, (0, 256 - len(audio)))
            
            if len(audio) > target_length:
                # 音声のエネルギーが最も高い部分を中心に切り取る
                energy = librosa.feature.rms(y=audio)[0]
                max_energy_idx = np.argmax(energy)
                center_idx = int(max_energy_idx * (len(audio) / len(energy)))
                
                # 中心を基準に前後に均等に分割
                start_idx = max(0, center_idx - target_length // 2)
                end_idx = min(len(audio), start_idx + target_length)
                
                # 開始位置が0の場合は、終了位置を調整
                if start_idx == 0:
                    end_idx = target_length
                    start_idx = 0
                # 終了位置が音声長を超える場合は、開始位置を調整
                elif end_idx > len(audio):
                    start_idx = len(audio) - target_length
                    end_idx = len(audio)
                
                audio = audio[start_idx:end_idx]
            else:
                # 音声が短い場合は、無音パディングを前後に均等に追加
                pad_length = target_length - len(audio)
                pad_left = pad_length // 2
                pad_right = pad_length - pad_left
                audio = np.pad(audio, (pad_left, pad_right))
            
            # メルスペクトログラムの計算（サイズを固定）
            # FFTサイズを音声長に合わせて調整（最小値は256）
            n_fft = max(256, min(2048, len(audio)))
            hop_length = min(512, len(audio) // 4)  # ホップ長も調整
            n_mels = 128
            
            # 音声が短すぎる場合は、パディングを追加
            if len(audio) < n_fft:
                audio = np.pad(audio, (0, n_fft - len(audio)))
            
            # メルスペクトログラムの計算
            try:
                mel_spec = librosa.feature.melspectrogram(
                    y=audio, 
                    sr=self.sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    fmax=8000
                )
            except Exception as e:
                print(f"メルスペクトログラム計算エラー: {e}")
                return None
            
            # メルスペクトログラムの時間フレーム数を固定
            target_frames = int(target_length / hop_length) + 1
            if mel_spec.shape[1] > target_frames:
                mel_spec = mel_spec[:, :target_frames]
            elif mel_spec.shape[1] < target_frames:
                pad_width = target_frames - mel_spec.shape[1]
                mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)))
            
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 特徴量の形状を確認
            print(f"特徴量の形状: {mel_spec_db.shape}")
            
            return mel_spec_db
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None

    def prepare_data(self, data_dir, meta_file, custom_voice_dir=None, custom_fingersnap_dir=None, oversample=True):
        """データセットの準備（5クラス + カスタム音声）"""
        features = []
        labels = []
        
        # メタデータの読み込み
        meta_df = pd.read_csv(meta_file)
        
        # 5クラスのデータのみを選択（human_voiceとfinger_snapを除く）
        meta_df = meta_df[meta_df['category'].isin(TARGET_CLASSES[:-2])]
        
        # クラスごとのサンプル数をカウント
        class_counts = {}
        
        # ESC-50データセットの処理
        for _, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
            file_path = os.path.join(data_dir, row['filename'])
            feature = self.load_audio(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(row['category'])
                class_counts[row['category']] = class_counts.get(row['category'], 0) + 1
        
        # カスタム音声の処理
        if custom_voice_dir and os.path.exists(custom_voice_dir):
            for file_name in os.listdir(custom_voice_dir):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(custom_voice_dir, file_name)
                    feature = self.load_audio(file_path)
                    if feature is not None:
                        features.append(feature)
                        labels.append('human_voice')
                        class_counts['human_voice'] = class_counts.get('human_voice', 0) + 1
        
        # 指パッチンの音声の処理
        if custom_fingersnap_dir and os.path.exists(custom_fingersnap_dir):
            for file_name in os.listdir(custom_fingersnap_dir):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(custom_fingersnap_dir, file_name)
                    feature = self.load_audio(file_path)
                    if feature is not None:
                        features.append(feature)
                        labels.append('finger_snap')
                        class_counts['finger_snap'] = class_counts.get('finger_snap', 0) + 1
        
        # クラスごとのサンプル数を表示
        print("\nクラスごとのサンプル数（オーバーサンプリング前）:")
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count}サンプル")
        
        # オーバーサンプリング
        if oversample:
            # 最大サンプル数を取得
            max_samples = max(class_counts.values())
            
            # 各クラスのサンプルを複製
            oversampled_features = []
            oversampled_labels = []
            
            # クラスごとに処理
            for class_name in class_counts.keys():
                # 現在のクラスのインデックスを取得
                class_indices = [i for i, label in enumerate(labels) if label == class_name]
                
                # 元のサンプルを追加
                for idx in class_indices:
                    oversampled_features.append(features[idx])
                    oversampled_labels.append(labels[idx])
                
                # 少数クラスの場合、サンプルを複製してデータ拡張を適用
                if class_counts[class_name] < max_samples:
                    # 複製回数を計算
                    num_copies = max_samples // class_counts[class_name]
                    
                    # 指パッチンの場合は特別なデータ拡張を適用
                    is_finger_snap = (class_name == 'finger_snap')
                    transform = MelSpectrogramTransform(p=0.5, is_finger_snap=is_finger_snap)
                    
                    for _ in range(num_copies - 1):
                        # 各サンプルに対して複製を作成
                        for idx in class_indices:
                            try:
                                # データ拡張を適用
                                augmented_feature = transform(features[idx])
                                oversampled_features.append(augmented_feature)
                                oversampled_labels.append(labels[idx])
                            except Exception as e:
                                print(f"データ拡張エラー: {e}")
                                # エラーが発生した場合は元のサンプルを追加
                                oversampled_features.append(features[idx])
                                oversampled_labels.append(labels[idx])
            
            features = oversampled_features
            labels = oversampled_labels
            
            # オーバーサンプリング後のクラスごとのサンプル数を表示
            class_counts = {}
            for label in labels:
                class_counts[label] = class_counts.get(label, 0) + 1
            
            print("\nクラスごとのサンプル数（オーバーサンプリング後）:")
            for class_name, count in class_counts.items():
                print(f"{class_name}: {count}サンプル")
        
        # ラベルのエンコーディング
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # 特徴量の形状を確認
        print(f"特徴量の数: {len(features)}")
        if len(features) > 0:
            print(f"最初の特徴量の形状: {features[0].shape}")
        
        # 特徴量の正規化
        try:
            # 特徴量の形状を統一
            max_time_frames = max(feat.shape[1] for feat in features)
            normalized_features = []
            
            for feat in features:
                # 時間フレーム数を統一
                if feat.shape[1] < max_time_frames:
                    pad_width = max_time_frames - feat.shape[1]
                    feat = np.pad(feat, ((0, 0), (0, pad_width)))
                normalized_features.append(feat)
            
            # 特徴量をスタック
            features = np.stack(normalized_features)
            print(f"スタック後の特徴量の形状: {features.shape}")
            
            # 正規化
            features = (features - np.mean(features)) / np.std(features)
            
            # チャンネル次元の追加
            features = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)
        except Exception as e:
            print(f"特徴量の正規化エラー: {e}")
            # エラーが発生した場合は、元の特徴量をそのまま使用
            features = np.stack(features)
            features = (features - np.mean(features)) / np.std(features)
            features = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)
        
        return features, encoded_labels

    def calculate_class_weights(self, labels):
        """クラス重みの計算（改善版）"""
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        
        # 重みの計算方法を改善
        # クラスごとのサンプル数の逆数の平方根を使用
        class_weights = np.sqrt(total_samples / class_counts)
        
        # 指パッチンクラスの重みをさらに強化
        finger_snap_idx = self.label_encoder.transform(['finger_snap'])[0]
        class_weights[finger_snap_idx] *= 1.5  # 指パッチンクラスの重みを1.5倍に
        
        # 重みの正規化
        class_weights = class_weights / np.sum(class_weights)
        
        return torch.FloatTensor(class_weights).to(self.device)

    def train(self, data_dir, meta_file, custom_voice_dir=None, custom_fingersnap_dir=None, epochs=100, batch_size=32, learning_rate=0.0003):
        """モデルの学習"""
        # データの準備
        features, labels = self.prepare_data(data_dir, meta_file, custom_voice_dir, custom_fingersnap_dir)
        
        # データの分割
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # クラス重みの計算
        class_weights = self.calculate_class_weights(y_train)
        
        # データ拡張の設定
        transform = MelSpectrogramTransform(p=0.5)
        
        # データセットとデータローダーの作成
        train_dataset = AudioDataset(X_train, y_train, transform=transform)
        test_dataset = AudioDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # モデルの構築
        self.model = SoundClassifierNet(num_classes=len(TARGET_CLASSES)).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        # 学習
        best_accuracy = 0.0
        patience_counter = 0
        patience_limit = 25
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for features, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 検証
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for features, labels in test_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(features)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f'Epoch {epoch+1}, Loss: {train_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')
            
            # 学習率の調整
            scheduler.step(accuracy)
            
            # 最適モデルの保存
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy,
                    'label_encoder': self.label_encoder
                }, 'best_model_5class.pth')
                print(f'Best model saved with accuracy: {accuracy:.2f}%')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience_limit:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

    def predict(self, audio_file, threshold=0.7):
        """音声ファイルの分類"""
        if self.model is None:
            print("モデルが学習されていません。")
            return None
            
        feature = self.load_audio(audio_file)
        if feature is None:
            return None
        
        # 特徴量の正規化と形状の変更
        feature = (feature - np.mean(feature)) / np.std(feature)
        feature = feature.reshape(1, feature.shape[0], feature.shape[1], 1)
        
        # 予測
        self.model.eval()
        with torch.no_grad():
            feature = torch.FloatTensor(feature).to(self.device)
            outputs = self.model(feature)
            probabilities = torch.softmax(outputs, dim=1)
            
            # 指パッチンの確率を確認
            finger_snap_idx = self.label_encoder.transform(['finger_snap'])[0]
            finger_snap_prob = probabilities[0, finger_snap_idx].item()
            
            # 指パッチンの確率が高い場合は、閾値を下げる
            if finger_snap_prob > 0.3:  # 指パッチンの確率が30%以上の場合
                threshold = 0.5  # 閾値を下げる
            
            max_prob, predicted = torch.max(probabilities, 1)
            
            # 確率が閾値未満の場合はUnknown
            if max_prob.item() < threshold:
                return "Unknown", max_prob.item()
            
            predicted_label = self.label_encoder.inverse_transform([predicted.item()])
            return predicted_label[0], max_prob.item()

    def load_model(self, model_path):
        """保存されたモデルの読み込み"""
        checkpoint = torch.load(model_path)
        self.model = SoundClassifierNet(num_classes=len(TARGET_CLASSES)).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.label_encoder = checkpoint['label_encoder']
        print(f"Loaded model with accuracy: {checkpoint['accuracy']:.2f}%")

if __name__ == "__main__":
    # ESC-50データセットのパスを指定
    data_dir = "ESC-50-master/audio"
    meta_file = "ESC-50-master/meta/esc50.csv"
    
    # カスタム音声のディレクトリを指定
    custom_voice_dir = "custom_voice"
    custom_fingersnap_dir = "custom_fingersnap"
    
    # 分類器のインスタンス化
    classifier = SoundClassifier()
    
    # モデルの学習
    classifier.train(data_dir, meta_file, custom_voice_dir, custom_fingersnap_dir) 
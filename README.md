# DETR Balloon Dataset ファインチューニング

このノートブックは、DETR（Detection Transformer）モデルをBalloon Dataset（風船検出）でファインチューニングするためのコードです。

## 概要

- **モデル**: `facebook/detr-resnet-50` をベースにファインチューニング
- **データセット**: Balloon Dataset（風船検出）
- **タスク**: 1クラス物体検出（balloon + 背景クラス）
- **アノテーション形式**: VIA形式 → COCO形式に自動変換
- **実行環境**: Google Colab推奨（GPU使用）

## 特徴

- データセットの自動ダウンロードと準備
- VIA形式からCOCO形式への自動変換
- Google Colabでそのまま実行可能
- カスタムデータセットクラスとTrainerクラス
- 学習曲線の可視化
- 推論と可視化機能

## データセット

- **トレーニング画像**: 61枚
- **バリデーション画像**: 13枚
- **アノテーション数**: トレーニング255個、バリデーション50個
- **データセットURL**: [Matterport Balloon Dataset](https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip)

## 必要なライブラリ

```bash
pip install transformers torch torchvision pillow matplotlib requests datasets accelerate
```

## 使用方法

### 1. Google Colabでの実行

1. Google Colabでこのノートブックを開く
2. **Runtime > Change runtime type > GPU** を選択してGPUを有効化
3. セルを順番に実行する

### 2. ローカル環境での実行

ノートブック内のパス設定をローカル環境に合わせて変更してください：

```python
# Google Colab用のパス（/content）をローカルパスに変更
DATASET_DIR = "/path/to/your/dataset"
```

## ノートブックの構成

### 1. 必要なライブラリのインストール
必要なPythonパッケージをインストールします。

### 2. GPUの確認
CUDAの利用可能性とGPU情報を確認します。

### 3. Balloon Datasetのダウンロードと準備
- MatterportのBalloon Datasetを自動ダウンロード
- ZIPファイルを展開
- データセット構造を確認

### 4. VIA形式からCOCO形式への変換
- VIA形式のアノテーション（`via_region_data.json`）をCOCO形式に変換
- ポリゴンアノテーションからバウンディングボックスを計算
- トレーニング/バリデーション分割（80/20）

### 5. 設定とインポート
- モデルパス、データセットパスなどの設定
- 必要なライブラリのインポート

### 6. データセットクラスの定義
- `CocoDetection`クラス: COCO形式のデータセットを読み込む
- 画像の前処理とラベルの準備

### 7. モデルとプロセッサの読み込み
- `DetrImageProcessor`の読み込み
- `DetrForObjectDetection`モデルの読み込み
- カスタムラベル設定（background + balloon）

### 8. データセットの読み込み
- トレーニングデータセットとバリデーションデータセットの作成
- データセット構造の確認

### 9. トレーニング引数の設定
- バッチサイズ、エポック数、学習率などの設定
- 評価戦略、保存戦略の設定

### 10. カスタムTrainerクラス
- `CustomTrainer`: 損失計算をオーバーライド
- `detr_collate_fn`: DETR用のカスタムデータコレクター

### 11. トレーニングの実行
- ファインチューニングの実行
- 各エポックでバリデーション評価

### 12. モデルの保存
- ベストモデルの自動保存
- Google Driveへの保存（オプション）

### 13. 学習曲線の可視化
- トレーニング損失とバリデーション損失の可視化
- 学習の進捗を確認

### 14. 推論（風船検知）
- 保存したモデルを使用した推論
- 検出結果の可視化（バウンディングボックスとスコア）

## トレーニング設定

- **エポック数**: 35
- **バッチサイズ**: 4（per device）
- **学習率**: 5e-5
- **Weight Decay**: 1e-4
- **学習率スケジューラー**: Cosine
- **Warmup Ratio**: 0.1
- **評価戦略**: 各エポック終了時
- **保存戦略**: 各エポック終了時（ベスト3モデルのみ保持）

## 出力

- **モデル保存先**: `outputs/` ディレクトリ
- **ベストモデル**: バリデーション損失が最小のモデルが自動保存
- **Google Drive保存**: `/content/drive/MyDrive/detr_balloon_model`（オプション）

## 注意事項

- **GPUの使用を強く推奨**: CPUでは非常に時間がかかります
- **トレーニング時間**: 数時間かかる場合があります（GPU使用時）
- **メモリ使用量**: GPUメモリが不足する場合はバッチサイズを減らしてください
- **データセット**: 初回実行時に自動ダウンロードされます

## トラブルシューティング

### GPUが認識されない場合
- Google Colabで **Runtime > Change runtime type > GPU** を確認
- `torch.cuda.is_available()` で確認

### メモリ不足エラー
- `per_device_train_batch_size` を減らす（例: 4 → 2）
- `gradient_accumulation_steps` を増やす

### データセットが見つからない
- ダウンロードセルを再実行
- パス設定を確認

## ライセンス

このノートブックは教育・研究目的で使用できます。使用するモデルとデータセットのライセンスを確認してください。

## zenn記事
 - [DETRを既存の物体検出データセットを用いフルファインチューニングでカスタマイズする！](https://zenn.dev/rakushaking/articles/ba08e64ffd54bf)

## 参考資料

- [DETR論文](https://arxiv.org/abs/2005.12872)
- [Hugging Face DETR](https://huggingface.co/docs/transformers/model_doc/detr)
- [Balloon Dataset](https://github.com/matterport/Mask_RCNN)



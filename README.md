# Edge Trash Detection with AITRIOS and IMX500

## 概要
このプロジェクトは、**SSDLite MobileDet** モデルを用いて、trash-detectionデータセットで転移学習を行い、8ビット量子化された**TFLiteモデル**を生成することで、**IMX500デバイス**上でのリアルタイムのゴミ検出を実現することを目的としています。量子化によりエッジデバイスのリソース消費を最小限に抑え、高効率なリアルタイム推論が可能です。

## プロジェクトの目的
- 軽量かつリソース効率の高いゴミ検出モデルをエッジデバイス用に構築し、環境モニタリングを効率化すること。
- 転移学習および8ビット量子化を施したモデルの生成。

## 必要な前提条件

- **ハードウェア**：
  - NVIDIA GPU（CUDA Compute Capability 6.1以上）
  - 最低8GBのGPUメモリ

- **ソフトウェア**：
  - Docker 20.10.0以上
  - NVIDIAドライバ 450.80.02以上
  - NVIDIA Container Toolkit
  - Git 2.28.0以上
  - Python 3.8.x
  - TensorFlow 1.15.5（モデル学習用）
  - TensorFlow 2.4.1（モデル量子化用）

## プロジェクト構成

```
edge-trash-detection/
├── Dockerfile                             # Dockerイメージビルド用の設定ファイル
├── README.md                              # プロジェクトの概要と実行手順
├── build_docker.sh                        # Dockerイメージのビルドスクリプト
├── config/
│   └── ssdlite_mobilenet_td.config        # 転移学習用の設定ファイル
├── data/
│   └── trash-detection/                   # データセット格納ディレクトリ
├── dataset/
│   └── prepare_trash_detection_dataset.py # データセット準備スクリプト
├── models/                                # 学習済みモデルやチェックポイントを保存
│   ├── pretrained_model/                  # 事前学習済みモデル
│   └── retrained_ssdlite_mobiledet_td_<timestamp>/ # trash-detectionデータセットでの再学習モデル
├── scripts/                               # 実行スクリプト
│   ├── download_model.sh                  # 事前学習済みモデルのダウンロードスクリプト
│   ├── quantize.sh                        # モデルの量子化とTFLite変換スクリプト
│   ├── retrain.sh                         # モデルの転移学習スクリプト
│   ├── retrain_ssdlite_mobiledet.py       # モデル学習スクリプト
│   ├── run_prepare_trash_detection_dataset.sh # データセット準備の実行スクリプト
│   └── tensorboard.sh                     # TensorBoardサーバー起動スクリプト
```

## Docker環境構築とタスクの実行手順

### 1. Dockerイメージのビルド
Dockerfileを使用して環境を構築します。以下のコマンドを実行してイメージをビルドしてください。

```bash
bash build_docker.sh
```

### 2. タスクの実行

各タスクは、構築されたDockerコンテナ内で `bash scripts/xxx.sh` の形式で実行されます。以下の手順に従って各タスクを順番に実行してください。

#### (1) 事前学習済みモデルのダウンロード

事前学習済みのSSDLite MobileDetモデルをダウンロードします。

```bash
bash scripts/download_model.sh
```

#### (2) データセットの準備

trash-detectionデータセットを準備します。

```bash
bash scripts/run_prepare_trash_detection_dataset.sh
```

#### (3) モデルの転移学習

trash-detectionデータセットを使用して、SSDLite MobileDetモデルを転移学習します。

```bash
bash scripts/retrain.sh
```

#### (4) モデルの量子化とTFLite変換

転移学習済みモデルを8ビット量子化し、TFLite形式に変換します。量子化されたモデルはエッジデバイス向けに最適化されています。

```bash
bash scripts/quantize.sh
```

#### (5) TFLiteモデルの解析

TFLiteモデルを解析するために、以下のコマンドを実行します。`--model_path_1`と`--model_path_2`の引数で解析したいモデルのパスを指定してください。

```bash
docker run --rm -v $(pwd):/workspace -w /workspace tensorflow/tensorflow:2.4.1 python3 scripts/analyze_tflite.py --model_path_1
```

## 注意事項
- **NVIDIA GPUが必須**：転移学習や量子化にはNVIDIAドライバとContainer Toolkitが必要です。
- **データセットの準備には時間がかかる場合があります**：trash-detectionデータセットは大規模であり、ダウンロードや整形に時間がかかることがあります。

## 追加スクリプトの詳細

### download_model.sh
事前学習済みのSSDLite MobileDetモデルをダウンロードします。

### run_prepare_trash_detection_dataset.sh
データセット準備スクリプトを呼び出し、trash-detectionデータセットを適切な形式に整形します。

### retrain.sh
`retrain_ssdlite_mobiledet.py` を実行して、trash-detectionデータセットを用いたモデルの転移学習を行います。

### quantize.sh
以下の手順でフローズングラフの生成と8ビット量子化を行い、TFLite形式に変換します。
- 最新の学習済みモデルディレクトリを確認し、エクスポートと量子化を実行します。
- `tflite_convert` コマンドを使用してモデルをTFLiteに変換します。

### tensorboard.sh
学習プロセスを可視化するために、TensorBoardサーバーを起動します。
# Edge Trash Detection with AITRIOS and IMX500

---

このプロジェクトは、TACOデータセットを使用して、**SSDLite MobileDet**の転移学習と8bit量子化された**TFLiteモデル**を生成し、**AITRIOSのIMX500**デバイス上でリアルタイムにゴミ検出を行うことを目的としています。

## プロジェクトの目的

リアルタイム環境モニタリングやゴミ検出をエッジデバイス上で効率的に行うための**軽量モデル**を生成します。量子化モデルはエッジデバイス上で低リソースで動作し、リソース効率を最大化します。

## 必要な前提条件

- Docker
- NVIDIA GPU + NVIDIA Driver
- NVIDIA Container Toolkit
- Git
- Python 3.x

## プロジェクト構成

```
├── data/
│   ├── TACO/                   # TACOデータセットを格納するディレクトリ
│   └── prepare_taco_dataset.py # TACOデータセットの準備スクリプト
│
├── models/
│   ├── ssdlite_mobiledet/       # COCOで事前学習済みのSSDLite MobileDetモデル
│   └── tflite_model/            # 量子化されたTFLite形式のモデル
│
├── scripts/
│   ├── download_model.sh        # 事前学習済みモデルのダウンロードスクリプト
│   ├── retrain.sh               # 転移学習用のスクリプト
│   ├── evaluate.sh              # モデルの評価スクリプト
│   └── quantize_model.py        # モデルを8bit量子化してTFLiteモデルに変換するスクリプト
│
├── Dockerfile                   # Dockerイメージをビルドするための設定ファイル
└── README.md                    # プロジェクトの概要と実行手順

```

## セットアップ手順

### 1. リポジトリのクローン

以下のコマンドでリポジトリをクローンします。

```bash
git clone <リポジトリのURL>
cd edge-trash-detection

```

### 2. Dockerイメージのビルド

Dockerを使用してイメージをビルドします。

```bash
./scripts/build_docker.sh

```

### 3. データセットの準備

TACOデータセットをダウンロードし、必要な形式に整形します。

```bash
python data/prepare_taco_dataset.py

```

### 4. モデルの転移学習

SSDLite MobileDetの事前学習済みモデルをTACOデータセットで転移学習します。

```bash
./scripts/retrain.sh

```

### 5. モデルの量子化

転移学習されたモデルを量子化して、TFLite形式に変換します。

```bash
python scripts/quantize_model.py

```

### 6. モデルの評価

量子化されたモデルの精度を評価し、TACOデータセットでの推論結果を確認します。

```bash
./scripts/evaluate.sh

```

## 注意事項

- **NVIDIA GPUが必須です**：転移学習や量子化のプロセスにはNVIDIAドライバとContainer Toolkitが必要です。
- **データセットの準備に時間がかかる可能性があります**：TACOデータセットは大規模であるため、ダウンロードに時間がかかることがあります。
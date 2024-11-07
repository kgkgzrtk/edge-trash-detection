# ベースイメージをTensorFlow 1.x系に変更
FROM nvcr.io/nvidia/tensorflow:22.04-tf1-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# 必要なパッケージをインストール
RUN apt-get update && \
    apt-get install -y \
    curl \
    unzip \
    wget \
    vim \
    protobuf-compiler \
    build-essential \
    && apt-get clean

# 必要なディレクトリを作成
RUN mkdir -p /tensorflow/models

# TensorFlowモデルのリポジトリをクローンし、指定のコミットにチェックアウト
RUN git clone https://github.com/tensorflow/models.git /tensorflow/models && \
    cd /tensorflow/models && \
    git checkout 420a7253e034a12ae2208e6ec94d3e4936177a53

# 必要なPythonパッケージをインストール
RUN python3 -m pip install --upgrade pip && \
    pip install \
    'Cython==0.29.21' \
    contextlib2 \
    pillow \
    lxml \
    jupyter \
    matplotlib \
    tf_slim 

# pycocotoolsのインストール
RUN git clone --depth 1 https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    make -j8 && \
    pip install . && \
    cp -r pycocotools /tensorflow/models/research && \
    cd ../../ && \
    rm -rf cocoapi

# protocを使用してオブジェクト検出プロトコルをコンパイル
RUN cd /tensorflow/models/research && \
    protoc object_detection/protos/*.proto --python_out=.

# PYTHONPATHを設定
ENV PYTHONPATH $PYTHONPATH:/tensorflow/models/research:/tensorflow/models/research/slim

# 作業ディレクトリを設定
WORKDIR /tensorflow/models/research

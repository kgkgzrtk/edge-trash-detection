# ベースイメージをTensorFlow 1.x系に変更
FROM tensorflow/tensorflow:1.15.5

ARG DEBIAN_FRONTEND=noninteractive

# 必要なパッケージをインストール
RUN apt-get update && \
    apt-get install -y python python-tk git curl unzip

# 必要なディレクトリを作成
RUN mkdir -p /tensorflow/models

# TensorFlowモデルのリポジトリをクローン
RUN git clone https://github.com/tensorflow/models.git && \
    (cd models && git checkout f788046ca876a8820e05b0b48c1fc2e16b0955bc) && \
    cp -r models/research /tensorflow/models/ && \
    rm -rf models

# 必要なPythonパッケージをインストール
RUN pip install Cython && \
    pip install contextlib2 && \
    pip install pillow && \
    pip install lxml && \
    pip install jupyter && \
    pip install matplotlib

# protocのインストール
RUN curl -OL "https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip" && \
    unzip protoc-3.0.0-linux-x86_64.zip -d proto3 && \
    mv proto3/bin/* /usr/local/bin && \
    mv proto3/include/* /usr/local/include && \
    rm -rf proto3 protoc-3.0.0-linux-x86_64.zip

# pycocoapiのインストール
RUN git clone --depth 1 https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    make -j8 && \
    cp -r pycocotools /tensorflow/models/research && \
    cd ../../ && \
    rm -rf cocoapi

# protocを使用してオブジェクト検出プロトコルをコンパイル
RUN cd /tensorflow/models/research && \
    protoc object_detection/protos/*.proto --python_out=.

# PYTHONPATHを設定
ENV PYTHONPATH $PYTHONPATH:/tensorflow/models/research:/tensorflow/models/research/slim

# wgetとエディタをインストール
RUN apt-get update && \
    apt-get install -y wget vim emacs nano

ARG work_dir=/tensorflow/models/research

WORKDIR ${work_dir}

# Prism ML fork の llama.cpp (CUDA) を組み込み、RunPod Serverless のワーカーにする。
# モデル本体はネットワークボリューム (/runpod-volume/models) に置き、初回だけ
# Hugging Face から取得して以降はそこから読み込む。
#
# llama.cpp は自前ビルドせず、Prism フォークの公式リリースバイナリを使う。
# 自前ビルドは 7 アーキテクチャ分を nvcc で焼くため 80 分以上かかり、しかも
# リンク時に libcuda.so.1 を解決できず失敗する (詳細は Dockerfile.build)。
# リリース版は ubuntu-22.04 + CUDA 12.8 でビルドされており、実行イメージと
# 同じ glibc なのでそのまま動く。
FROM debian:12-slim AS fetch

# モデルカードが検証済みとしているリビジョンのリリースに固定する。
ARG LLAMA_TAG=prism-b10709-9a9394a
ARG LLAMA_TARBALL_SHA256=8aec67eb023b251712c7e6490f367b5671bf587eced1436a9b85f4a90c3b7d3d

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# 同梱の共有ライブラリは RUNPATH=$ORIGIN なので、展開先にまとめて置けば解決できる。
RUN curl -fsSL -o /tmp/llama.tar.gz \
      "https://github.com/PrismML-Eng/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-linux-cuda-12.8-x64.tar.gz" \
 && echo "${LLAMA_TARBALL_SHA256}  /tmp/llama.tar.gz" | sha256sum -c - \
 && mkdir -p /opt/llama \
 && tar -xzf /tmp/llama.tar.gz -C /opt/llama --strip-components=1

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# LD_LIBRARY_PATH は念のため。同梱ライブラリは RUNPATH=$ORIGIN で解決される。
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    LLAMA_SERVER=/opt/llama/llama-server \
    LD_LIBRARY_PATH=/opt/llama

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip libgomp1 ca-certificates aria2 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=fetch /opt/llama/ /opt/llama/

RUN pip3 install --no-cache-dir runpod huggingface_hub hf_transfer hf_xet

COPY handler.py /opt/handler.py
WORKDIR /opt
CMD ["python3", "-u", "/opt/handler.py"]

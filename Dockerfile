# Prism ML fork の llama.cpp (CUDA) を組み込み、RunPod Serverless のワーカーにする。
#
# モデルをどこから拾うかは環境変数だけで切り替えられる (handler.py の docstring 参照):
#   ボリュームを付ける   : /runpod-volume/models をマスターに使う (DC に固定される)
#   ボリュームを付けない : S3 互換 API からコンテナディスクへ落とす (DC を選べる)
#   どちらも無い         : Hugging Face から落とす (コールドスタートのたびに取得)
#
# OrcaBonsai (公開の重みはそのまま、振る舞いだけ実行時に変える rank-1 LoRA) を
# 使うときの env。Pod で試した組み合わせをそのまま Serverless に持ち込める:
#   MODEL_REPO=prism-ml/Ternary-Bonsai-2-27B-gguf
#   MODEL_FILE=Ternary-Bonsai-2-27B-PTQ1_0.gguf   (PTQ1_0 が LoRA の実測済み)
#   LORA_FILE=bonsai-abliterate-lora.gguf         (9.7MB。空なら LoRA 無し)
#   LORA_SCALE=1                                  (2 で頑固なプロンプトも折る)
#   LLAMA_EXTRA_ARGS=--jinja --temp 0.7
# ベースと LoRA は 1 回そろえれば S3 かボリュームに退避され、次回から速い。
# 同じ取得手順は tools/orcarouter_setup.py にまとめてある (Pod でも使う)。
#
# S3 の読み書きは boto3 に任せる。6.7GB 級の GGUF は単発 PUT の 5GiB を超えるので
# multipart が要り、boto3 の download_file / upload_file がそれを面倒を見る。
# aws CLI を入れるより軽いので、こちらを使う。
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

# boto3 は S3 互換 API 用 (multipart の download / upload)。
RUN pip3 install --no-cache-dir runpod huggingface_hub hf_transfer hf_xet boto3

COPY handler.py /opt/handler.py
WORKDIR /opt
CMD ["python3", "-u", "/opt/handler.py"]

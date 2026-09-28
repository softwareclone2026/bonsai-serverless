# bonsai-serverless

RunPod Serverless 用の [Prism ML llama.cpp](https://github.com/PrismML-Eng/llama.cpp) ワーカー。
uncensored な Bonsai 2 27B（ternary / PQ2_0）をネットワークボリュームから読み込み、
llama-server を起動して推論を中継する。

## 構成

| 項目 | 内容 |
|---|---|
| ベース | `nvidia/cuda:12.8.1-runtime-ubuntu22.04` |
| llama.cpp | `PrismML-Eng/llama.cpp` の公式リリース `prism-b10709-9a9394a`（PTQ1_0 / PQ2_0 対応） |
| モデル | `OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-GGUF` の PQ2_0 |
| 保存先 | ネットワークボリューム `/runpod-volume/models` |
| 起動 | 初回リクエストでモデル取得 → `llama-server` 起動 → API 中継 |

## 環境変数

| 変数 | 既定値 | 用途 |
|---|---|---|
| `MODEL_REPO` | `OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-GGUF` | HF リポジトリ |
| `MODEL_FILE` | `Ternary-Bonsai-2-27B-Uncensored-Heretic-PQ2_0.gguf` | 取得するファイル |
| `CTX_SIZE` | `16384` | コンテキスト長 |
| `GPU_LAYERS` | `99` | GPU に載せる層数 |
| `LLAMA_EXTRA_ARGS` | `--jinja --reasoning on --reasoning-effort medium --temp 1.0 --top-p 0.95 --top-k 20` | 追加起動引数 |
| `MAX_TOKENS` | `512` | 既定の最大生成トークン数 |

## llama.cpp の入手方法

既定の [Dockerfile](Dockerfile) は、Prism フォークのリリースバイナリ
（`llama-prism-<tag>-bin-linux-cuda-12.8-x64.tar.gz`）を取得して配置する。
タグと SHA256 は Dockerfile の `LLAMA_TAG` / `LLAMA_TARBALL_SHA256` で固定しており、
ビルドは数分で終わる。

リリース版の `libggml-cuda.so` は次の GPU を対象にビルドされている
（既定の `CMAKE_CUDA_ARCHITECTURES` のため）。

| 世代 | 例 | 対応 |
|---|---|---|
| sm_75 / 80 | T4, A100 | PTX から JIT（初回起動が少し遅い） |
| sm_86 / 89 | A6000, RTX 3090 / 4090, L4, L40S | ネイティブ |
| sm_90 | H100, H200 | PTX から JIT |
| sm_120 | RTX PRO 6000 Blackwell, RTX 5090 | ネイティブ（120a） |
| sm_100 | B200, B300 | 非対応 |

B200 など sm_100 を使いたい場合や、別のリビジョンで自前ビルドしたい場合は
[Dockerfile.build](Dockerfile.build) を使う。nvcc で全アーキテクチャを焼くため
GitHub のランナーで 80 分以上かかる。

## 使い方

```sh
docker build -t bonsai-serverless .
```

RunPod 側の作成は親リポジトリの `scripts/deploy_bonsai_serverless.py` が行う。

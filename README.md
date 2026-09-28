# bonsai-serverless

RunPod Serverless 用の [Prism ML llama.cpp](https://github.com/PrismML-Eng/llama.cpp) ワーカー。
uncensored な Bonsai 2 27B（ternary / PQ2_0）をネットワークボリュームから読み込み、
llama-server を起動して推論を中継する。

## 構成

| 項目 | 内容 |
|---|---|
| ベース | `nvidia/cuda:12.8.1`（build: devel / runtime: runtime） |
| llama.cpp | `PrismML-Eng/llama.cpp` @ `9a9394a8`（PTQ1_0 / PQ2_0 対応） |
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

## 使い方

```sh
docker build -t bonsai-serverless .
```

RunPod 側の作成は親リポジトリの `scripts/deploy_bonsai_serverless.py` が行う。

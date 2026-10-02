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
| `MAX_TOKENS` | `512` | 最大生成トークン数。**env を書いたときだけ下限としても働く**（後述） |
| `MAX_NEW_TOKENS` | (空) | `MAX_TOKENS` の別名 |
| `MAX_TOKENS_OVERRIDE` / `FORCE_MAX_TOKENS` | (空) | 設定するとリクエストより優先する上限 |
| `MODEL_SOURCE` | `auto` | `wasabi` で rclone キャッシュ（Wasabi）を最優先で使う。未設定なら従来どおり |
| `MODEL_CACHE` | `wasabi_crypt3:/vidgen_cache` | rclone のリモートパス（`WAN22_MODEL_CACHE` / `RCLONE_REMOTE` でも可） |
| `RCLONE_CONFIG_B64` | (空) | rclone.conf を base64 にしたもの（**秘密**。env でのみ渡す） |
| `CACHE_MODEL_FILE` | `llm/llm.gguf` | キャッシュ側のモデル名（リモートからの相対） |
| `CACHE_LORA_FILE` | `llm/llm-lora.gguf` | キャッシュ側の LoRA 名 |
| `RCLONE_BIN` | (空) | rclone の実体。未指定なら PATH → 無ければ取得 |
| `MODEL_REVISION` | `main` | 取得するリビジョン |
| `MODEL_FILES` | (空) | 追加で取得するファイル。分割 GGUF をカンマ区切りで並べる |
| `DOWNLOAD_CONNECTIONS` | `16` | 1 ファイルあたりの並列接続数 |
| `DOWNLOAD_JOBS` | `4` | 複数ファイルを同時に落とす本数 |
| `DOWNLOAD_BACKEND` | `auto` | `auto` / `aria2` / `hf` |
| `MODEL_MIN_BYTES` | `1000000000` | 先頭ファイルを取得済みとみなす最小サイズ |

## 生成の上限（`MAX_TOKENS`）

呼び出し側の `japanese-video-gateway/tools/bonsai27b` は**常に** `-n`（既定 1024）を
`input.max_tokens` として送る。そのため worker 側で `MAX_TOKENS=2048` にしても、
リクエストの 1024 が優先されて 1024 で切れていた。

`MAX_TOKENS` を**明示したとき**は、その値を下限としても使う（リクエストが小さければ
引き上げる）。未設定ならリクエストの値をそのまま使うので、従来の挙動は変わらない。

| 設定 | 実際に渡る `max_tokens` |
|---|---|
| 未設定 + リクエスト 1024 | 1024（従来どおり） |
| `MAX_TOKENS=2048` + リクエスト 1024 | 2048（引き上げる） |
| `MAX_TOKENS=2048` + リクエスト 4096 | 4096 |
| `MAX_TOKENS_OVERRIDE=2048` + リクエスト 1024 | 2048（常に上書き） |

warmup の応答（`max_tokens`）と、ジョブの応答（`max_tokens`）で実際の値を確認できる。

## Wasabi（rclone 暗号化キャッシュ）から取得

S3 互換 API が 401 などで使えない場合の取得元。**既存の取得経路はそのまま**で、
`MODEL_SOURCE=wasabi` を設定したときだけ、ボリュームの次・S3 の前に試す。

渡し方は `tools/vast_movie.py` と同じ。rclone の設定を base64 にして env で渡し、
worker が `~/.config/rclone/rclone.conf` に書き出す。crypt のパスワードは conf の中に
obscure 済みで入っているため、**新しい秘密の形式は増やさない**。秘密は必ず env から
読み、コードには書かない（ログにも応答にも出さず、有無だけ出す）。

```sh
# 手元の rclone.conf を base64 にしてテンプレート env に渡す（値は例示）
RCLONE_CONFIG_B64=<REDACTED>
MODEL_SOURCE=wasabi
MODEL_CACHE=wasabi_crypt3:/vidgen_cache
CACHE_MODEL_FILE=llm/llm.gguf
CACHE_LORA_FILE=llm/llm-lora.gguf
```

`MODEL_SOURCE` 未設定なら rclone を一切見ない（後方互換）。S3 の失敗は
`InvalidAccessKeyId` / `SignatureDoesNotMatch` / `AccessDenied` / `NoSuchBucket` などの
Code・Message・HTTP ステータス・region・RequestId と、S3 系 env の有無をログに出す
（値は出さない）。

## ComfyUI について

**このワーカーは ComfyUI を起動しない。** イメージに入るのは Prism 版 `llama-server` と
`handler.py` だけで、`handler.py` はモデルを用意して `llama-server` を起動し、
RunPod Serverless のジョブ API で推論を中継する。健康判定は llama-server の
`/health`（`{"status":"ok"}`）を見る。したがって、呼び出し側が ComfyUI の
`/system_stats` を見て `comfy=true` を待つ健康チェックは、このワーカーには当てはまらない。

## モデル取得の高速化

初回リクエスト時のモデル取得は `aria2c` で行う。1 ファイルを既定 16 接続で分割受信し、
途中で切れても `.part` から再開する。完了したファイルだけを本体名へ移すため、
途中のファイルを取得済みと誤認しない。

`aria2c` が無い・失敗した場合は `huggingface_hub` に自動で切り替える。このとき
`hf_transfer` / `hf_xet` が有効ならそちらの高速転送を使う。分割 GGUF のように
複数ファイルを取る場合は `MODEL_FILES` に列挙すると `DOWNLOAD_JOBS` 本まで並列に落とす。

```sh
# 例: 並列 24 接続にする
DOWNLOAD_CONNECTIONS=24
```

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

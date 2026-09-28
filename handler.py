#!/usr/bin/env python3
"""RunPod Serverless ワーカー: Prism fork の llama-server を起動して推論を中継する。

初回リクエストでモデルをネットワークボリュームへ取得し (2回目以降は再利用)、
llama-server を起動して HTTP で受け取ったジョブを /v1/chat/completions へ流す。
モデルの取得は aria2c の並列ダウンロード (既定 16 接続、途中再開あり) で行い、
aria2c が無い環境では huggingface_hub に切り替える。
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import runpod

# --------------------------------------------------------------------------- #
# 設定 (RunPod の環境変数で上書きできる)
# --------------------------------------------------------------------------- #
VOLUME_DIR = "/runpod-volume"
MODEL_REPO = os.environ.get(
    "MODEL_REPO", "OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-GGUF")
MODEL_FILE = os.environ.get(
    "MODEL_FILE", "Ternary-Bonsai-2-27B-Uncensored-Heretic-PQ2_0.gguf")

if os.path.isdir(VOLUME_DIR):
    MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(VOLUME_DIR, "models"))
else:
    # ネットワークボリューム未接続だとコールドスタートごとに再取得になる。
    MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/models")
    print(f"[worker] 警告: {VOLUME_DIR} がありません。{MODEL_DIR} を使います"
          " (毎回ダウンロードが走ります)", flush=True)

# 取得方法: 既定は aria2c の並列ダウンロード。無い環境では huggingface_hub を使う。
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
MODEL_FILES = [name for name in re.split(r"[,\s]+", os.environ.get("MODEL_FILES", "")) if name]
MODEL_FILES = MODEL_FILES or [MODEL_FILE]
DOWNLOAD_CONNECTIONS = max(1, int(os.environ.get("DOWNLOAD_CONNECTIONS", "16") or 16))
DOWNLOAD_JOBS = max(1, int(os.environ.get("DOWNLOAD_JOBS", "4") or 4))
DOWNLOAD_BACKEND = os.environ.get("DOWNLOAD_BACKEND", "auto").strip().lower()
MIN_MODEL_BYTES = int(os.environ.get("MODEL_MIN_BYTES", "1000000000") or 0)
HF_TOKEN = os.environ.get("HF_TOKEN", "")

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", "/opt/llama/bin/llama-server")
PORT = int(os.environ.get("WORKER_PORT", "8080"))
CTX_SIZE = os.environ.get("CTX_SIZE", "16384")
GPU_LAYERS = os.environ.get("GPU_LAYERS", "99")
READY_TIMEOUT = float(os.environ.get("READY_TIMEOUT", "900"))
# モデル作者推奨のサンプリング設定と thinking モード。
# 空文字を渡すと追加引数なしで起動する。
DEFAULT_EXTRA = "--jinja --reasoning on --reasoning-effort medium --temp 1.0 --top-p 0.95 --top-k 20"
EXTRA_ARGS = os.environ.get("LLAMA_EXTRA_ARGS", DEFAULT_EXTRA).split()
MAX_TOKENS_DEFAULT = int(os.environ.get("MAX_TOKENS", "512"))

_proc = None
_lock = threading.Lock()
_started_at = None


def log(message: str) -> None:
    print(f"[worker] {message}", flush=True)


# --------------------------------------------------------------------------- #
# モデルの準備と llama-server の起動
# --------------------------------------------------------------------------- #
def model_is_ready() -> bool:
    """必要なファイルがそろっているか。途中の .part は完了とみなさない。"""
    for index, name in enumerate(MODEL_FILES):
        path = os.path.join(MODEL_DIR, name)
        if not os.path.isfile(path):
            return False
        size = os.path.getsize(path)
        # 分割ファイルは先頭だけ下限を課す (続きは小さくてもよい)。
        if size <= 0 or (index == 0 and size < MIN_MODEL_BYTES):
            return False
    return True


def missing_files() -> list:
    todo = []
    for index, name in enumerate(MODEL_FILES):
        path = os.path.join(MODEL_DIR, name)
        if not os.path.isfile(path):
            todo.append(name)
            continue
        size = os.path.getsize(path)
        if size <= 0 or (index == 0 and size < MIN_MODEL_BYTES):
            todo.append(name)
    return todo


def file_url(name: str) -> str:
    override = os.environ.get("MODEL_URL")
    if override and len(MODEL_FILES) == 1:
        return override
    from huggingface_hub import hf_hub_url

    return hf_hub_url(repo_id=MODEL_REPO, filename=name, revision=MODEL_REVISION)


def finalize(files) -> None:
    """aria2c が残した .part を本体の名前へ移す。"""
    for name in files:
        part = os.path.join(MODEL_DIR, name) + ".part"
        dest = os.path.join(MODEL_DIR, name)
        if os.path.isfile(part):
            os.replace(part, dest)


def aria2_download(files) -> None:
    """aria2c で取得する。1 ファイルは複数接続、複数ファイルは並列に落とす。"""
    jobs = max(1, min(len(files), DOWNLOAD_JOBS))
    per_file = max(1, DOWNLOAD_CONNECTIONS // jobs) if jobs > 1 else DOWNLOAD_CONNECTIONS
    listing = []
    for name in files:
        dest = os.path.join(MODEL_DIR, name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        part = dest + ".part"
        listing += [file_url(name),
                    "  dir=" + os.path.dirname(part),
                    "  out=" + os.path.basename(part)]
    command = [
        "aria2c",
        "--continue=true", "--auto-file-renaming=false", "--allow-overwrite=true",
        "--file-allocation=none",
        "-x" + str(per_file), "-s" + str(per_file), "-k1M", "-j" + str(jobs),
        "--max-tries=10", "--retry-wait=5", "--connect-timeout=20", "--timeout=60",
        "--summary-interval=15", "--console-log-level=warn", "--show-console-readout=false",
        "--input-file=-",
    ]
    if HF_TOKEN:
        command.append("--header=Authorization: Bearer " + HF_TOKEN)
    log(f"aria2c 並列取得: {len(files)} ファイル / ファイルあたり {per_file} 接続")
    result = subprocess.run(command, input="\n".join(listing) + "\n", text=True,
                            stdout=sys.stdout, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"aria2c が終了コード {result.returncode} で失敗しました")


def hf_download(files) -> None:
    from huggingface_hub import hf_hub_download

    for name in files:
        hf_hub_download(repo_id=MODEL_REPO, filename=name, revision=MODEL_REVISION,
                        local_dir=MODEL_DIR, token=HF_TOKEN or None)


def ensure_model() -> None:
    if model_is_ready():
        total = sum(os.path.getsize(os.path.join(MODEL_DIR, name)) for name in MODEL_FILES)
        log(f"モデルは取得済み: {MODEL_DIR} ({total / 1e9:.2f} GB)")
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    todo = missing_files()
    log(f"Hugging Face から取得: {MODEL_REPO} / {', '.join(todo)}")
    started = time.time()
    use_aria2 = DOWNLOAD_BACKEND in ("auto", "aria2", "aria2c") and shutil.which("aria2c")
    if use_aria2:
        try:
            aria2_download(todo)
            finalize(todo)
        except Exception as error:
            log(f"aria2c を続けられません ({error})。huggingface_hub で取り直します")
            hf_download(todo)
            finalize(todo)
    else:
        if DOWNLOAD_BACKEND in ("aria2", "aria2c"):
            log("aria2c が見つかりません。huggingface_hub を使います")
        hf_download(todo)
        finalize(todo)
    if not model_is_ready():
        raise RuntimeError(f"モデルを取得できませんでした: {', '.join(todo)}")
    log(f"取得完了 ({time.time() - started:.1f} 秒)")


def wait_until_healthy(timeout: float) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{PORT}/health"
    while time.time() < deadline:
        if _proc.poll() is not None:
            raise RuntimeError(f"llama-server が異常終了しました (code={_proc.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if json.loads(response.read().decode()).get("status") == "ok":
                    return
        except Exception:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"llama-server が {timeout} 秒以内に応答しませんでした")


def start_server() -> None:
    """ワーカーごとに一度だけ起動する。2回目以降は既存プロセスを再利用する。"""
    global _proc, _started_at
    with _lock:
        if _proc is not None and _proc.poll() is None:
            return
        ensure_model()
        command = [
            LLAMA_SERVER, "-m", MODEL_PATH,
            "-ngl", GPU_LAYERS, "-c", CTX_SIZE,
            "--host", "127.0.0.1", "--port", str(PORT),
            *EXTRA_ARGS,
        ]
        log("起動: " + " ".join(command))
        started = time.time()
        _proc = subprocess.Popen(command, stdout=sys.stdout, stderr=subprocess.STDOUT)
        try:
            wait_until_healthy(READY_TIMEOUT)
        except Exception:
            if _proc and _proc.poll() is None:
                _proc.kill()
            raise
        _started_at = time.time()
        log(f"準備完了 ({_started_at - started:.1f} 秒)")


# --------------------------------------------------------------------------- #
# リクエスト処理
# --------------------------------------------------------------------------- #
def build_payload(job_input: dict) -> dict:
    if "messages" in job_input:
        messages = job_input["messages"]
    elif "prompt" in job_input:
        messages = [{"role": "user", "content": job_input["prompt"]}]
    else:
        raise ValueError("input には 'messages' か 'prompt' が必要です")

    payload = {
        "messages": messages,
        "max_tokens": int(job_input.get("max_tokens", MAX_TOKENS_DEFAULT)),
        "temperature": float(job_input.get("temperature", 1.0)),
        "top_p": float(job_input.get("top_p", 0.95)),
        "top_k": int(job_input.get("top_k", 20)),
        "stream": False,
    }
    if job_input.get("system"):
        payload["messages"] = [{"role": "system", "content": job_input["system"]}, *messages]
    if job_input.get("stop"):
        payload["stop"] = job_input["stop"]
    return payload


def post_chat(payload: dict) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=float(os.environ.get("GEN_TIMEOUT", "600"))) as response:
        return json.loads(response.read().decode())


def handler(job: dict) -> dict:
    job_input = job.get("input") or {}

    # {"input": {"warmup": true}} でモデルだけ読み込む（事前ウォームアップ用）。
    if job_input.get("warmup"):
        start_server()
        return {"status": "ready", "model": MODEL_FILE,
                "gpu": os.environ.get("RUNPOD_GPU_TYPE", "unknown-until-first-job")}

    start_server()
    payload = build_payload(job_input)
    started = time.time()
    try:
        result = post_chat(payload)
    except urllib.error.HTTPError as error:
        return {"error": f"llama-server HTTP {error.code}",
                "detail": error.read().decode(errors="replace")[:800]}

    choice = result["choices"][0]["message"]
    return {
        "text": choice.get("content", ""),
        "reasoning": choice.get("reasoning_content"),
        "finish_reason": result["choices"][0].get("finish_reason"),
        "usage": result.get("usage"),
        "elapsed_seconds": round(time.time() - started, 2),
        "timings": result.get("timings"),
        "model": MODEL_FILE,
    }


if __name__ == "__main__":
    log(f"ワーカー開始: model={MODEL_FILE} ctx={CTX_SIZE} port={PORT}")
    runpod.serverless.start({"handler": handler})

#!/usr/bin/env python3
"""RunPod Serverless ワーカー: Prism fork の llama-server を起動して推論を中継する。

モデルは「コンテナディスク = 使い捨ての作業コピー、外にマスターを 1 つ」として扱い、
起動のたびに次の順で探して最初に見つかったものを使う:

1. コンテナディスク (:data:`LOCAL_PATH`) — 同じワーカーでの 2 回目以降
2. マウントされたボリューム (:data:`CACHE_PATH`) — global Volume を付けた場合
3. S3 互換 API (Network Volume) — ボリュームを**マウントせず**使う場合
4. Hugging Face (:data:`MODEL_REPO`)

3 と 4 で取ってきたときは、使えるマスター (ボリューム or S3) へ**退避**して、
次回のコールドスタートで 1 か 2 から拾えるようにする。コンテナディスクは起動の
たびに捨てられるので、残るのはマスターだけ。llama-server には常にコンテナディスク
側のコピーを渡す。

S3 を使うときはボリュームをマウントしないので、ワーカーがデータセンターに固定
されない (代わりにコールドスタートのたびにダウンロードが走る)。マウントがあれば
そちらを優先するので、ボリュームの有無だけで A/B/C を切り替えられる。

Hugging Face からの取得 (4) は aria2c の並列ダウンロード (既定 16 接続、途中再開
あり) で行う。10GB 級の GGUF を 1 本の HTTP で落とすと桁違いに遅いため。
aria2c が無い環境や失敗時は huggingface_hub に切り替える (``DOWNLOAD_BACKEND``)。

OrcaBonsai のような LoRA を当てるときは ``LORA_FILE`` を設定する (既定は空 =
LoRA 無し)。アダプタは 9.7MB なのでボリュームや S3 を介さず、コンテナディスクへ
素の HTTP で取り直す。``LORA_SCALE`` で強度を変えられる (1=厳密な射影 / 2=頑固な
プロンプトも折る / 3 以上は崩れる)。
"""
import json
import os
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
#: ボリュームのマウント先 (global Volume / Network Volume のどちらでも同じ)。
VOLUME_DIR = os.environ.get("VOLUME_DIR", "/runpod-volume")
#: ボリューム上のマスター。マウントされていればここから拾う。
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(VOLUME_DIR, "models"))
#: コンテナディスク側の作業コピー。コールドスタートのたびに捨てられる。
#: 以前の MODEL_DIR でも指定できる。
LOCAL_DIR = os.environ.get("LOCAL_DIR") or os.environ.get("MODEL_DIR") or "/opt/models"
#: 0 / false / no / off でボリュームとのコピーを止める。
VOLUME_CACHE = os.environ.get("VOLUME_CACHE", "1").strip().lower() not in (
    "0", "false", "no", "off")
#: これ未満は「途中までしか無い」とみなす (既定 1GB)。MODEL_MIN_BYTES でも可。
MIN_MODEL_BYTES = int(os.environ.get("MODEL_MIN_BYTES")
                      or os.environ.get("MIN_MODEL_BYTES") or "1000000000")

MODEL_REPO = os.environ.get(
    "MODEL_REPO", "OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-GGUF")
MODEL_FILE = os.environ.get(
    "MODEL_FILE", "Ternary-Bonsai-2-27B-Uncensored-Heretic-PQ2_0.gguf")
#: Hugging Face のリビジョン (ブランチ / タグ / コミット)。
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main").strip() or "main"
#: 取得先を丸ごと差し替える (HF 以外の直リンクを使うとき)。
MODEL_URL = os.environ.get("MODEL_URL", "").strip()
#: private リポジトリ用の HF トークン。
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
#: 取得方法: auto (aria2c があれば使う) / aria2 / hf。
DOWNLOAD_BACKEND = os.environ.get("DOWNLOAD_BACKEND", "auto").strip().lower()
#: aria2c の接続数 (1 ファイルを何本に分けて落とすか)。
DOWNLOAD_CONNECTIONS = max(1, int(os.environ.get("DOWNLOAD_CONNECTIONS", "16") or 16))
#: 分割 GGUF 用の MODEL_FILES は今のところ未対応 (先頭だけ落とす)。
MODEL_FILES = [name for name in
               (os.environ.get("MODEL_FILES") or "").replace(",", " ").split() if name]

CACHE_PATH = os.path.join(CACHE_DIR, MODEL_FILE)
LOCAL_PATH = os.path.join(LOCAL_DIR, MODEL_FILE)

# --------------------------------------------------------------------------- #
# LoRA (任意)。OrcaBonsai は「公開の重みはそのまま、振る舞いだけ実行時に変える」
# rank-1 アダプタとして配る。llama.cpp は LoRA を重みに焼き込まず計算グラフに
# 足すだけなので、1.7 ビットの重みでも消えない (丸め込まれる心配がない)。
# --------------------------------------------------------------------------- #
#: ファイル名。空にすると LoRA 無しで起動する。
LORA_FILE = os.environ.get("LORA_FILE", "").strip()
#: 取得先。既定は OrcaBonsai のリポジトリに同梱の GGUF (GitHub の raw)。
#: Hugging Face ではないので huggingface_hub は使わず、素の HTTP で取る。
LORA_URL = os.environ.get(
    "LORA_URL",
    "https://github.com/Continuum-AI-Corp/OrcaBonsai-27B-Uncensored/"
    "raw/main/gguf/bonsai-abliterate-lora.gguf").strip()
#: 期待する sha256。空なら検証しない (自己責任)。
LORA_SHA256 = os.environ.get(
    "LORA_SHA256",
    "f1669534803d340a496015f5c45125f3437b4d13ec764f40e34488ce83967f42").strip()
#: 1 が厳密な射影。2 で頑固なプロンプトも折れる。3 以上は過剰射影で崩れる。
LORA_SCALE = os.environ.get("LORA_SCALE", "1").strip()
#: これ未満は「途中までしか無い」とみなす (LoRA は 9.7MB なので小さくてよい)。
LORA_MIN_BYTES = int(os.environ.get("LORA_MIN_BYTES", "1000000"))

LOCAL_LORA = os.path.join(LOCAL_DIR, LORA_FILE) if LORA_FILE else ""
CACHE_LORA = os.path.join(CACHE_DIR, LORA_FILE) if LORA_FILE else ""

# S3 互換 API (Network Volume を、マウントせずに使う)。RUNPOD_S3_* でも可。
S3_ENDPOINT = (os.environ.get("S3_ENDPOINT") or os.environ.get("RUNPOD_S3_ENDPOINT") or "").strip()
S3_REGION = (os.environ.get("S3_REGION") or os.environ.get("RUNPOD_S3_REGION") or "").strip()
S3_BUCKET = (os.environ.get("S3_BUCKET") or os.environ.get("RUNPOD_S3_BUCKET") or "").strip()
S3_ACCESS_KEY = (os.environ.get("S3_ACCESS_KEY")
                 or os.environ.get("RUNPOD_S3_ACCESS_KEY") or "").strip()
S3_SECRET_KEY = (os.environ.get("S3_SECRET_KEY")
                 or os.environ.get("RUNPOD_S3_SECRET_KEY") or "").strip()
#: ボリューム上のオブジェクト名。既定は models/<MODEL_FILE>。
REMOTE_KEY = os.environ.get("REMOTE_KEY", "").strip()
#: 0 / false / no / off で S3 への退避を止める。
REMOTE_CACHE = os.environ.get("REMOTE_CACHE", "1").strip().lower() not in (
    "0", "false", "no", "off")

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
#: 直近でモデルをどこから用意したか ("local" / "volume" / "s3" / "huggingface")。
_model_source = "unknown"


def log(message: str) -> None:
    print(f"[worker] {message}", flush=True)


# --------------------------------------------------------------------------- #
# マスターの場所
# --------------------------------------------------------------------------- #
def model_ready(path: str) -> bool:
    """ファイルが存在して、途中で切れていない大きさか。"""
    try:
        return os.path.isfile(path) and os.path.getsize(path) >= MIN_MODEL_BYTES
    except OSError:
        return False


def volume_cache_enabled() -> bool:
    """マウントされたボリュームをマスターに使うか。"""
    if not VOLUME_CACHE:
        return False
    if os.path.abspath(CACHE_PATH) == os.path.abspath(LOCAL_PATH):
        return False
    return os.path.isdir(VOLUME_DIR)


def s3_enabled() -> bool:
    return bool(S3_ENDPOINT and S3_REGION and S3_BUCKET and S3_ACCESS_KEY and S3_SECRET_KEY)


def remote_key() -> str:
    return REMOTE_KEY or f"models/{MODEL_FILE}"


def s3_client():
    """S3 互換 API 用の boto3 クライアント。

    RunPod の endpoint は ``https://s3api-<dc>.runpod.io/`` でパスに bucket を
    持つため、**path スタイル**を明示する (既定の virtual-hosted では名前解決に失敗する)。
    """
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def s3_object_state() -> tuple[str, int, str]:
    """``(状態, 大きさ, 補足)`` を返す。状態は ok / missing / error / disabled。"""
    if not s3_enabled():
        return "disabled", 0, "S3 の設定がありません"
    try:
        import boto3  # noqa: F401
        from botocore.exceptions import ClientError
    except ImportError as error:
        return "error", 0, f"boto3 がありません ({error})"
    try:
        response = s3_client().head_object(Bucket=S3_BUCKET, Key=remote_key())
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return "missing", 0, "S3 にまだありません"
        return "error", 0, code or str(error)
    except Exception as error:  # 到達不能・DNS 失敗などもここで拾って報告する
        return "error", 0, f"{type(error).__name__}: {error}"
    return "ok", int(response.get("ContentLength") or 0), ""


# --------------------------------------------------------------------------- #
# モデルの準備 (ローカル / ボリューム / S3 / Hugging Face)
# --------------------------------------------------------------------------- #
def copy_model(source: str, destination: str, label: str) -> bool:
    """モデルをコピーする (退避 / 復元)。失敗しても推論は続けられるよう bool を返す。

    global Volume は atomic rename を持たないので、途中で切れたファイルは
    大きさの検査で弾いて次回やり直す。
    """
    size = os.path.getsize(source)
    parent = os.path.dirname(destination)
    started = time.time()
    try:
        if parent:
            os.makedirs(parent, exist_ok=True)
        shutil.copyfile(source, destination)
    except OSError as error:
        log(f"警告: {label} に失敗しました ({error})。続行します")
        return False
    seconds = max(time.time() - started, 0.001)
    log(f"{label} 完了: {destination} ({size / 1e9:.2f} GB / {seconds:.1f} 秒"
        f" / {size / 1e6 / seconds:.0f} MB/s)")
    return True


def s3_download(destination: str) -> bool:
    parent = os.path.dirname(destination) or "."
    os.makedirs(parent, exist_ok=True)
    log(f"S3 から取得: s3://{S3_BUCKET}/{remote_key()} -> {destination}")
    started = time.time()
    try:
        s3_client().download_file(S3_BUCKET, remote_key(), destination)
    except Exception as error:
        log(f"警告: S3 からの取得に失敗しました ({error})")
        return False
    log(f"S3 からの取得完了 ({os.path.getsize(destination) / 1e9:.2f} GB / "
        f"{time.time() - started:.1f} 秒)")
    return True


def s3_upload(source: str) -> bool:
    """S3 へ退避する。6.7GB 級は 5GiB を超えるので multipart になる (boto3 が面倒を見る)。"""
    size = os.path.getsize(source)
    log(f"S3 へ退避: {source} -> s3://{S3_BUCKET}/{remote_key()} ({size / 1e9:.2f} GB)")
    started = time.time()
    try:
        s3_client().upload_file(source, S3_BUCKET, remote_key())
    except Exception as error:
        log(f"警告: S3 への退避に失敗しました ({error})")
        return False
    log(f"S3 への退避完了 ({time.time() - started:.1f} 秒)")
    return True


def evacuate(local_path: str) -> str:
    """ローカルのモデルを使えるマスターへ書き戻し、書き戻し先を返す。"""
    if volume_cache_enabled():
        return "volume" if copy_model(local_path, CACHE_PATH, "ボリュームへの退避") else "none"
    if REMOTE_CACHE and s3_enabled():
        return "s3" if s3_upload(local_path) else "none"
    return "none"


def model_url() -> str:
    """モデルの直リンク。``MODEL_URL`` があればそれを優先する。"""
    if MODEL_URL:
        return MODEL_URL
    from huggingface_hub import hf_hub_url

    return hf_hub_url(repo_id=MODEL_REPO, filename=MODEL_FILE, revision=MODEL_REVISION)


def aria2_available() -> bool:
    """aria2c を使うか (使い捨ての取得経路なので、無ければ HF に落とす)。"""
    if DOWNLOAD_BACKEND not in ("auto", "aria2", "aria2c"):
        return False
    if not shutil.which("aria2c"):
        if DOWNLOAD_BACKEND != "auto":
            log("aria2c が見つかりません。huggingface_hub を使います")
        return False
    return True


def aria2_download(destination: str) -> None:
    """aria2c で並列に落とす (``.part`` へ書いてから本体の名前に移す)。

    6.7GB 級の GGUF を 1 本の HTTP で落とすとレート制限もあって非常に遅い。
    HF は Range に対応しているので、16 本に分けるだけで桁が変わる。
    ``--continue=true`` なので、途中で死んでも次回は続きから進む。
    """
    part = destination + ".part"
    folder = os.path.dirname(part) or "."
    os.makedirs(folder, exist_ok=True)
    url = model_url()
    command = [
        "aria2c",
        "--continue=true", "--auto-file-renaming=false", "--allow-overwrite=true",
        "--file-allocation=none", "--max-tries=10", "--retry-wait=5",
        "--connect-timeout=20", "--timeout=60", "--summary-interval=30",
        # 進捗は 30 秒ごとに 1 行だけ出す (10GB を無言で待たせない)。
        "--console-log-level=notice", "--show-console-readout=false",
        "-x" + str(DOWNLOAD_CONNECTIONS), "-s" + str(DOWNLOAD_CONNECTIONS), "-k1M",
        "--dir=" + folder, "--out=" + os.path.basename(part), url,
    ]
    if HF_TOKEN:
        command.insert(-1, "--header=Authorization: Bearer " + HF_TOKEN)
    log(f"aria2c で取得 ({DOWNLOAD_CONNECTIONS} 接続): {url}")
    result = subprocess.run(command, stdout=sys.stdout, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"aria2c が終了コード {result.returncode} で失敗しました")
    if os.path.isfile(part):
        os.replace(part, destination)


def hf_download(destination: str) -> None:
    """huggingface_hub で落とす (aria2c が使えないときの逃げ道)。"""
    parent = os.path.dirname(destination) or "."
    os.makedirs(parent, exist_ok=True)
    from huggingface_hub import hf_hub_download

    log(f"Hugging Face から取得: {MODEL_REPO}@{MODEL_REVISION} / {MODEL_FILE}")
    hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, revision=MODEL_REVISION,
                    local_dir=parent, token=HF_TOKEN or None)


def download_model(destination: str) -> None:
    parent = os.path.dirname(destination) or "."
    os.makedirs(parent, exist_ok=True)
    started = time.time()
    if aria2_available():
        try:
            aria2_download(destination)
        except (OSError, RuntimeError) as error:
            log(f"aria2c を続けられません ({error})。huggingface_hub で取り直します")
            try:
                os.remove(destination + ".part")
            except OSError:
                pass
            hf_download(destination)
    else:
        hf_download(destination)
    log(f"取得完了 ({time.time() - started:.1f} 秒)")


def ensure_model() -> str:
    """llama-server に渡すモデルをローカルに用意して、そのパスを返す。"""
    global _model_source

    if model_ready(LOCAL_PATH):
        _model_source = "local"
        log(f"コンテナディスクのモデルを使います: {LOCAL_PATH} "
            f"({os.path.getsize(LOCAL_PATH) / 1e9:.2f} GB)")
        return LOCAL_PATH

    if volume_cache_enabled() and model_ready(CACHE_PATH):
        log(f"ボリュームのマスターからコピーします: {CACHE_PATH} -> {LOCAL_PATH}")
        if copy_model(CACHE_PATH, LOCAL_PATH, "コンテナディスクへの復元"):
            _model_source = "volume"
            return LOCAL_PATH

    if s3_enabled():
        state, size, note = s3_object_state()
        if state == "ok" and size >= MIN_MODEL_BYTES:
            if s3_download(LOCAL_PATH):
                _model_source = "s3"
                return LOCAL_PATH
        elif state == "ok":
            log(f"S3 のオブジェクトが小さすぎます ({size} バイト)。Hugging Face を使います")
        elif state == "missing":
            log("S3 にまだありません。Hugging Face から取得します")
        else:
            log(f"S3 を確認できませんでした ({note})。Hugging Face から取得します")
    elif VOLUME_CACHE and not volume_cache_enabled():
        log(f"注意: {VOLUME_DIR} が無く、S3 も未設定です (起動のたびに取得します)")

    download_model(LOCAL_PATH)
    _model_source = "huggingface"
    where = evacuate(LOCAL_PATH)
    if where == "none":
        log("退避先がありません (ボリュームを付けるか S3_* を設定してください)")
    return LOCAL_PATH


def master_report() -> dict:
    """いまのマスターの状態 (warmup の応答に載せて切り分けに使う)。"""
    state, size, note = s3_object_state()
    return {
        "source": _model_source,
        "local_path": LOCAL_PATH,
        "volume_dir": VOLUME_DIR,
        "volume_path": CACHE_PATH if volume_cache_enabled() else None,
        "s3": {
            "enabled": s3_enabled(),
            "endpoint": S3_ENDPOINT,
            "bucket": S3_BUCKET,
            "key": remote_key(),
            "state": state,
            "size": size,
            "note": note,
        },
        "lora": lora_report(),
    }


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


# --------------------------------------------------------------------------- #
# LoRA の準備
# --------------------------------------------------------------------------- #
def lora_ready(path: str) -> bool:
    """LoRA が途中で切れずに置いてあるか。"""
    if not path:
        return False
    try:
        return os.path.isfile(path) and os.path.getsize(path) >= LORA_MIN_BYTES
    except OSError:
        return False


def download_lora(destination: str) -> bool:
    """LoRA を素の HTTP で取る (9.7MB なので huggingface_hub は要らない)。"""
    import hashlib
    import urllib.request

    parent = os.path.dirname(destination) or "."
    os.makedirs(parent, exist_ok=True)
    part = destination + ".part"
    log(f"LoRA を取得: {LORA_URL}")
    digest = hashlib.sha256()
    try:
        request = urllib.request.Request(
            LORA_URL, headers={"User-Agent": "bonsai-worker/1.0"})
        with urllib.request.urlopen(request, timeout=300) as response:
            with open(part, "wb") as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
                    digest.update(block)
    except OSError as error:
        log(f"警告: LoRA を取得できませんでした ({error})。LoRA 無しで起動します")
        return False
    if LORA_SHA256 and digest.hexdigest() != LORA_SHA256:
        log("警告: LoRA の sha256 が違います。LoRA 無しで起動します")
        os.unlink(part)
        return False
    if os.path.getsize(part) < LORA_MIN_BYTES:
        log("警告: LoRA が小さすぎます。LoRA 無しで起動します")
        os.unlink(part)
        return False
    os.replace(part, destination)
    log(f"LoRA の取得完了 ({os.path.getsize(destination) / 1e6:.1f} MB)")
    return True


def ensure_lora() -> str:
    """llama-server に渡す LoRA を用意する。無効・失敗なら空文字を返す。"""
    if not LORA_FILE:
        return ""
    if lora_ready(LOCAL_LORA):
        log(f"コンテナディスクの LoRA を使います: {LOCAL_LORA}")
        return LOCAL_LORA
    if volume_cache_enabled() and lora_ready(CACHE_LORA):
        if copy_model(CACHE_LORA, LOCAL_LORA, "LoRA のコピー"):
            return LOCAL_LORA
    if download_lora(LOCAL_LORA):
        return LOCAL_LORA
    return ""


def lora_report() -> dict:
    """LoRA の設定と、いま手元にあるかを返す。"""
    return {
        "enabled": bool(LORA_FILE),
        "file": LORA_FILE or None,
        "scale": LORA_SCALE,
        "url": LORA_URL or None,
        "sha256": LORA_SHA256 or None,
        "path": LOCAL_LORA or None,
        "ready": lora_ready(LOCAL_LORA),
    }


def lora_args(path: str) -> list:
    """llama-server に渡す LoRA の引数。scale 1 のときだけ短い方を使う。"""
    if not path:
        return []
    if LORA_SCALE in ("", "1", "1.0"):
        return ["--lora", path]
    return ["--lora-scaled", f"{path}:{LORA_SCALE}"]


def find_llama_server() -> str:
    """``LLAMA_SERVER`` が無ければ、イメージの中から探す。

    リリース tarball の階層は版によって変わる (``llama-server`` が直下のことも
    ``bin/`` の下のこともある)。起動できないよりは探した方がよい。
    """
    if os.path.isfile(LLAMA_SERVER):
        return LLAMA_SERVER
    root = os.path.dirname(LLAMA_SERVER) or "/opt/llama"
    if os.path.basename(root) == "bin":
        root = os.path.dirname(root)
    for base, _dirs, files in os.walk(root):
        if "llama-server" in files:
            found = os.path.join(base, "llama-server")
            log(f"注意: LLAMA_SERVER を {found} に読み替えます (設定は {LLAMA_SERVER})")
            return found
    return LLAMA_SERVER


def start_server() -> None:
    """ワーカーごとに一度だけ起動する。2回目以降は既存プロセスを再利用する。"""
    global _proc, _started_at
    with _lock:
        if _proc is not None and _proc.poll() is None:
            return
        model_path = ensure_model()
        lora_path = ensure_lora()
        command = [
            find_llama_server(), "-m", model_path,
            "-ngl", GPU_LAYERS, "-c", CTX_SIZE,
            "--host", "127.0.0.1", "--port", str(PORT),
            *lora_args(lora_path),
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
        return {"status": "ready", "model": MODEL_FILE, **master_report(),
                "gpu": os.environ.get("RUNPOD_GPU_TYPE", "unknown-until-first-job")}

    # {"input": {"evacuate": true}} でマスターへ書き戻す（退避だけしたいとき）。
    if job_input.get("evacuate"):
        path = ensure_model()
        return {"status": "ready", "evacuated_to": evacuate(path), **master_report()}

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
    log(f"  ローカル(使い捨て): {LOCAL_DIR}")
    log(f"  マスター(ボリューム): {CACHE_DIR}"
        + ("" if volume_cache_enabled() else "  ※使いません"))
    log(f"  マスター(S3): {'s3://%s/%s' % (S3_BUCKET, remote_key()) if s3_enabled() else '未設定'}")
    log(f"  取得: {DOWNLOAD_BACKEND}"
        + (f" / aria2c で {DOWNLOAD_CONNECTIONS} 接続" if aria2_available() else " / huggingface_hub"))
    log(f"  LoRA: {LORA_FILE or '無し'}"
        + (f" (scale={LORA_SCALE}, {LORA_URL})" if LORA_FILE else ""))
    if len(MODEL_FILES) > 1:
        log(f"  注意: MODEL_FILES に {len(MODEL_FILES)} 件ありますが、"
            "いまは分割 GGUF に未対応です (MODEL_FILE の 1 件だけ取得します)")
    runpod.serverless.start({"handler": handler})

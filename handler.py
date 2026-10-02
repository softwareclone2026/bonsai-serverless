#!/usr/bin/env python3
"""RunPod Serverless ワーカー: Prism fork の llama-server を起動して推論を中継する。

モデルは「コンテナディスク = 使い捨ての作業コピー、外にマスターを 1 つ」として扱い、
起動のたびに次の順で探して最初に見つかったものを使う:

1. コンテナディスク (:data:`LOCAL_PATH`) — 同じワーカーでの 2 回目以降
2. マウントされたボリューム (:data:`CACHE_PATH`) — global Volume を付けた場合
3. Wasabi などの rclone キャッシュ — ``MODEL_SOURCE=wasabi`` のときだけ
4. S3 互換 API (Network Volume) — ボリュームを**マウントせず**使う場合
5. Hugging Face (:data:`MODEL_REPO`)

4 と 5 で取ってきたときは、使えるマスター (ボリューム or S3) へ**退避**して、
次回のコールドスタートで 1 か 2 から拾えるようにする。コンテナディスクは起動の
たびに捨てられるので、残るのはマスターだけ。llama-server には常にコンテナディスク
側のコピーを渡す。

3 の rclone キャッシュは「既に外にあるマスター」なので退避しない。渡し方は
既存ツール (tools/vast_movie.py) と**同じ**: ``RCLONE_CONFIG_B64`` に rclone.conf
を base64 で入れて env で渡す (crypt のパスワードは conf の中に obscure 済みで
入っているので、新しい秘密の形式は増やさない)。``MODEL_SOURCE`` を設定しない限り
この経路は使わない = 従来の挙動のまま。

生成の上限は ``MAX_TOKENS`` (別名 ``MAX_NEW_TOKENS``) で決める。呼び出し側
(bonsai27b) は常に ``-n`` (既定 1024) を送ってくるので、リクエストの値だけでは
env で上げられない。env を**明示したとき**はその値を下限として働かせ、
``MAX_TOKENS_OVERRIDE`` (別名 ``FORCE_MAX_TOKENS``) を書けば常にそれを使う。

このワーカーは **ComfyUI を起動しない**。動くのは llama-server だけで、健康判定は
llama-server の ``/health`` を見る。ComfyUI の ``/system_stats`` を見る呼び出し側の
健康チェックは、このワーカーには当てはまらない。

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


def _int_value(raw):
    """env の生値が整数として読めればその値、読めなければ None (警告は出さない)。"""
    if raw is None or not str(raw).strip():
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def _int_env(names, default: int) -> int:
    """env から整数を読む。壊れた値でも起動を止めず、警告して既定に落とす。"""
    for name in names:
        raw = os.environ.get(name)
        if raw is None or not str(raw).strip():
            continue
        try:
            return int(str(raw).strip())
        except ValueError:
            print(f"[worker] 警告: {name}={raw!r} は整数ではありません。無視します",
                  flush=True)
    return default


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

# --------------------------------------------------------------------------- #
# Wasabi などの rclone キャッシュ (暗号化リモート) から取る経路。
#
# 既存ツール (tools/vast_movie.py の rclone_config_b64 / _rclone_ready_shell) と
# **同じ渡し方**にする: rclone.conf を丸ごと base64 にした ``RCLONE_CONFIG_B64``
# を env で受け取り、``base64 -d`` 相当で ``~/.config/rclone/rclone.conf`` に書く。
# crypt のパスワードは conf の中に obscure 済みで入っているので、新しい秘密の
# 形式は発明しない。env が無ければ既存の rclone.conf をそのまま使う。
# ``MODEL_SOURCE`` を設定しない限りこの経路は使わない (後方互換)。
# --------------------------------------------------------------------------- #
#: 取得元: auto (従来どおり) / wasabi (rclone キャッシュを最優先)。
MODEL_SOURCE = (os.environ.get("MODEL_SOURCE") or "auto").strip().lower()
#: rclone のリモートパス。既存ツールの既定と同じ (--model-cache / WAN22_MODEL_CACHE)。
RCLONE_REMOTE = (os.environ.get("MODEL_CACHE")
                 or os.environ.get("WAN22_MODEL_CACHE")
                 or os.environ.get("RCLONE_REMOTE")
                 or "wasabi_crypt3:/vidgen_cache").strip()
#: rclone.conf を base64 にしたもの。**秘密**。ログにも応答にも出さない。
RCLONE_CONFIG_B64 = (os.environ.get("RCLONE_CONFIG_B64") or "").strip()
#: キャッシュ側のファイル名 (リモートからの相対)。実績のある llm/ 配下を使う。
CACHE_MODEL_FILE = (os.environ.get("CACHE_MODEL_FILE") or "llm/llm.gguf").strip()
CACHE_LORA_FILE = (os.environ.get("CACHE_LORA_FILE") or "llm/llm-lora.gguf").strip()
#: rclone の実体。未指定なら PATH → 無ければ公式バイナリを取る。
RCLONE_BIN = (os.environ.get("RCLONE_BIN") or "").strip()
#: 1 ファイルを分割して落とす本数 (0 で分割しない)。
RCLONE_STREAMS = max(0, _int_env(["RCLONE_STREAMS", "RCLONE_MULTI_THREAD_STREAMS"], 4))
#: 書き込む rclone.conf の場所。
RCLONE_CONFIG_PATH = (os.environ.get("RCLONE_CONFIG") or "").strip() or os.path.join(
    os.path.expanduser("~"), ".config", "rclone", "rclone.conf")

LLAMA_SERVER = os.environ.get("LLAMA_SERVER", "/opt/llama/bin/llama-server")
PORT = int(os.environ.get("WORKER_PORT", "8080"))
CTX_SIZE = os.environ.get("CTX_SIZE", "16384")
GPU_LAYERS = os.environ.get("GPU_LAYERS", "99")
READY_TIMEOUT = float(os.environ.get("READY_TIMEOUT", "900"))
# モデル作者推奨のサンプリング設定と thinking モード。
# 空文字を渡すと追加引数なしで起動する。
DEFAULT_EXTRA = "--jinja --reasoning on --reasoning-effort medium --temp 1.0 --top-p 0.95 --top-k 20"
EXTRA_ARGS = os.environ.get("LLAMA_EXTRA_ARGS", DEFAULT_EXTRA).split()

# 生成の上限。呼び出し側 (bonsai27b) は常に ``-n`` (既定 1024) を送ってくるため、
# ワーカーの env は「リクエストが無いときの既定」だけでは足りない (env を 2048 に
# しても 1024 で切れる)。そこで env を**明示したときだけ**下限としても働かせる。
# 未設定なら従来どおり = リクエストの値をそのまま使う。
#: 既定の最大生成トークン数。MAX_NEW_TOKENS でも指定できる (別名)。
_MAX_TOKENS_ENV = os.environ.get("MAX_TOKENS") or os.environ.get("MAX_NEW_TOKENS")
MAX_TOKENS_DEFAULT = _int_env(["MAX_TOKENS", "MAX_NEW_TOKENS"], 512)
#: env を明示したときだけ効く下限 (0 = 無効)。リクエストが小さいときに引き上げる。
#: 壊れた値 (整数でない) のときは下限にしない (警告だけ出して従来どおり)。
MAX_TOKENS_FLOOR = (MAX_TOKENS_DEFAULT
                    if _int_value(_MAX_TOKENS_ENV) is not None else 0)
#: リクエストより優先する上限。設定すると常にこれを使う。FORCE_MAX_TOKENS でも可。
MAX_TOKENS_OVERRIDE = _int_env(["MAX_TOKENS_OVERRIDE", "FORCE_MAX_TOKENS"], 0)

_proc = None
_lock = threading.Lock()
_started_at = None
#: 直近でモデルをどこから用意したか ("local" / "volume" / "wasabi" / "s3" / "huggingface")。
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


#: S3 の設定がどの env で入っているかを**秘密の値抜き**で確認するための名前。
S3_ENV_NAMES = (
    "S3_ENDPOINT", "RUNPOD_S3_ENDPOINT",
    "S3_REGION", "RUNPOD_S3_REGION",
    "S3_BUCKET", "RUNPOD_S3_BUCKET",
    "S3_ACCESS_KEY", "RUNPOD_S3_ACCESS_KEY",
    "S3_SECRET_KEY", "RUNPOD_S3_SECRET_KEY",
)


def s3_env_present() -> dict:
    """S3 系 env の**有無だけ**を返す (値・秘密は出さない)。"""
    return {name: bool(os.environ.get(name)) for name in S3_ENV_NAMES}


def s3_env_summary() -> str:
    """S3 の切り分け用の 1 行 (region / endpoint / 設定済み env 名)。秘密は出さない。"""
    present = [name for name, ok in s3_env_present().items() if ok]
    return (f"region={S3_REGION or '未設定'} / endpoint={S3_ENDPOINT or '未設定'}"
            f" / bucket={S3_BUCKET or '未設定'}"
            f" / 設定済み env: {', '.join(present) or 'なし'}")


def s3_error_detail(error) -> str:
    """S3 の失敗を、原因が分かる形にする (401 だけでは切り分けられない)。

    RunPod の S3 互換 API は 401 を返すことがあり、``note: "401"`` だけでは
    ``InvalidAccessKeyId`` (キー違い) なのか ``SignatureDoesNotMatch`` (秘密違い /
    リージョン違い) なのか ``AccessDenied`` なのか ``NoSuchBucket`` なのか分からない。
    botocore は XML 本文の Code / Message と HTTP ステータス・RequestId を
    ``response`` に入れるので、それを出す。**秘密の値は出さない**。
    """
    response = getattr(error, "response", None) or {}
    meta = response.get("ResponseMetadata", {}) or {}
    body = response.get("Error", {}) or {}
    parts = []
    code = body.get("Code") or meta.get("HTTPStatusCode")
    if code:
        parts.append(f"code={code}")
    if body.get("Message"):
        parts.append(f"message={body['Message']}")
    if meta.get("HTTPStatusCode"):
        parts.append(f"http={meta['HTTPStatusCode']}")
    if S3_REGION:
        parts.append(f"region={S3_REGION}")
    if meta.get("RequestId"):
        parts.append(f"request_id={meta['RequestId']}")
    parts.append("endpoint=" + (S3_ENDPOINT or "未設定"))
    if not body and not meta:
        # ClientError 以外 (接続失敗など) は例外そのものを残す。
        return f"{type(error).__name__}: {error} / endpoint={S3_ENDPOINT or '未設定'}"
    return " / ".join(str(part) for part in parts)


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
        # Code だけでなく Message / HTTP / region / RequestId も残す。
        return "error", 0, s3_error_detail(error)
    except Exception as error:  # 到達不能・DNS 失敗などもここで拾って報告する
        return "error", 0, s3_error_detail(error)
    return "ok", int(response.get("ContentLength") or 0), ""


# --------------------------------------------------------------------------- #
# Wasabi などの rclone キャッシュ (暗号化リモート)
# --------------------------------------------------------------------------- #
#: MODEL_SOURCE で rclone 経由を選ぶ値 (別名は既存ツールの呼び方に合わせる)。
_WASABI_SOURCES = ("wasabi", "rclone", "cache", "vidgen_cache", "s3-cache")


def wasabi_enabled() -> bool:
    """``MODEL_SOURCE`` で rclone キャッシュ経由を選んだか (未設定なら False)。"""
    return MODEL_SOURCE in _WASABI_SOURCES


def wasabi_remote_path(cache_file: str) -> str:
    """rclone リモート上のパス (``<remote>/<cache_file>``)。"""
    base = RCLONE_REMOTE.rstrip("/")
    if not cache_file:
        return base
    return f"{base}/{cache_file.lstrip('/')}"


def prepare_rclone_config() -> bool:
    """``RCLONE_CONFIG_B64`` を rclone.conf へ書く (無ければ既存設定を使う)。

    既存ツールと同じく base64 で受け取り、``base64 -d`` 相当で書き出す。
    crypt のパスワードは conf の中に obscure 済みで入っているので、別の秘密は
    受け取らない。**渡された中身はログにも応答にも出さない** (バイト数だけ出す)。
    """
    if not RCLONE_CONFIG_B64:
        if os.path.isfile(RCLONE_CONFIG_PATH):
            return True
        log(f"警告: RCLONE_CONFIG_B64 が未設定で {RCLONE_CONFIG_PATH} もありません")
        return False
    import base64
    import binascii
    import re

    cleaned = re.sub(r"\s+", "", RCLONE_CONFIG_B64)
    try:
        data = base64.b64decode(cleaned, validate=True)
    except (ValueError, binascii.Error) as error:
        log(f"警告: RCLONE_CONFIG_B64 を base64 として読めません ({type(error).__name__})")
        return False
    parent = os.path.dirname(RCLONE_CONFIG_PATH) or "."
    try:
        os.makedirs(parent, exist_ok=True)
        with open(RCLONE_CONFIG_PATH, "wb") as handle:
            handle.write(data)
        os.chmod(RCLONE_CONFIG_PATH, 0o600)
    except OSError as error:
        log(f"警告: rclone.conf を書けません ({error})")
        return False
    log(f"rclone 設定を用意しました: {RCLONE_CONFIG_PATH} ({len(data)} バイト)")
    return True


def install_rclone() -> str:
    """rclone が無いとき公式の current バイナリを取る (zipfile で展開)。"""
    import zipfile

    url = os.environ.get(
        "RCLONE_URL",
        "https://downloads.rclone.org/rclone-current-linux-amd64.zip")
    target_dir = os.environ.get("RCLONE_DIR", "/usr/local/bin")
    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError:
        target_dir = LOCAL_DIR or "/tmp"
        try:
            os.makedirs(target_dir, exist_ok=True)
        except OSError:
            log("警告: rclone の置き場所を作れません")
            return ""
    target = os.path.join(target_dir, "rclone")
    archive = os.path.join(target_dir, "rclone.zip")
    try:
        with urllib.request.urlopen(url, timeout=180) as response, \
                open(archive, "wb") as handle:
            shutil.copyfileobj(response, handle)
        with zipfile.ZipFile(archive) as bundle:
            members = [name for name in bundle.namelist() if name.endswith("/rclone")]
            if not members:
                raise RuntimeError("zip の中に rclone がありません")
            with bundle.open(members[0]) as source, open(target, "wb") as handle:
                shutil.copyfileobj(source, handle)
        os.chmod(target, 0o755)
    except Exception as error:  # 取得できなくても推論自体は続けられる
        log(f"警告: rclone を用意できませんでした ({type(error).__name__}: {error})")
        return ""
    finally:
        try:
            os.remove(archive)
        except OSError:
            pass
    log(f"rclone を用意しました: {target}")
    return target


def ensure_rclone() -> str:
    """使える rclone のパスを返す (無ければ入れる。失敗なら空文字)。"""
    if RCLONE_BIN and os.path.isfile(RCLONE_BIN):
        return RCLONE_BIN
    found = shutil.which("rclone")
    if found:
        return found
    log("rclone が見つかりません。取得します (Wasabi キャッシュ用)")
    return install_rclone()


def rclone_download(remote_file: str, destination: str, label: str,
                    min_bytes: int = 0) -> bool:
    """rclone でキャッシュから 1 ファイル取得する (失敗は False、秘密は出さない)。

    既存ツールと同じフラグ立て: ``--multi-thread-streams`` はダウンロードにだけ
    効くので、数 GB の GGUF を分割して落とすために使う。宛先は ``.part`` に書いて
    から rename する (途中で死んだファイルを完成品と誤認しない)。
    """
    binary = ensure_rclone()
    if not binary:
        return False
    if not prepare_rclone_config():
        return False
    if min_bytes <= 0:
        min_bytes = MIN_MODEL_BYTES
    remote = wasabi_remote_path(remote_file)
    part = destination + ".part"
    parent = os.path.dirname(part) or "."
    os.makedirs(parent, exist_ok=True)
    command = [
        binary, "copyto", remote, part,
        "--config", RCLONE_CONFIG_PATH,
        "--checkers", "16",
        "--stats", "10s", "--stats-one-line",
        "--stats-log-level", "NOTICE", "--log-level", "NOTICE",
    ]
    if RCLONE_STREAMS > 0:
        command += ["--multi-thread-streams", str(RCLONE_STREAMS),
                    "--multi-thread-cutoff", "50M"]
    log(f"{label}を rclone で取得: {remote} -> {destination}")
    started = time.time()
    result = subprocess.run(command, stdout=sys.stdout, stderr=subprocess.STDOUT)
    if result.returncode != 0 or not os.path.isfile(part):
        log(f"警告: rclone の取得に失敗しました (終了コード {result.returncode})")
        try:
            os.remove(part)
        except OSError:
            pass
        return False
    size = os.path.getsize(part)
    if size < min_bytes:
        log(f"警告: rclone の取得物が小さすぎます ({size} バイト)。捨てます")
        try:
            os.remove(part)
        except OSError:
            pass
        return False
    os.replace(part, destination)
    seconds = max(time.time() - started, 0.001)
    log(f"{label}の取得完了: {destination} ({size / 1e9:.2f} GB / {seconds:.1f} 秒"
        f" / {size / 1e6 / seconds:.0f} MB/s)")
    return True


def wasabi_report() -> dict:
    """Wasabi 経路の設定 (秘密は出さず、有無だけ)。warmup の切り分けに使う。"""
    return {
        "enabled": wasabi_enabled(),
        "model_source": MODEL_SOURCE,
        "remote": RCLONE_REMOTE,
        "model_file": CACHE_MODEL_FILE,
        "lora_file": CACHE_LORA_FILE if LORA_FILE else None,
        "config_b64": bool(RCLONE_CONFIG_B64),
        "config_path": RCLONE_CONFIG_PATH,
        "rclone": RCLONE_BIN if RCLONE_BIN and os.path.isfile(RCLONE_BIN)
                  else (shutil.which("rclone") or None),
        "streams": RCLONE_STREAMS,
    }


# --------------------------------------------------------------------------- #
# モデルの準備 (ローカル / ボリューム / Wasabi / S3 / Hugging Face)
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
        log(f"警告: S3 からの取得に失敗しました ({s3_error_detail(error)})")
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
        log(f"警告: S3 への退避に失敗しました ({s3_error_detail(error)})")
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

    # Wasabi (rclone キャッシュ)。MODEL_SOURCE=wasabi のときだけ通る。
    # 既に外にあるマスターなので、取れたら退避 (書き戻し) はしない。
    if wasabi_enabled():
        log(f"Wasabi キャッシュ: {wasabi_remote_path(CACHE_MODEL_FILE)}"
            f" (rclone={shutil.which('rclone') or RCLONE_BIN or '未取得'})")
        if rclone_download(CACHE_MODEL_FILE, LOCAL_PATH, "モデル"):
            _model_source = "wasabi"
            return LOCAL_PATH
        log("Wasabi キャッシュから取得できませんでした。次の取得元を試します")

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
            # 401 などは「何が」悪いのか分かる形で出す (秘密の値は出さない)。
            log(f"S3 を確認できませんでした: {note}")
            log(f"S3 の設定: {s3_env_summary()}")
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
            "region": S3_REGION,
            "bucket": S3_BUCKET,
            "key": remote_key(),
            "state": state,
            "size": size,
            "note": note,
            # どの env が入っているか (値・秘密は出さない)。
            "env_present": s3_env_present(),
        },
        "wasabi": wasabi_report(),
        "max_tokens": {
            "default": MAX_TOKENS_DEFAULT,
            "floor": MAX_TOKENS_FLOOR,
            "override": MAX_TOKENS_OVERRIDE or None,
        },
        "download": {
            "backend": DOWNLOAD_BACKEND,
            "aria2c": bool(shutil.which("aria2c")),
            "connections": DOWNLOAD_CONNECTIONS,
            "revision": MODEL_REVISION,
            "url": model_url() if not MODEL_URL else MODEL_URL,
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
    if wasabi_enabled():
        # LoRA も同じキャッシュに置いてある (llm/llm-lora.gguf)。9.7MB なので
        # rclone の起動コストは小さいが、取得元を 1 つに寄せておく。
        if rclone_download(CACHE_LORA_FILE, LOCAL_LORA, "LoRA",
                           min_bytes=LORA_MIN_BYTES):
            return LOCAL_LORA
        log("Wasabi キャッシュから LoRA を取得できませんでした。HTTP で取り直します")
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
def effective_max_tokens(requested) -> int:
    """llama-server へ実際に渡す ``max_tokens`` を決める。

    1024 で切れる原因はワーカーではなく**呼び出し側が常に送る** ``max_tokens``
    (bonsai27b の ``-n``、既定 1024) だった。ワーカーの env は「未指定時の既定」
    だけでは効かないので、env を明示したときは下限としても働かせる:

    - ``MAX_TOKENS_OVERRIDE`` (``FORCE_MAX_TOKENS``): 常にこれ (リクエストより優先)
    - リクエストに ``max_tokens`` が無い: ``MAX_TOKENS_DEFAULT``
    - env の ``MAX_TOKENS`` を明示した: ``max(リクエスト, MAX_TOKENS_FLOOR)``
    - env 未設定: リクエストの値をそのまま使う (従来どおり)
    """
    if MAX_TOKENS_OVERRIDE:
        return MAX_TOKENS_OVERRIDE
    if requested is None:
        return MAX_TOKENS_DEFAULT
    try:
        value = int(requested)
    except (TypeError, ValueError):
        log(f"警告: max_tokens={requested!r} を整数として読めません。既定を使います")
        return MAX_TOKENS_DEFAULT
    if MAX_TOKENS_FLOOR and value < MAX_TOKENS_FLOOR:
        log(f"max_tokens {value} を MAX_TOKENS={MAX_TOKENS_FLOOR} まで引き上げます")
        return MAX_TOKENS_FLOOR
    return value


def build_payload(job_input: dict) -> dict:
    if "messages" in job_input:
        messages = job_input["messages"]
    elif "prompt" in job_input:
        messages = [{"role": "user", "content": job_input["prompt"]}]
    else:
        raise ValueError("input には 'messages' か 'prompt' が必要です")

    payload = {
        "messages": messages,
        "max_tokens": effective_max_tokens(job_input.get("max_tokens")),
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
        # 実際に渡した上限。呼び出し側の -n が env でどう変わったかを追えるようにする。
        "max_tokens": payload["max_tokens"],
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
    log(f"  S3 の設定: {s3_env_summary()}")
    log(f"  Wasabi キャッシュ: {'有効' if wasabi_enabled() else '無効 (MODEL_SOURCE 未設定)'}"
        f" / remote={RCLONE_REMOTE} / config_b64={'あり' if RCLONE_CONFIG_B64 else 'なし'}")
    log(f"  max_tokens: 既定={MAX_TOKENS_DEFAULT}"
        f" / 下限={MAX_TOKENS_FLOOR or 'なし'} (MAX_TOKENS env"
        f"{'あり' if MAX_TOKENS_FLOOR else 'なし/無効'})"
        f" / 強制={MAX_TOKENS_OVERRIDE or 'なし'}")
    log(f"  取得: {DOWNLOAD_BACKEND}"
        + (f" / aria2c で {DOWNLOAD_CONNECTIONS} 接続" if aria2_available() else " / huggingface_hub"))
    log(f"  LoRA: {LORA_FILE or '無し'}"
        + (f" (scale={LORA_SCALE}, {LORA_URL})" if LORA_FILE else ""))
    if len(MODEL_FILES) > 1:
        log(f"  注意: MODEL_FILES に {len(MODEL_FILES)} 件ありますが、"
            "いまは分割 GGUF に未対応です (MODEL_FILE の 1 件だけ取得します)")
    runpod.serverless.start({"handler": handler})

"""
AI 多模态聊天服务器
支持: 文本 + 图片 + 音频 + 多轮对话历史
存储: ~/.gemma3n/ (参考 Codex 架构)

后端模式:
  - mmproj (默认): 使用 llama.cpp 多模态，无需 PyTorch
  - mps (进阶): 使用 PyTorch MPS 加速，需要安装 torch/transformers
"""
import os
import io
import sys
import base64
import uuid
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import warnings
import time
import psutil
import subprocess
import platform
import threading
import signal
import atexit

warnings.filterwarnings("ignore")

# ========== 日志配置 ==========
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.environ.get(
    "LOG_FORMAT",
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("webui")

# 减少第三方库的日志噪音
logging.getLogger("werkzeug").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

# PyTorch 是可选依赖 (仅 MPS 模式需要)
PYTORCH_AVAILABLE = False
torch = None
np = None
librosa = None
AutoProcessor = None
Gemma3nForConditionalGeneration = None

# 先单独导入 librosa (用于音频处理，不依赖 PyTorch)
try:
    import librosa as _librosa
    librosa = _librosa
except ImportError:
    pass

try:
    import torch as _torch
    import numpy as _np
    from transformers import AutoProcessor as _AutoProcessor
    from transformers import Gemma3nForConditionalGeneration as _Gemma3nForConditionalGeneration
    torch = _torch
    np = _np
    AutoProcessor = _AutoProcessor
    Gemma3nForConditionalGeneration = _Gemma3nForConditionalGeneration
    PYTORCH_AVAILABLE = True
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
except ImportError:
    pass

app = Flask(__name__, static_folder="static")

# CORS 配置 - 支持环境变量自定义允许的域
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "").split(",") if os.environ.get("ALLOWED_ORIGINS") else None
if ALLOWED_ORIGINS:
    CORS(app, origins=ALLOWED_ORIGINS)
else:
    # 默认允许本地开发访问
    CORS(app, origins=[
        "http://localhost:*",
        "http://127.0.0.1:*",
        "http://localhost:5001",
        "http://localhost:8080",
        "http://127.0.0.1:5001",
        "http://127.0.0.1:8080"
    ])

# ========== 请求验证配置 ==========
# 最大请求体大小 (默认 500MB，用于支持大型音频/图像)
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 500 * 1024 * 1024))
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 请求速率限制 (每分钟最大请求数)
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))

# 简单的内存速率限制器
_rate_limit_cache: dict = {}
_rate_limit_lock = threading.Lock()


def check_rate_limit(client_ip: str) -> bool:
    """
    检查请求是否超过速率限制

    Args:
        client_ip: 客户端 IP 地址

    Returns:
        True 如果请求被允许，False 如果超过限制
    """
    if RATE_LIMIT_PER_MINUTE <= 0:
        return True  # 禁用速率限制

    current_time = time.time()
    window_start = current_time - 60  # 1 分钟窗口

    with _rate_limit_lock:
        if client_ip not in _rate_limit_cache:
            _rate_limit_cache[client_ip] = []

        # 清理过期的请求记录
        _rate_limit_cache[client_ip] = [
            t for t in _rate_limit_cache[client_ip] if t > window_start
        ]

        # 检查是否超过限制
        if len(_rate_limit_cache[client_ip]) >= RATE_LIMIT_PER_MINUTE:
            return False

        # 记录当前请求
        _rate_limit_cache[client_ip].append(current_time)
        return True


@app.before_request
def before_request_handler():
    """请求前处理: 验证和速率限制"""
    # 跳过健康检查端点
    if request.path in ["/health", "/api/status"]:
        return None

    # 跳过静态文件
    if request.path.startswith("/static/") or request.path in ["/", "/chat", "/docs"]:
        return None

    # 检查速率限制
    client_ip = request.remote_addr or "unknown"
    if not check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return jsonify({
            "error": "Rate limit exceeded. Please try again later."
        }), 429

    # 验证 Content-Type (仅对 POST 请求)
    if request.method == "POST":
        content_type = request.content_type or ""
        if not content_type.startswith("application/json"):
            return jsonify({
                "error": f"Invalid Content-Type: {content_type}. Expected application/json"
            }), 415

    return None


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.utcfromtimestamp(float(value))
        except ValueError:
            return None


@app.errorhandler(413)
def request_entity_too_large(error):
    """处理请求体过大错误"""
    return jsonify({
        "error": f"Request body too large. Maximum size is {MAX_CONTENT_LENGTH // (1024*1024)}MB"
    }), 413


# 全局变量
model = None
processor = None
model_loaded = False
model_info = {}
dummy_image = None
# 默认使用 llama.cpp (mmproj)，MPS 需要 PyTorch
DEFAULT_BACKEND = os.environ.get("GEMMA3N_BACKEND", "mmproj")  # mmproj | mps

# llama.cpp 路径/模型（mmproj 模式使用）
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVOLUTION_AVAILABLE = False
try:
    from contexts.evolution import (
        EvolutionConfig,
        MemoryManager,
        build_embedding_service_from_env,
        create_index,
        SQLiteLogStore,
        SQLiteMemoryStore,
    )
    from contexts.evolution.schema import Feedback, InteractionLog, MemoryFragment
    EVOLUTION_AVAILABLE = True
except Exception:
    EVOLUTION_AVAILABLE = False
LINGKONG_HOME = Path.home() / ".lingkong"

# 优先使用 ~/.lingkong 安装目录，回退到项目目录
def _find_binary(name, fallback):
    lingkong_path = LINGKONG_HOME / "bin" / name
    if lingkong_path.exists():
        return str(lingkong_path)
    return fallback

def _find_model(name, fallback):
    lingkong_path = LINGKONG_HOME / "models" / name
    if lingkong_path.exists():
        return str(lingkong_path)
    return fallback

LLAMA_MTMD_BIN = os.environ.get("LLAMA_MTMD_BIN", _find_binary("llama-mtmd-cli", str(REPO_ROOT / "infra/llama.cpp/build/bin/llama-mtmd-cli")))
LLAMA_MM_MODEL = os.environ.get("LLAMA_MM_MODEL", _find_model("gemma-3n-E2B-it-Q4_K_M.gguf", str(REPO_ROOT / "artifacts/gguf/gemma-3n-E2B-it-Q4_K_M.gguf")))
LLAMA_MM_PROJ_IMAGE = os.environ.get("LLAMA_MM_PROJ_IMAGE", _find_model("gemma-3n-vision-mmproj-f16.gguf", str(REPO_ROOT / "artifacts/gguf/gemma-3n-vision-mmproj-f16.gguf")))
LLAMA_MM_PROJ_AUDIO = os.environ.get("LLAMA_MM_PROJ_AUDIO", _find_model("gemma-3n-audio-mmproj-f16.gguf", str(REPO_ROOT / "artifacts/gguf/gemma-3n-audio-mmproj-f16.gguf")))
LLAMA_MM_PROJ = os.environ.get("LLAMA_MM_PROJ", "")
LLAMA_MM_PROJ_COMBINED = ",".join([p for p in [LLAMA_MM_PROJ_IMAGE, LLAMA_MM_PROJ_AUDIO] if p]) if (LLAMA_MM_PROJ_IMAGE or LLAMA_MM_PROJ_AUDIO) else LLAMA_MM_PROJ
LLAMA_MM_N_PREDICT = int(os.environ.get("LLAMA_MM_N_PREDICT", "128"))
LLAMA_MM_DEVICE = os.environ.get("LLAMA_MM_DEVICE", "none")
LLAMA_MM_N_GPU_LAYERS = os.environ.get("LLAMA_MM_N_GPU_LAYERS", "0")

# llama-run 路径/模型（llama.cpp 纯文本模式）
LLAMA_RUN_BIN = os.environ.get("LLAMA_RUN_BIN", _find_binary("llama-run", str(REPO_ROOT / "infra/llama.cpp/build/bin/llama-run")))
LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", _find_binary("llama-server", str(REPO_ROOT / "infra/llama.cpp/build/bin/llama-server")))
LLAMA_SERVER_PORT = int(os.environ.get("LLAMA_SERVER_PORT", "8081"))
LLAMA_RUN_MODEL = os.environ.get("LLAMA_RUN_MODEL", "")
# 自动查找可用的 GGUF 模型
if not LLAMA_RUN_MODEL:
    for candidate in [
        LINGKONG_HOME / "models/gemma-3n-E2B-it-Q4_K_M.gguf",
        REPO_ROOT / "artifacts/gguf/gemma-3n-finetuned-Q4_K_M.gguf",
        REPO_ROOT / "artifacts/gguf/gemma-3n-E2B-it-Q4_K_M.gguf",
        REPO_ROOT / "artifacts/gguf/gemma-3n-E2B-it-fp16.gguf",
    ]:
        if candidate.exists():
            LLAMA_RUN_MODEL = str(candidate)
            break

# llama-server 进程管理 (纯文本模式)
llama_server_process = None
llama_server_ready = False

# llama-server mmproj 模式 (多模态)
LLAMA_MMPROJ_SERVER_PORT = int(os.environ.get("LLAMA_MMPROJ_SERVER_PORT", "8082"))
llama_mmproj_server_process = None
llama_mmproj_server_ready = False


# ========== 线程安全的进程状态管理 ==========
class ThreadSafeProcessState:
    """线程安全的进程状态容器"""

    def __init__(self):
        self._lock = threading.RLock()
        self._llama_server_process = None
        self._llama_server_ready = False
        self._llama_mmproj_server_process = None
        self._llama_mmproj_server_ready = False
        self._shutdown_requested = False

    @property
    def llama_server_process(self):
        with self._lock:
            return self._llama_server_process

    @llama_server_process.setter
    def llama_server_process(self, value):
        with self._lock:
            self._llama_server_process = value

    @property
    def llama_server_ready(self):
        with self._lock:
            return self._llama_server_ready

    @llama_server_ready.setter
    def llama_server_ready(self, value):
        with self._lock:
            self._llama_server_ready = value

    @property
    def llama_mmproj_server_process(self):
        with self._lock:
            return self._llama_mmproj_server_process

    @llama_mmproj_server_process.setter
    def llama_mmproj_server_process(self, value):
        with self._lock:
            self._llama_mmproj_server_process = value

    @property
    def llama_mmproj_server_ready(self):
        with self._lock:
            return self._llama_mmproj_server_ready

    @llama_mmproj_server_ready.setter
    def llama_mmproj_server_ready(self, value):
        with self._lock:
            self._llama_mmproj_server_ready = value

    @property
    def shutdown_requested(self):
        with self._lock:
            return self._shutdown_requested

    @shutdown_requested.setter
    def shutdown_requested(self, value):
        with self._lock:
            self._shutdown_requested = value


# 全局状态实例
_process_state = ThreadSafeProcessState()


def cleanup_subprocesses():
    """清理所有 llama-server 子进程"""
    for proc, name in [
        (_process_state.llama_server_process, "llama-server"),
        (_process_state.llama_mmproj_server_process, "llama-mmproj-server"),
    ]:
        if proc is not None:
            logger.info(f"正在关闭 {name} 子进程...")
            try:
                proc.terminate()
                proc.wait(timeout=10)
                logger.info(f"{name} 子进程已正常关闭")
            except subprocess.TimeoutExpired:
                logger.warning(f"{name} 未响应 SIGTERM，强制终止...")
                proc.kill()
                proc.wait()
                logger.info(f"{name} 子进程已强制终止")
            except OSError as e:
                logger.error(f"关闭 {name} 时 OS 错误: {e}")
            except ProcessLookupError:
                logger.debug(f"{name} 进程已不存在")

    _process_state.llama_server_process = None
    _process_state.llama_server_ready = False
    _process_state.llama_mmproj_server_process = None
    _process_state.llama_mmproj_server_ready = False


def signal_handler(signum, frame):
    """处理终止信号"""
    sig_name = signal.Signals(signum).name
    logger.info(f"收到信号 {sig_name}，开始优雅关闭...")
    _process_state.shutdown_requested = True
    cleanup_subprocesses()
    sys.exit(0)


# 注册信号处理器和退出钩子
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
atexit.register(cleanup_subprocesses)

# 存储路径 (~/.gemma3n/)
GEMMA3N_HOME = Path.home() / ".gemma3n"
SESSIONS_DIR = GEMMA3N_HOME / "sessions"
HISTORY_FILE = GEMMA3N_HOME / "history.jsonl"

# Evolution logging (SQLite + memory index)
EVOLUTION_ENABLED = os.environ.get("EVOLUTION_ENABLED", "1") == "1"
EVOLUTION_PERSIST_MEDIA = os.environ.get("EVOLUTION_PERSIST_MEDIA", "1") == "1"
EVOLUTION_INDEX_METRIC = os.environ.get("EVOLUTION_INDEX_METRIC", "cosine")
EVOLUTION_EMBED_MEDIA = os.environ.get("EVOLUTION_EMBED_MEDIA", "0") == "1"
EVOLUTION_EMBED_TEXT = os.environ.get("EVOLUTION_EMBED_TEXT", "0") == "1"
evolution_config = None
evolution_log_store = None
evolution_memory_manager = None
evolution_index = None
evolution_embedding_service = None

# 会话管理 (内存缓存)
sessions = {}  # session_id -> {"messages": [...], "created_at": timestamp, "file_path": ...}
MAX_SESSIONS = 100
MAX_HISTORY_TURNS = 10  # 保留最近N轮对话

# sudo 权限状态
sudo_authorized = False
sudo_refresh_thread = None

# ========== Thought Signature 系统 (压缩记忆) ==========
# 核心思想: 图片/音频 → 模型理解 → 存入 signature → 后续轮次恢复
import hashlib
import hmac

THOUGHT_SIGNATURE_SECRET = "gemma3n-thought-signature-key"
media_understanding_cache = {}  # media_ref -> {"understanding": "...", "session_id": "..."}
thought_states = {}  # session_id -> {"turn_index": 0, "media_refs": [...]}

# 缓存配置
MAX_CACHE_SIZE = 1000  # 最大缓存条目数
CACHE_TTL = 3600  # 缓存过期时间 (秒)


def cleanup_caches():
    """清理过期和超量的缓存条目"""
    global media_understanding_cache, thought_states
    current_time = time.time()

    # 清理过期的 media_understanding_cache
    expired_keys = [
        key for key, val in media_understanding_cache.items()
        if current_time - val.get("created_at", 0) > CACHE_TTL
    ]
    for key in expired_keys:
        del media_understanding_cache[key]

    # 如果缓存太大，移除最旧的条目
    if len(media_understanding_cache) > MAX_CACHE_SIZE:
        sorted_items = sorted(
            media_understanding_cache.items(),
            key=lambda x: x[1].get("created_at", 0)
        )
        for key, _ in sorted_items[:len(media_understanding_cache) - MAX_CACHE_SIZE]:
            del media_understanding_cache[key]

    # 清理对应的 thought_states
    valid_session_ids = set(val.get("session_id") for val in media_understanding_cache.values())
    stale_sessions = [sid for sid in thought_states if sid not in valid_session_ids and sid not in sessions]
    for sid in stale_sessions:
        del thought_states[sid]


def start_cache_cleanup_thread():
    """启动后台缓存清理线程"""
    def cleanup_loop():
        while True:
            time.sleep(600)  # 每10分钟清理一次
            try:
                cleanup_caches()
            except Exception as e:
                logger.warning(f"缓存清理出错: {e}")

    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()
    return thread


def generate_media_signature(session_id: str, turn_index: int, understanding: str) -> str:
    """
    为媒体理解生成签名引用

    Returns: media_ref (用于后续恢复)
    """
    import base64
    timestamp = int(time.time())
    media_ref = hashlib.md5(f"{session_id}:{turn_index}:{timestamp}".encode()).hexdigest()[:12]

    # 存储理解到缓存
    media_understanding_cache[media_ref] = {
        "session_id": session_id,
        "turn_index": turn_index,
        "understanding": understanding,
        "created_at": timestamp
    }

    # 更新会话的思维状态
    if session_id not in thought_states:
        thought_states[session_id] = {"turn_index": 0, "media_refs": []}
    thought_states[session_id]["media_refs"].append(media_ref)
    thought_states[session_id]["turn_index"] = turn_index

    return media_ref


def get_session_media_context(session_id: str) -> str:
    """
    获取会话中所有媒体理解的上下文

    这是 thought signature 作为"压缩记忆"的核心：
    - 第1轮: 用户上传图片 → 模型生成理解 → 存入 signature
    - 第2轮: 从 signature 恢复理解 → 注入到上下文中
    """
    context_parts = []

    state = thought_states.get(session_id, {})
    media_refs = state.get("media_refs", [])

    for media_ref in media_refs:
        cached = media_understanding_cache.get(media_ref)
        if cached:
            turn = cached.get("turn_index", 0)
            understanding = cached.get("understanding", "")
            if understanding:
                context_parts.append(f"[Turn {turn} - Media Understanding]: {understanding}")

    return "\n".join(context_parts) if context_parts else ""

stats = {
    "total_requests": 0,
    "total_tokens": 0,
    "total_time": 0,
    "avg_speed": 0
}

def init_evolution_storage():
    """Initialize evolution SQLite stores and vector index placeholder."""
    global evolution_config, evolution_log_store, evolution_memory_manager, evolution_index
    if not (EVOLUTION_ENABLED and EVOLUTION_AVAILABLE):
        return
    if evolution_log_store is not None:
        return

    evolution_root = Path(
        os.environ.get("EVOLUTION_STORAGE_ROOT", str(GEMMA3N_HOME / "evolution"))
    )
    sqlite_path = Path(
        os.environ.get("EVOLUTION_SQLITE_PATH", str(evolution_root / "evolution.db"))
    )
    artifact_root = Path(
        os.environ.get("EVOLUTION_ARTIFACT_ROOT", str(evolution_root / "artifacts"))
    )
    index_path = Path(
        os.environ.get("EVOLUTION_INDEX_PATH", str(evolution_root / "index"))
    )

    evolution_config = EvolutionConfig(
        storage_root=evolution_root,
        sqlite_path=sqlite_path,
        artifact_root=artifact_root,
        vector_index_path=index_path,
        retention_days=int(os.environ.get("EVOLUTION_RETENTION_DAYS", "30")),
        min_feedback_score=float(os.environ.get("EVOLUTION_MIN_FEEDBACK_SCORE", "0.6")),
        short_term_window_minutes=int(os.environ.get("EVOLUTION_SHORT_TERM_MINUTES", "5")),
        short_term_window_frames=int(os.environ.get("EVOLUTION_SHORT_TERM_FRAMES", "5")),
        history_retrieval_model=os.environ.get("EVOLUTION_HISTORY_MODEL", "siglip"),
        history_retrieval_backend=os.environ.get("EVOLUTION_HISTORY_BACKEND", "faiss"),
    )
    evolution_config.ensure_paths()
    evolution_log_store = SQLiteLogStore(sqlite_path)
    memory_store = SQLiteMemoryStore(sqlite_path, include_embeddings=False)
    evolution_index = create_index(
        backend=evolution_config.history_retrieval_backend,
        metric=EVOLUTION_INDEX_METRIC,
    )
    evolution_memory_manager = MemoryManager(store=memory_store, index=evolution_index)
    logger.info(f"Evolution storage ready: {evolution_root}")


def get_embedding_service():
    """Lazily load embedding service for evolution indexing."""
    global evolution_embedding_service
    if not (EVOLUTION_ENABLED and EVOLUTION_AVAILABLE):
        return None
    if not (EVOLUTION_EMBED_MEDIA or EVOLUTION_EMBED_TEXT):
        return None
    if evolution_embedding_service is not None:
        return evolution_embedding_service
    try:
        evolution_embedding_service = build_embedding_service_from_env()
        return evolution_embedding_service
    except Exception as exc:
        logger.warning(f"Evolution embedding service unavailable: {exc}")
        return None


def close_evolution_storage() -> None:
    """Close evolution stores to flush pending writes."""
    global evolution_log_store, evolution_memory_manager
    if evolution_log_store is not None:
        evolution_log_store.close()
        evolution_log_store = None
    if evolution_memory_manager is not None:
        evolution_memory_manager.close()
        evolution_memory_manager = None


def _build_media_path(
    *, session_id: str, turn_index: int, index: int, kind: str, ext: str
) -> str:
    if EVOLUTION_PERSIST_MEDIA and evolution_config is not None:
        base_dir = evolution_config.artifact_root / session_id / kind
    else:
        base_dir = Path("/tmp")
    base_dir.mkdir(parents=True, exist_ok=True)
    suffix = ext if ext.startswith(".") else f".{ext}"
    filename = f"{kind}_{session_id}_{turn_index}_{index}_{uuid.uuid4().hex[:6]}{suffix}"
    return str(base_dir / filename)


def _record_evolution_interaction(
    *,
    session_id: str,
    turn_index: int,
    user_text: str,
    response_text: str,
    image_paths: list,
    audio_paths: list,
    backend: str,
    response_time_ms: Optional[int],
    thought_signature: Optional[str],
    feedback_data: Optional[object],
    implicit_feedback_score: Optional[float],
    retrieved_doc_ids: Optional[list],
    embedding: Optional[object],
) -> None:
    if not (EVOLUTION_ENABLED and EVOLUTION_AVAILABLE):
        return
    if evolution_log_store is None or evolution_memory_manager is None:
        return

    timestamp = datetime.utcnow()
    modalities = []
    if user_text:
        modalities.append("text")
    if image_paths:
        modalities.append("image")
    if audio_paths:
        modalities.append("audio")

    feedback = None
    explicit_feedback = None
    if isinstance(feedback_data, dict):
        explicit_feedback = feedback_data.get("rating")
        feedback = Feedback(
            rating=explicit_feedback,
            score=feedback_data.get("score"),
            timestamp=timestamp,
            source=feedback_data.get("source", "explicit"),
        )
    elif isinstance(feedback_data, str):
        explicit_feedback = feedback_data
        feedback = Feedback(
            rating=feedback_data,
            score=None,
            timestamp=timestamp,
            source="explicit",
        )

    if implicit_feedback_score is not None:
        if feedback is None:
            feedback = Feedback(
                rating=None,
                score=implicit_feedback_score,
                timestamp=timestamp,
                source="implicit",
            )
        elif feedback.score is None:
            feedback.score = implicit_feedback_score

    try:
        log = InteractionLog(
            id=str(uuid.uuid4()),
            session_id=session_id,
            turn_index=turn_index,
            timestamp=timestamp,
            user_input=user_text,
            input_modalities=modalities or ["text"],
            attachments=[*image_paths, *audio_paths],
            assistant_response=response_text,
            thought_signature=thought_signature,
            response_time_ms=response_time_ms,
            model_version=model_info.get("name") or model_info.get("backend"),
            lora_version=None,
            retrieved_doc_ids=retrieved_doc_ids or [],
            feedback=feedback,
            metadata={
                "backend": backend,
                "image_paths": list(image_paths),
                "audio_paths": list(audio_paths),
            },
        )
        evolution_log_store.append(logs=[log])

        fragment = MemoryFragment(
            id=str(uuid.uuid4()),
            timestamp=timestamp,
            session_id=session_id,
            turn_index=turn_index,
            source="webui",
            keyframe_path=image_paths[0] if image_paths else None,
            audio_path=audio_paths[0] if audio_paths else None,
            text_input=user_text,
            transcript=None,
            scene_description=None,
            detected_objects=[],
            user_intent=None,
            ai_response=response_text,
            gaze_heatmap_path=None,
            focus_duration_s=None,
            explicit_feedback=explicit_feedback,
            implicit_feedback_score=implicit_feedback_score,
            model_version=model_info.get("name") or model_info.get("backend"),
            lora_version=None,
            metadata={
                "backend": backend,
                "image_paths": list(image_paths),
                "audio_paths": list(audio_paths),
                "thought_signature": thought_signature,
            },
        )
        if embedding is not None:
            fragment.multimodal_embedding = embedding

        evolution_memory_manager.add(fragments=[fragment])
    except Exception as exc:
        logger.warning(f"Evolution logging failed: {exc}")

def init_storage():
    """初始化存储目录结构"""
    GEMMA3N_HOME.mkdir(exist_ok=True)
    SESSIONS_DIR.mkdir(exist_ok=True)
    if not HISTORY_FILE.exists():
        HISTORY_FILE.touch()
    init_evolution_storage()
    logger.info(f"Storage 初始化完成: {GEMMA3N_HOME}")

atexit.register(close_evolution_storage)

def get_session_dir():
    """获取当天的会话目录 sessions/YYYY/MM/DD/"""
    now = datetime.now()
    day_dir = SESSIONS_DIR / str(now.year) / f"{now.month:02d}" / f"{now.day:02d}"
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir

def save_session_to_disk(session_id, session_data):
    """将会话保存到 JSONL 文件"""
    if "file_path" not in session_data:
        # 创建新文件
        now = datetime.now()
        timestamp = now.strftime("%H%M%S")
        filename = f"session-{timestamp}-{session_id}.jsonl"
        session_data["file_path"] = str(get_session_dir() / filename)

    file_path = Path(session_data["file_path"])

    # 写入会话数据 (完整覆盖)
    with open(file_path, "w", encoding="utf-8") as f:
        # 第一行: 会话元数据
        meta = {
            "type": "meta",
            "session_id": session_id,
            "created_at": session_data["created_at"],
            "title": session_data.get("title", "新对话"),
            "updated_at": time.time()
        }
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        # 后续行: 消息
        for msg in session_data["messages"]:
            item = {"type": "message", **msg}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def load_session_from_disk(file_path):
    """从 JSONL 文件加载会话"""
    file_path = Path(file_path)
    if not file_path.exists():
        return None

    session_data = {"messages": [], "file_path": str(file_path)}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("type") == "meta":
                session_data["created_at"] = item.get("created_at", time.time())
                session_data["title"] = item.get("title", "新对话")
                session_data["session_id"] = item.get("session_id")
            elif item.get("type") == "message":
                del item["type"]
                session_data["messages"].append(item)

    return session_data

def append_to_history(session_id, text):
    """追加到全局历史记录"""
    entry = {
        "session_id": session_id,
        "ts": int(time.time()),
        "text": text[:100]  # 只保存前100字符作为摘要
    }
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def list_all_sessions():
    """列出所有会话文件"""
    session_list = []
    for jsonl_file in SESSIONS_DIR.rglob("*.jsonl"):
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    meta = json.loads(first_line)
                    if meta.get("type") == "meta":
                        session_list.append({
                            "session_id": meta.get("session_id"),
                            "title": meta.get("title", "新对话"),
                            "created_at": meta.get("created_at", 0),
                            "updated_at": meta.get("updated_at", 0),
                            "file_path": str(jsonl_file)
                        })
        except Exception as e:
            logger.warning(f" 无法读取会话文件 {jsonl_file}: {e}")

    # 按更新时间倒序
    session_list.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
    return session_list[:50]  # 最多返回50个

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024 / 1024 / 1024
    return round(mem_gb, 2)


def request_sudo_permission():
    """
    启动时请求 sudo 权限（用户输入一次密码）
    并启动后台线程定期刷新凭证
    """
    global sudo_authorized, sudo_refresh_thread

    if platform.system() != "Darwin":
        return False

    logger.info("请求硬件监控权限 (用于获取 GPU 温度)")

    try:
        # 请求 sudo 权限
        result = subprocess.run(
            ["sudo", "-v"],
            timeout=60  # 给用户60秒输入密码
        )
        if result.returncode == 0:
            sudo_authorized = True
            logger.info("权限授权成功！GPU 温度监控已启用")

            # 启动后台线程定期刷新 sudo 凭证
            def refresh_sudo():
                while sudo_authorized:
                    time.sleep(240)  # 每4分钟刷新一次（sudo 默认5分钟超时）
                    try:
                        subprocess.run(["sudo", "-v"], capture_output=True, timeout=5)
                    except subprocess.TimeoutExpired:
                        pass  # 超时忽略
                    except Exception:
                        pass  # 其他错误忽略

            sudo_refresh_thread = threading.Thread(target=refresh_sudo, daemon=True)
            sudo_refresh_thread.start()
            return True
        else:
            logger.warning("权限未授权，GPU 温度监控将不可用")
            return False
    except subprocess.TimeoutExpired:
        logger.warning("授权超时，GPU 温度监控将不可用")
        return False
    except KeyboardInterrupt:
        logger.warning("已跳过权限授权，GPU 温度监控将不可用")
        return False
    except Exception as e:
        logger.warning(f"授权失败: {e}")
        return False


def get_hardware_stats():
    """获取硬件监控信息 (GPU使用率、显存、温度等)"""
    hw_stats = {
        "gpu_usage": "-",
        "vram_usage": "-",
        "gpu_temp": "-",
        "memory_usage": "-",
        "cpu_usage": "-"
    }

    try:
        # CPU 使用率 (不使用 interval 避免阻塞)
        hw_stats["cpu_usage"] = f"{psutil.cpu_percent():.1f}%"

        # 内存使用
        mem = psutil.virtual_memory()
        hw_stats["memory_usage"] = f"{mem.used / 1024**3:.1f} GB / {mem.total / 1024**3:.1f} GB"

        system = platform.system()

        if system == "Darwin":  # macOS
            try:
                # 检测 MPS 状态
                if torch.backends.mps.is_available():
                    hw_stats["gpu_usage"] = "MPS 运行中"
                else:
                    hw_stats["gpu_usage"] = "CPU 模式"

                # macOS 统一内存架构，显存和内存共享
                hw_stats["vram_usage"] = "统一内存"

                # 尝试通过 powermetrics 获取 GPU 温度 (需要 sudo 权限)
                if sudo_authorized:
                    try:
                        result = subprocess.run(
                            ["sudo", "powermetrics", "--samplers", "smc", "-i", "1", "-n", "1"],
                            capture_output=True, text=True, timeout=3
                        )
                        if result.returncode == 0:
                            output = result.stdout
                            # 查找 GPU 温度相关行
                            for line in output.split("\n"):
                                if "GPU" in line and "die" in line.lower():
                                    parts = line.split(":")
                                    if len(parts) >= 2:
                                        temp_str = parts[1].strip().replace("C", "").strip()
                                        try:
                                            temp = float(temp_str)
                                            hw_stats["gpu_temp"] = f"{temp:.0f}°C"
                                            break
                                        except:
                                            pass
                                elif "GPU" in line and "temp" in line.lower():
                                    parts = line.split(":")
                                    if len(parts) >= 2:
                                        temp_str = parts[1].strip().replace("C", "").strip()
                                        try:
                                            temp = float(temp_str)
                                            hw_stats["gpu_temp"] = f"{temp:.0f}°C"
                                            break
                                        except:
                                            pass

                            # 如果没找到 GPU 温度，尝试找 SOC 温度
                            if hw_stats["gpu_temp"] == "-":
                                for line in output.split("\n"):
                                    if "SOC" in line and "temp" in line.lower():
                                        parts = line.split(":")
                                        if len(parts) >= 2:
                                            temp_str = parts[1].strip().replace("C", "").strip()
                                            try:
                                                temp = float(temp_str)
                                                hw_stats["gpu_temp"] = f"{temp:.0f}°C"
                                                break
                                            except:
                                                pass
                    except subprocess.TimeoutExpired:
                        hw_stats["gpu_temp"] = "超时"
                    except Exception:
                        hw_stats["gpu_temp"] = "获取失败"
                else:
                    hw_stats["gpu_temp"] = "未授权"

            except Exception as e:
                logger.debug(f" macOS GPU info error: {e}")

        elif system == "Linux":
            # NVIDIA GPU (使用 nvidia-smi)
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if lines:
                        # 可能有多个 GPU，取第一个
                        parts = lines[0].split(",")
                        if len(parts) >= 4:
                            gpu_util = parts[0].strip()
                            mem_used = float(parts[1].strip())
                            mem_total = float(parts[2].strip())
                            temp = parts[3].strip()

                            hw_stats["gpu_usage"] = f"{gpu_util}%"
                            hw_stats["vram_usage"] = f"{mem_used/1024:.1f} GB / {mem_total/1024:.1f} GB"
                            hw_stats["gpu_temp"] = f"{temp}°C"
            except FileNotFoundError:
                # nvidia-smi 不存在，可能是 AMD 或无 GPU
                try:
                    # 尝试 AMD GPU (rocm-smi)
                    result = subprocess.run(
                        ["rocm-smi", "--showuse", "--showtemp", "--showmeminfo", "vram"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        hw_stats["gpu_usage"] = "AMD GPU"
                except:
                    pass
            except Exception as e:
                logger.debug(f" Linux GPU info error: {e}")

        elif system == "Windows":
            # Windows NVIDIA GPU
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, shell=True
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if lines:
                        parts = lines[0].split(",")
                        if len(parts) >= 4:
                            gpu_util = parts[0].strip()
                            mem_used = float(parts[1].strip())
                            mem_total = float(parts[2].strip())
                            temp = parts[3].strip()

                            hw_stats["gpu_usage"] = f"{gpu_util}%"
                            hw_stats["vram_usage"] = f"{mem_used/1024:.1f} GB / {mem_total/1024:.1f} GB"
                            hw_stats["gpu_temp"] = f"{temp}°C"
            except Exception as e:
                logger.debug(f" Windows GPU info error: {e}")

    except Exception as e:
        logger.debug(f" Hardware stats error: {e}")

    return hw_stats

def cleanup_old_sessions():
    """清理内存中的旧会话缓存 (磁盘文件保留)"""
    if len(sessions) > MAX_SESSIONS:
        # 按创建时间排序，从内存中移除最老的
        sorted_sessions = sorted(sessions.items(), key=lambda x: x[1]["created_at"])
        for sid, _ in sorted_sessions[:len(sessions) - MAX_SESSIONS]:
            del sessions[sid]

def load_model():
    """
    加载 PyTorch 模型 (仅 MPS 模式需要)
    mmproj 模式下跳过，使用 llama.cpp
    """
    global model, processor, model_loaded, model_info, dummy_image

    if model_loaded:
        return True

    # mmproj 模式不需要加载 PyTorch 模型
    if DEFAULT_BACKEND == "mmproj":
        logger.info("使用 llama.cpp 多模态后端 (mmproj)")
        logger.info("跳过 PyTorch 模型加载")
        model_info = {
            "name": "Gemma 3N (llama.cpp)",
            "params": "2B",
            "dtype": "Q4_K_M",
            "device": "GPU (Metal)",
            "load_time": 0,
            "memory_gb": 0,
            "capabilities": ["文本对话", "图像理解", "音频转录", "多轮对话"],
            "max_tokens": 8192,
            "backend": "llama.cpp mmproj"
        }
        model_loaded = True
        return True

    # MPS 模式需要 PyTorch
    if not PYTORCH_AVAILABLE:
        logger.error("MPS 模式需要安装 PyTorch")
        logger.error("请运行: pip install torch transformers librosa")
        logger.error("或切换到 mmproj 模式: export GEMMA3N_BACKEND=mmproj")
        return False

    logger.info("加载 AI 多模态模型 (PyTorch MPS)...")

    model_name = "google/gemma-3n-E2B-it"
    load_start = time.time()

    logger.info("[1/2] 加载处理器...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    logger.info("[2/2] 加载模型...")
    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        max_memory={"mps": "64GiB", "cpu": "64GiB"},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    dummy_image = Image.new('RGB', (64, 64), color='white')
    load_time = time.time() - load_start
    total_params = sum(p.numel() for p in model.parameters())

    model_info = {
        "name": "AI Multimodal",
        "params": f"{total_params / 1e9:.2f}B",
        "dtype": "bfloat16",
        "device": str(model.device),
        "load_time": round(load_time, 2),
        "memory_gb": get_memory_usage(),
        "capabilities": ["文本对话", "图像理解", "音频转录", "多轮对话"],
        "max_tokens": 8192,
    }

    model_loaded = True
    logger.info(f"模型加载完成! 耗时 {load_time:.2f}s")
    logger.info(f"内存占用: {model_info['memory_gb']} GB")
    return True

def start_llama_mmproj_server():
    """启动 llama-server with mmproj 作为持久化多模态推理服务"""
    if not Path(LLAMA_SERVER_BIN).exists():
        logger.info(f"llama-mmproj-server: 二进制文件不存在: {LLAMA_SERVER_BIN}")
        return False

    if not Path(LLAMA_MM_MODEL).exists():
        logger.info(f"llama-mmproj-server: 模型文件不存在: {LLAMA_MM_MODEL}")
        return False

    if not Path(LLAMA_MM_PROJ_IMAGE).exists():
        logger.info(f"llama-mmproj-server: mmproj 文件不存在: {LLAMA_MM_PROJ_IMAGE}")
        return False

    # 检查是否已在运行
    try:
        import requests
        resp = requests.get(f"http://127.0.0.1:{LLAMA_MMPROJ_SERVER_PORT}/health", timeout=2)
        if resp.status_code == 200:
            logger.info(f"llama-mmproj-server: 已在端口 {LLAMA_MMPROJ_SERVER_PORT} 运行")
            _process_state.llama_mmproj_server_ready = True
            return True
    except requests.exceptions.RequestException:
        pass  # 服务未运行，继续启动

    logger.info(f"llama-mmproj-server: 启动中... 端口: {LLAMA_MMPROJ_SERVER_PORT}")
    logger.info(f"llama-mmproj-server: 模型: {LLAMA_MM_MODEL}")
    logger.info(f"llama-mmproj-server: mmproj: {LLAMA_MM_PROJ_IMAGE}")

    env = os.environ.copy()
    bin_dir = str(Path(LLAMA_SERVER_BIN).parent)
    lib_dir = str(Path(LLAMA_SERVER_BIN).parent.parent / "lib")
    env["DYLD_LIBRARY_PATH"] = f"{lib_dir}:{bin_dir}:{env.get('DYLD_LIBRARY_PATH', '')}"

    cmd = [
        LLAMA_SERVER_BIN,
        "-m", LLAMA_MM_MODEL,
        "--mmproj", LLAMA_MM_PROJ_IMAGE,
        "--port", str(LLAMA_MMPROJ_SERVER_PORT),
        "--host", "127.0.0.1",
        "-ngl", "999",
        "-t", "8",
        "--ctx-size", "4096",
    ]

    _process_state.llama_mmproj_server_process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 等待服务启动
    import requests
    for _ in range(60):  # 最多等待 60 秒 (首次加载模型可能较慢)
        try:
            resp = requests.get(f"http://127.0.0.1:{LLAMA_MMPROJ_SERVER_PORT}/health", timeout=1)
            if resp.status_code == 200:
                logger.info(f"llama-mmproj-server: 启动成功！")
                _process_state.llama_mmproj_server_ready = True
                return True
        except requests.exceptions.RequestException:
            pass  # 等待服务启动
        time.sleep(1)

    logger.info("llama-mmproj-server: 启动超时")
    return False


def run_llama_mmproj(prompt, image_path=None, audio_path=None,
                     messages_history=None, session_id=None, has_media=False):
    """
    使用 llama-server with mmproj 生成回复
    优先使用持久化服务 (16x 更快)，失败时回退到 CLI

    Args:
        prompt: 用户输入的文本
        image_path: 图片路径 (单个字符串或路径列表)
        audio_path: 音频路径 (单个字符串或路径列表)
        messages_history: 历史消息列表 [{"role": "user/assistant", "text": "..."}]
        session_id: 会话ID (用于 thought signature)
        has_media: 当前消息是否包含媒体
    """
    # 标准化路径为列表
    image_paths = []
    if image_path:
        if isinstance(image_path, str):
            image_paths = [image_path]
        else:
            image_paths = list(image_path)

    audio_paths = []
    if audio_path:
        if isinstance(audio_path, str):
            audio_paths = [audio_path]
        else:
            audio_paths = list(audio_path)

    # 构建包含上下文的完整 prompt
    full_prompt = _build_mmproj_prompt(
        prompt, messages_history, session_id, has_media
    )

    # 音频暂不支持 server 模式，回退到 CLI
    # 多图片也使用 CLI (llama-server 单次请求只支持一张图)
    if audio_paths or len(image_paths) > 1:
        return run_llama_mmproj_cli(full_prompt, image_paths, audio_paths)

    # 尝试使用 server 模式 (单图片情况)
    if not _process_state.llama_mmproj_server_ready:
        if not start_llama_mmproj_server():
            logger.info("mmproj: Server 启动失败，回退到 CLI 模式")
            return run_llama_mmproj_cli(full_prompt, image_paths, audio_paths)

    start = time.time()
    try:
        import requests

        # 构建消息内容
        content = []
        if image_paths:
            # 将图片转为 base64 (server 模式只处理第一张)
            with open(image_paths[0], "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
        content.append({"type": "text", "text": full_prompt})

        resp = requests.post(
            f"http://127.0.0.1:{LLAMA_MMPROJ_SERVER_PORT}/v1/chat/completions",
            json={
                "model": "gemma-3n",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": LLAMA_MM_N_PREDICT,
                "temperature": 0.7,
            },
            timeout=120
        )

        elapsed = time.time() - start

        if resp.status_code != 200:
            # 服务器可能出错，回退到 CLI
            _process_state.llama_mmproj_server_ready = False
            return run_llama_mmproj_cli(full_prompt, image_paths, audio_paths)

        data = resp.json()
        response = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        # 从 usage 获取 token 数
        usage = data.get("usage", {})
        output_tokens = usage.get("completion_tokens", len(response) // 4)
        speed = output_tokens / elapsed if elapsed > 0 else 0

        return {
            "response": response,
            "metrics": {
                "total_time": round(elapsed, 2),
                "speed": round(speed, 1),
                "tokens": output_tokens,
                "backend": "mmproj-server"
            }
        }
    except requests.exceptions.Timeout:
        logger.info("mmproj: Server 请求超时，回退到 CLI 模式")
        _process_state.llama_mmproj_server_ready = False
        return run_llama_mmproj_cli(full_prompt, image_paths, audio_paths)
    except requests.exceptions.ConnectionError as e:
        logger.info(f"mmproj: Server 连接失败: {e}，回退到 CLI 模式")
        _process_state.llama_mmproj_server_ready = False
        return run_llama_mmproj_cli(full_prompt, image_paths, audio_paths)
    except (json.JSONDecodeError, KeyError, OSError) as e:
        logger.info(f"mmproj: Server 响应处理失败: {e}，回退到 CLI 模式")
        _process_state.llama_mmproj_server_ready = False
        return run_llama_mmproj_cli(full_prompt, image_paths, audio_paths)


def _build_mmproj_prompt(prompt, messages_history=None, session_id=None, has_media=False):
    """
    构建包含历史上下文和媒体理解的完整 prompt

    策略:
    - 如果有新媒体: 不注入历史上下文，让模型专注于当前媒体
    - 如果没有新媒体: 注入历史对话 + 媒体理解 (thought signature)
    """
    if has_media:
        # 有新媒体时，直接返回原始 prompt
        logger.info(f"mmproj: 有新媒体，跳过历史上下文注入")
        return prompt

    context_parts = []

    # 1. 获取媒体理解上下文 (thought signature 压缩记忆)
    if session_id:
        media_context = get_session_media_context(session_id)
        if media_context:
            context_parts.append(f"[Previous Media Understanding]\n{media_context}")
            logger.info(f"mmproj: 注入媒体理解上下文: {len(media_context)} 字符")

    # 2. 获取历史对话上下文
    if messages_history:
        history_parts = []
        for msg in messages_history[-MAX_HISTORY_TURNS * 2:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_parts.append(f"{role}: {msg['text']}")
        if history_parts:
            history_context = "\n".join(history_parts)
            context_parts.append(f"[Previous Conversation]\n{history_context}")
            logger.info(f"mmproj: 注入历史对话: {len(history_parts)} 条消息")

    # 3. 构建完整 prompt
    if context_parts:
        context_str = "\n\n".join(context_parts)
        full_prompt = f"""{context_str}

[Current Message]: {prompt}

Please respond to the current message, taking into account the context above."""
        return full_prompt
    else:
        return prompt


def run_llama_mmproj_cli(prompt, image_paths=None, audio_paths=None):
    """
    使用 llama-mtmd-cli 生成回复 (支持多图片/多音频)

    注意: llama.cpp 目前不支持同时加载视觉和音频 projector (Metal bug)
    解决方案: 分两次调用，先处理图片，再处理音频

    Args:
        prompt: 用户输入的文本
        image_paths: 图片路径列表
        audio_paths: 音频路径列表
    """
    if not Path(LLAMA_MTMD_BIN).exists():
        return {"error": f"llama-mtmd-cli 不存在: {LLAMA_MTMD_BIN}"}
    if not Path(LLAMA_MM_MODEL).exists():
        return {"error": f"模型文件不存在: {LLAMA_MM_MODEL}"}

    # 标准化为列表
    if image_paths is None:
        image_paths = []
    elif isinstance(image_paths, str):
        image_paths = [image_paths]

    if audio_paths is None:
        audio_paths = []
    elif isinstance(audio_paths, str):
        audio_paths = [audio_paths]

    logger.debug(f" mmproj CLI: {len(image_paths)} 张图片, {len(audio_paths)} 个音频")

    # llama.cpp 不支持同时加载视觉和音频 projector (Metal bug)
    # 解决方案: 分两次处理
    if image_paths and audio_paths:
        logger.debug("同时有图片和音频，分两次处理...")

        # 第一次: 处理图片
        image_result = _run_mmproj_single(prompt + " (Focus on describing the images)", image_paths, None)
        if "error" in image_result:
            return image_result

        # 第二次: 处理音频
        audio_result = _run_mmproj_single("Transcribe the audio content", None, audio_paths)
        if "error" in audio_result:
            return audio_result

        # 合并结果
        combined_response = f"**图像分析:**\n{image_result['response']}\n\n**音频转录:**\n{audio_result['response']}"
        total_time = image_result['metrics']['total_time'] + audio_result['metrics']['total_time']

        return {
            "response": combined_response,
            "metrics": {
                "total_time": round(total_time, 2),
                "speed": round((image_result['metrics'].get('speed', 0) + audio_result['metrics'].get('speed', 0)) / 2, 1),
                "backend": "mmproj-cli (split)",
                "images": len(image_paths),
                "audios": len(audio_paths)
            }
        }
    else:
        # 只有图片或只有音频，直接处理
        return _run_mmproj_single(prompt, image_paths, audio_paths)


def _run_mmproj_single(prompt, image_paths=None, audio_paths=None):
    """单次 mmproj CLI 调用 (只处理图片或只处理音频)"""
    image_paths = image_paths or []
    audio_paths = audio_paths or []

    # 根据输入类型动态选择 mmproj
    mmproj_list = []
    if image_paths and LLAMA_MM_PROJ_IMAGE and Path(LLAMA_MM_PROJ_IMAGE).exists():
        mmproj_list.append(LLAMA_MM_PROJ_IMAGE)
    if audio_paths and LLAMA_MM_PROJ_AUDIO and Path(LLAMA_MM_PROJ_AUDIO).exists():
        mmproj_list.append(LLAMA_MM_PROJ_AUDIO)

    if not mmproj_list:
        if LLAMA_MM_PROJ and Path(LLAMA_MM_PROJ).exists():
            mmproj_list = [LLAMA_MM_PROJ]
        elif LLAMA_MM_PROJ_COMBINED:
            mmproj_list = [p.strip() for p in LLAMA_MM_PROJ_COMBINED.split(",") if p.strip() and Path(p.strip()).exists()]

    if not mmproj_list:
        return {"error": "未配置有效的 mmproj 路径"}

    mmproj_combined = ",".join(mmproj_list)
    logger.debug(f" mmproj CLI 使用: {mmproj_combined}")

    cmd = [
        LLAMA_MTMD_BIN,
        "--log-verbosity", "0",
        "--no-warmup",
        "-m", LLAMA_MM_MODEL,
        "--mmproj", mmproj_combined,
        "-p", prompt,
        "-n", str(LLAMA_MM_N_PREDICT),
        "--temp", "0.7",
    ]
    if image_paths:
        cmd += ["--image", ",".join(image_paths)]
    if audio_paths:
        cmd += ["--audio", ",".join(audio_paths)]

    start = time.time()
    try:
        env = os.environ.copy()
        bin_dir = str(Path(LLAMA_MTMD_BIN).parent)
        env["DYLD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('DYLD_LIBRARY_PATH', '')}"
        out = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180, env=env)
        elapsed = time.time() - start
        lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
        content_lines = [
            ln for ln in lines
            if not ln.startswith(("ggml", "AVX", "gguf", "llama", "clip", "Using", "model", "warmup", "load"))
        ]
        response = "\n".join(content_lines).strip() if content_lines else "\n".join(lines).strip()

        output_tokens = len(response) // 4
        speed = output_tokens / elapsed if elapsed > 0 else 0

        return {
            "response": response,
            "metrics": {
                "total_time": round(elapsed, 2),
                "speed": round(speed, 1),
                "backend": "mmproj-cli",
                "images": len(image_paths),
                "audios": len(audio_paths)
            }
        }
    except subprocess.TimeoutExpired:
        return {"error": "llama-mtmd-cli 超时"}
    except subprocess.CalledProcessError as e:
        return {"error": f"llama-mtmd-cli 失败: {e.stderr or e.stdout}"}


def run_llama_run(prompt, history_context=""):
    """
    使用 llama-run 进行纯文本推理
    这是一个轻量级的 llama.cpp 后端，不需要加载 PyTorch 模型
    """
    if not Path(LLAMA_RUN_BIN).exists():
        return {"error": f"llama-run 不存在: {LLAMA_RUN_BIN}"}
    if not LLAMA_RUN_MODEL or not Path(LLAMA_RUN_MODEL).exists():
        return {"error": f"GGUF 模型文件不存在: {LLAMA_RUN_MODEL}"}

    # 构建完整的提示词（包含历史上下文）
    full_prompt = prompt
    if history_context:
        full_prompt = f"{history_context}\n\nUser: {prompt}\n\nAssistant:"

    start = time.time()
    try:
        env = os.environ.copy()
        bin_dir = str(Path(LLAMA_RUN_BIN).parent)
        env["DYLD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('DYLD_LIBRARY_PATH', '')}"

        cmd = [
            LLAMA_RUN_BIN,
            "--ngl", "999",  # 使用 GPU 加速
            "--temp", "0.7",
            "-t", "8",  # 使用 8 个线程
            LLAMA_RUN_MODEL,
            full_prompt
        ]

        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=env
        )

        elapsed = time.time() - start

        if out.returncode != 0:
            return {"error": f"llama-run 失败: {out.stderr}"}

        # llama-run 的输出是干净的，直接使用
        # 过滤掉可能的 ANSI 转义码
        response = out.stdout.strip()
        # 移除 ANSI 控制字符
        import re
        response = re.sub(r'\x1b\[[0-9;]*m', '', response)
        response = response.strip()

        # 估算 token 数（简单估算：字符数/4）
        output_tokens = len(response) // 4
        speed = output_tokens / elapsed if elapsed > 0 else 0

        return {
            "response": response,
            "metrics": {
                "total_time": round(elapsed, 2),
                "speed": round(speed, 1),
                "backend": "llama.cpp"
            }
        }
    except subprocess.TimeoutExpired:
        return {"error": "llama-run 超时"}
    except Exception as e:
        return {"error": f"llama-run 错误: {str(e)}"}


def start_llama_server():
    """启动 llama-server 作为持久化推理服务"""
    if not Path(LLAMA_SERVER_BIN).exists():
        logger.info(f"llama-server: 二进制文件不存在: {LLAMA_SERVER_BIN}")
        return False

    if not LLAMA_RUN_MODEL or not Path(LLAMA_RUN_MODEL).exists():
        logger.info(f"llama-server: 模型文件不存在: {LLAMA_RUN_MODEL}")
        return False

    # 检查是否已在运行
    try:
        import requests
        resp = requests.get(f"http://127.0.0.1:{LLAMA_SERVER_PORT}/health", timeout=2)
        if resp.status_code == 200:
            logger.info(f"llama-server: 已在端口 {LLAMA_SERVER_PORT} 运行")
            _process_state.llama_server_ready = True
            return True
    except requests.exceptions.RequestException:
        pass  # 服务未运行，继续启动

    logger.info(f"llama-server: 启动中... 端口: {LLAMA_SERVER_PORT}")
    logger.info(f"llama-server: 模型: {LLAMA_RUN_MODEL}")

    env = os.environ.copy()
    bin_dir = str(Path(LLAMA_SERVER_BIN).parent)
    lib_dir = str(Path(LLAMA_SERVER_BIN).parent.parent / "lib")
    env["DYLD_LIBRARY_PATH"] = f"{lib_dir}:{bin_dir}:{env.get('DYLD_LIBRARY_PATH', '')}"

    cmd = [
        LLAMA_SERVER_BIN,
        "-m", LLAMA_RUN_MODEL,
        "--port", str(LLAMA_SERVER_PORT),
        "--host", "127.0.0.1",
        "-ngl", "999",
        "-t", "8",
        "--ctx-size", "4096",
        "--flash-attn", "on",
    ]

    _process_state.llama_server_process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 等待服务启动
    import requests
    for _ in range(30):  # 最多等待 30 秒
        try:
            resp = requests.get(f"http://127.0.0.1:{LLAMA_SERVER_PORT}/health", timeout=1)
            if resp.status_code == 200:
                logger.info(f"llama-server: 启动成功！")
                _process_state.llama_server_ready = True
                return True
        except requests.exceptions.RequestException:
            pass  # 等待服务启动
        time.sleep(1)

    logger.info("llama-server: 启动超时")
    return False


def query_llama_server(prompt, history_context=""):
    """
    通过 llama-server API 进行推理
    速度更快，因为模型已经预加载
    """
    if not _process_state.llama_server_ready:
        if not start_llama_server():
            # 回退到 llama-run
            return run_llama_run(prompt, history_context)

    # 构建完整的提示词
    full_prompt = prompt
    if history_context:
        full_prompt = f"{history_context}\n\nUser: {prompt}\n\nAssistant:"

    start = time.time()
    try:
        import requests
        resp = requests.post(
            f"http://127.0.0.1:{LLAMA_SERVER_PORT}/completion",
            json={
                "prompt": full_prompt,
                "n_predict": 256,
                "temperature": 0.7,
                "stop": ["</s>", "<eos>", "\n\nUser:", "\nUser:"],
                "stream": False,
            },
            timeout=60
        )

        elapsed = time.time() - start

        if resp.status_code != 200:
            return {"error": f"llama-server 请求失败: {resp.status_code}"}

        data = resp.json()
        response = data.get("content", "").strip()
        tokens_predicted = data.get("tokens_predicted", len(response) // 4)

        # 从 timings 获取真实速度
        timings = data.get("timings", {})
        speed = timings.get("predicted_per_second", 0)
        if speed == 0:
            speed = tokens_predicted / elapsed if elapsed > 0 else 0

        return {
            "response": response,
            "metrics": {
                "total_time": round(elapsed, 2),
                "speed": round(speed, 1),
                "tokens": tokens_predicted,
                "backend": "llama-server"
            }
        }
    except requests.exceptions.Timeout:
        _process_state.llama_server_ready = False
        return {"error": "llama-server 请求超时"}
    except requests.exceptions.ConnectionError as e:
        _process_state.llama_server_ready = False
        return {"error": f"llama-server 连接失败: {e}"}
    except (json.JSONDecodeError, KeyError) as e:
        return {"error": f"llama-server 响应解析失败: {e}"}


def generate_response(messages_history, current_content, session_id=None, has_media=False, media_type=None):
    """
    生成回复
    messages_history: 历史消息列表 [{"role": "user/assistant", "text": "..."}]
    current_content: 当前消息的content列表
    session_id: 会话ID (用于 thought signature)
    has_media: 当前消息是否包含媒体 (图片/音频)
    media_type: 当前媒体类型 ("image", "audio", None)

    注意: Gemma 3n 处理器要求每条消息都有图片，否则会报批次大小不一致错误。
    解决方案: 将历史对话合并成上下文文本，只发送一条包含图片的当前消息。
    """
    global stats

    if not model_loaded:
        return {"error": "模型未加载"}

    start_time = time.time()
    history_turns = len(messages_history) // 2  # 记录实际的历史轮次

    # 获取历史媒体理解上下文 (thought signature 压缩记忆)
    # 策略：
    # - 如果当前消息有新媒体：不注入历史媒体理解（避免干扰）
    # - 如果当前消息没有新媒体：注入历史媒体理解，让模型能够回答关于之前媒体的问题
    media_context = ""
    if session_id and not has_media:
        # 只在没有新媒体时才注入历史媒体理解
        media_context = get_session_media_context(session_id)
        if media_context:
            logger.debug(f" 注入媒体理解上下文: {len(media_context)} 字符")
    elif has_media:
        logger.debug(f" 当前有新{media_type or '媒体'}，跳过历史媒体理解注入")

    # 方案: 将历史对话合并成上下文文本
    # 这样就只有一条 user 消息，避免批次不一致问题
    # 注意: 当有新媒体时，不注入历史对话，避免干扰模型对当前媒体的理解
    history_context = ""
    if messages_history and not has_media:
        history_parts = []
        for msg in messages_history[-MAX_HISTORY_TURNS * 2:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_parts.append(f"{role}: {msg['text']}")
        history_context = "\n".join(history_parts)
    elif has_media:
        logger.debug(f" 当前有新媒体，跳过历史对话注入")

    # 构建单条消息 (包含历史上下文 + 当前输入)
    messages = []

    # 修改当前消息内容，添加历史上下文
    modified_content = []
    current_text = ""

    for item in current_content:
        if item.get("type") == "text":
            current_text = item.get("text", "")
        else:
            modified_content.append(item)

    # 构建完整的上下文提示
    # 包含: 媒体理解 + 历史对话 + 当前消息
    full_context_parts = []

    # 1. 媒体理解上下文 (thought signature 压缩记忆) - 只在无新媒体时注入
    if media_context:
        full_context_parts.append(f"[Previous Media Understanding]\n{media_context}")

    # 2. 历史对话上下文
    if history_context:
        full_context_parts.append(f"[Previous Conversation]\n{history_context}")

    # 3. 当前消息
    if full_context_parts:
        # 有上下文，构建完整提示
        context_str = "\n\n".join(full_context_parts)
        context_prompt = f"""{context_str}

[Current Message]: {current_text}

Please respond to the current message, taking into account the context above."""
        modified_content.append({"type": "text", "text": context_prompt})
    else:
        # 无上下文，直接使用当前文本
        modified_content.append({"type": "text", "text": current_text})

    messages.append({"role": "user", "content": modified_content})

    logger.debug(f" 消息数量: {len(messages)}, 历史轮次: {history_turns}, 有新媒体: {has_media}")

    # 处理输入
    tokenize_start = time.time()
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    tokenize_time = time.time() - tokenize_start

    # 准备生成参数
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    input_tokens = input_ids.shape[1]

    logger.debug(f" input_tokens: {input_tokens}")

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": 512,
        "do_sample": False,
    }

    # 处理图像
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        generate_kwargs["pixel_values"] = inputs["pixel_values"].to(model.device, dtype=model.dtype)

    # 处理音频
    if "input_features" in inputs and inputs["input_features"] is not None:
        generate_kwargs["input_features"] = inputs["input_features"].to(model.device, dtype=model.dtype)
        generate_kwargs["input_features_mask"] = inputs["input_features_mask"].to(model.device)

    # 生成
    generate_start = time.time()
    with torch.inference_mode():
        outputs = model.generate(**generate_kwargs)
    generate_time = time.time() - generate_start

    # 解码
    output_tokens = len(outputs[0]) - input_tokens
    response = processor.tokenizer.decode(
        outputs[0][input_ids.shape[1]:],
        skip_special_tokens=True
    )

    total_time = time.time() - start_time
    speed = output_tokens / generate_time if generate_time > 0 else 0

    # 更新统计
    stats["total_requests"] += 1
    stats["total_tokens"] += output_tokens
    stats["total_time"] += total_time
    stats["avg_speed"] = stats["total_tokens"] / stats["total_time"] if stats["total_time"] > 0 else 0

    return {
        "response": response,
        "metrics": {
            "total_time": round(total_time, 2),
            "tokenize_time": round(tokenize_time, 3),
            "generate_time": round(generate_time, 2),
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "speed": round(speed, 1),
            "history_turns": history_turns  # 使用实际的历史轮次
        }
    }

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/chat")
def chat_page():
    return send_from_directory("static", "chat.html")

@app.route("/docs")
def docs_page():
    return send_from_directory("static", "docs.html")


@app.route("/health")
def health():
    """
    健康检查端点 (用于负载均衡器和监控)

    返回服务器状态、子进程状态、系统资源使用情况等。
    HTTP 200 表示健康，HTTP 503 表示未就绪。
    """
    # 基本状态
    is_ready = model_loaded
    status = "ok" if is_ready else "starting"

    # 子进程状态
    subprocess_status = {}

    # llama-server (纯文本)
    proc = _process_state.llama_server_process
    if proc:
        try:
            proc_info = psutil.Process(proc.pid)
            subprocess_status["llama_server"] = {
                "pid": proc.pid,
                "status": proc_info.status(),
                "memory_mb": round(proc_info.memory_info().rss / (1024**2), 2),
                "ready": _process_state.llama_server_ready
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            subprocess_status["llama_server"] = {"status": "not_found"}
    else:
        subprocess_status["llama_server"] = {"status": "not_started"}

    # llama-mmproj-server (多模态)
    proc = _process_state.llama_mmproj_server_process
    if proc:
        try:
            proc_info = psutil.Process(proc.pid)
            subprocess_status["llama_mmproj_server"] = {
                "pid": proc.pid,
                "status": proc_info.status(),
                "memory_mb": round(proc_info.memory_info().rss / (1024**2), 2),
                "ready": _process_state.llama_mmproj_server_ready
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            subprocess_status["llama_mmproj_server"] = {"status": "not_found"}
    else:
        subprocess_status["llama_mmproj_server"] = {"status": "not_started"}

    # 系统资源
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=None)
        system_info = {
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "memory_percent_used": memory.percent,
            "cpu_percent": cpu_percent
        }
    except (psutil.Error, OSError, AttributeError):
        system_info = None

    # 缓存统计
    cache_stats = {
        "active_sessions": len(sessions),
        "media_cache_entries": len(media_understanding_cache),
        "thought_states": len(thought_states)
    }

    response = {
        "status": status,
        "model_loaded": is_ready,
        "backend": DEFAULT_BACKEND,
        "subprocesses": subprocess_status,
        "system": system_info,
        "cache": cache_stats,
        "shutdown_requested": _process_state.shutdown_requested
    }

    # HTTP 状态码: 200 如果就绪，503 如果还在启动
    http_status = 200 if is_ready else 503
    return jsonify(response), http_status

@app.route("/api/status")
def status():
    hw_stats = get_hardware_stats() if model_loaded else {}
    mmproj_paths = [p.strip() for part in LLAMA_MM_PROJ_COMBINED.split(",") for p in part.split(";") if p.strip()]
    mmproj_files = {
        "bin": Path(LLAMA_MTMD_BIN).exists(),
        "model": Path(LLAMA_MM_MODEL).exists(),
        "proj": all(Path(p).exists() for p in mmproj_paths),
    }
    if LLAMA_MM_PROJ_IMAGE:
        mmproj_files["proj_image"] = Path(LLAMA_MM_PROJ_IMAGE).exists()
    if LLAMA_MM_PROJ_AUDIO:
        mmproj_files["proj_audio"] = Path(LLAMA_MM_PROJ_AUDIO).exists()

    # llama.cpp 后端状态
    llama_cpp_ready = Path(LLAMA_RUN_BIN).exists() and LLAMA_RUN_MODEL and Path(LLAMA_RUN_MODEL).exists()
    llama_cpp_files = {
        "bin": Path(LLAMA_RUN_BIN).exists(),
        "model": bool(LLAMA_RUN_MODEL and Path(LLAMA_RUN_MODEL).exists()),
        "model_path": LLAMA_RUN_MODEL or "未找到"
    }

    return jsonify({
        "loaded": model_loaded,
        "stats": stats,
        "memory_gb": get_memory_usage() if model_loaded else 0,
        "hardware": hw_stats,
        "active_sessions": len(sessions),
        "default_backend": DEFAULT_BACKEND,
        "mmproj_ready": all(mmproj_files.values()),
        "mmproj_files": mmproj_files,
        "llama_cpp_ready": llama_cpp_ready,
        "llama_cpp_files": llama_cpp_files,
    })


@app.route("/api/evolution/logs", methods=["GET"])
def evolution_logs():
    if not (EVOLUTION_ENABLED and EVOLUTION_AVAILABLE):
        return jsonify({"error": "Evolution logging disabled"}), 503
    init_evolution_storage()
    if evolution_log_store is None:
        return jsonify({"error": "Evolution log store unavailable"}), 503
    session_id = request.args.get("session_id")
    limit = int(request.args.get("limit", "50"))
    limit = max(1, min(limit, 500))
    since = _parse_datetime(request.args.get("since"))
    until = _parse_datetime(request.args.get("until"))
    logs = list(
        evolution_log_store.list_recent(
            session_id=session_id, limit=limit, start=since, end=until
        )
    )
    return jsonify({"count": len(logs), "logs": [log.to_dict() for log in logs]})

@app.route("/api/session/new", methods=["POST"])
def new_session():
    """创建新会话"""
    session_id = str(uuid.uuid4())[:8]
    session_data = {
        "messages": [],
        "created_at": time.time(),
        "title": "新对话"
    }
    sessions[session_id] = session_data
    # 立即保存到磁盘
    save_session_to_disk(session_id, session_data)
    cleanup_old_sessions()
    return jsonify({"session_id": session_id})

@app.route("/api/session/list", methods=["GET"])
def list_sessions():
    """列出所有会话"""
    session_list = list_all_sessions()
    return jsonify({"sessions": session_list})

@app.route("/api/session/<session_id>/load", methods=["GET"])
def load_session(session_id):
    """加载指定会话"""
    # 先检查内存缓存
    if session_id in sessions:
        session = sessions[session_id]
        return jsonify({
            "session_id": session_id,
            "messages": session["messages"],
            "title": session.get("title", "新对话")
        })

    # 从磁盘查找
    for jsonl_file in SESSIONS_DIR.rglob("*.jsonl"):
        if session_id in jsonl_file.name:
            session_data = load_session_from_disk(jsonl_file)
            if session_data:
                sessions[session_id] = session_data
                return jsonify({
                    "session_id": session_id,
                    "messages": session_data["messages"],
                    "title": session_data.get("title", "新对话")
                })

    return jsonify({"error": "会话不存在"}), 404

@app.route("/api/session/<session_id>/delete", methods=["POST"])
def delete_session(session_id):
    """删除会话"""
    # 从内存移除
    if session_id in sessions:
        file_path = sessions[session_id].get("file_path")
        del sessions[session_id]
        # 删除磁盘文件
        if file_path and Path(file_path).exists():
            Path(file_path).unlink()
            return jsonify({"success": True})

    # 从磁盘查找并删除
    for jsonl_file in SESSIONS_DIR.rglob("*.jsonl"):
        if session_id in jsonl_file.name:
            jsonl_file.unlink()
            return jsonify({"success": True})

    return jsonify({"success": True})

@app.route("/api/session/<session_id>/clear", methods=["POST"])
def clear_session(session_id):
    """清空会话历史"""
    if session_id in sessions:
        sessions[session_id]["messages"] = []
        sessions[session_id]["title"] = "新对话"
        # 保存到磁盘
        save_session_to_disk(session_id, sessions[session_id])
    return jsonify({"success": True})

@app.route("/api/session/<session_id>/history", methods=["GET"])
def get_history(session_id):
    """获取会话历史"""
    if session_id not in sessions:
        return jsonify({"messages": []})
    return jsonify({"messages": sessions[session_id]["messages"]})

@app.route("/api/thought/state/<session_id>", methods=["GET"])
def get_thought_state(session_id):
    """获取会话的 thought signature 状态"""
    state = thought_states.get(session_id, {"turn_index": 0, "media_refs": []})
    media_understandings = []

    for media_ref in state.get("media_refs", []):
        cached = media_understanding_cache.get(media_ref)
        if cached:
            media_understandings.append({
                "media_ref": media_ref,
                "turn_index": cached.get("turn_index", 0),
                "understanding_preview": cached.get("understanding", "")[:100] + "...",
                "created_at": cached.get("created_at", 0)
            })

    return jsonify({
        "session_id": session_id,
        "turn_index": state.get("turn_index", 0),
        "media_count": len(state.get("media_refs", [])),
        "media_understandings": media_understandings
    })

@app.route("/api/thought/stats", methods=["GET"])
def thought_stats():
    """获取 thought signature 全局统计"""
    return jsonify({
        "total_sessions": len(thought_states),
        "total_media_cached": len(media_understanding_cache),
        "sessions": [
            {
                "session_id": sid,
                "turn_index": state.get("turn_index", 0),
                "media_count": len(state.get("media_refs", []))
            }
            for sid, state in thought_states.items()
        ]
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        request_started = time.perf_counter()
        # 请求大小限制
        MAX_REQUEST_SIZE = 500 * 1024 * 1024  # 500MB
        MAX_IMAGE_SIZE = 50 * 1024 * 1024     # 50MB per image
        MAX_AUDIO_SIZE = 100 * 1024 * 1024    # 100MB per audio
        MAX_TEXT_LENGTH = 50000               # 50k chars

        if request.content_length and request.content_length > MAX_REQUEST_SIZE:
            return jsonify({"error": f"请求过大 (最大 {MAX_REQUEST_SIZE // 1024 // 1024}MB)"}), 413

        data = request.json
        text = data.get("text", "")

        # 验证文本长度
        if len(text) > MAX_TEXT_LENGTH:
            return jsonify({"error": f"文本过长 (最大 {MAX_TEXT_LENGTH} 字符)"}), 400

        # 支持单个或多个图片/音频 (兼容旧API)
        image_data = data.get("image")  # 单个图片 (向后兼容)
        images_data = data.get("images", [])  # 多个图片 (新API)
        audio_data = data.get("audio")  # 单个音频 (向后兼容)
        audios_data = data.get("audios", [])  # 多个音频 (新API)
        session_id = data.get("session_id")
        backend = data.get("backend") or DEFAULT_BACKEND

        # 合并单个和多个文件
        if image_data and image_data not in images_data:
            images_data = [image_data] + images_data
        if audio_data and audio_data not in audios_data:
            audios_data = [audio_data] + audios_data

        # 限制最多14张图片，10个音频 (与MPS模式保持一致)
        MAX_IMAGES = 14
        MAX_AUDIOS = 10
        if len(images_data) > MAX_IMAGES:
            return jsonify({"error": f"最多支持 {MAX_IMAGES} 张图片，当前 {len(images_data)} 张"}), 400
        if len(audios_data) > MAX_AUDIOS:
            return jsonify({"error": f"最多支持 {MAX_AUDIOS} 个音频，当前 {len(audios_data)} 个"}), 400

        # 验证单个文件大小
        for i, img in enumerate(images_data):
            try:
                img_base64 = img.split(",")[1] if "," in img else img
                if len(base64.b64decode(img_base64)) > MAX_IMAGE_SIZE:
                    return jsonify({"error": f"图片 {i+1} 过大 (最大 {MAX_IMAGE_SIZE // 1024 // 1024}MB)"}), 400
            except Exception:
                pass  # 解码失败会在后续处理中报错

        for i, aud in enumerate(audios_data):
            try:
                aud_base64 = aud.split(",")[1] if "," in aud else aud
                if len(base64.b64decode(aud_base64)) > MAX_AUDIO_SIZE:
                    return jsonify({"error": f"音频 {i+1} 过大 (最大 {MAX_AUDIO_SIZE // 1024 // 1024}MB)"}), 400
            except Exception:
                pass  # 解码失败会在后续处理中报错

        import sys
        # 日志中不记录用户输入内容，保护隐私
        logger.info(f"[API /api/chat] backend={backend}, images={len(images_data)}, audios={len(audios_data)}, has_text={bool(text)}")
        sys.stdout.flush()

        # 获取或创建会话
        if not session_id or session_id not in sessions:
            # 尝试从磁盘加载
            loaded = False
            if session_id:
                for jsonl_file in SESSIONS_DIR.rglob("*.jsonl"):
                    if session_id in jsonl_file.name:
                        session_data = load_session_from_disk(jsonl_file)
                        if session_data:
                            sessions[session_id] = session_data
                            loaded = True
                            break
            if not loaded:
                session_id = str(uuid.uuid4())[:8]
                sessions[session_id] = {"messages": [], "created_at": time.time(), "title": "新对话"}

        session = sessions[session_id]
        turn_index = len(session["messages"]) // 2 + 1
        images = []  # PIL Image 对象列表 (MPS模式用)
        audios = []  # (audio_array, sr) 元组列表 (MPS模式用)
        image_paths = []  # 图片路径列表 (mmproj模式用)
        audio_paths = []  # 音频路径列表 (mmproj模式用)

        # 处理多个图片
        for idx, img_data in enumerate(images_data):
            try:
                if "," in img_data:
                    img_data = img_data.split(",")[1]
                image_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                images.append(img)
                # 保存图片供 mmproj 使用，并可持久化到 Evolution 存储
                img_path = _build_media_path(
                    session_id=session_id,
                    turn_index=turn_index,
                    index=idx,
                    kind="image",
                    ext=".png",
                )
                img.save(img_path)
                image_paths.append(img_path)
            except OSError as e:
                return jsonify({"error": f"保存图片失败: {e}"}), 500
            except Exception as e:
                return jsonify({"error": f"处理图片 {idx+1} 失败: {e}"}), 400

        # 处理多个音频
        for idx, aud_data in enumerate(audios_data):
            try:
                mime_part = ""
                if "," in aud_data:
                    mime_part = aud_data.split(",")[0]
                    aud_data = aud_data.split(",")[1]
                audio_bytes = base64.b64decode(aud_data)

                if "wav" in mime_part:
                    ext = ".wav"
                elif "webm" in mime_part:
                    ext = ".webm"
                elif "ogg" in mime_part:
                    ext = ".ogg"
                elif "mp3" in mime_part or "mpeg" in mime_part:
                    ext = ".mp3"
                elif "flac" in mime_part:
                    ext = ".flac"
                else:
                    ext = ".wav"

                temp_path = _build_media_path(
                    session_id=session_id,
                    turn_index=turn_index,
                    index=idx,
                    kind="audio",
                    ext=ext,
                )
                with open(temp_path, "wb") as f:
                    f.write(audio_bytes)

                audio_array, sr = librosa.load(temp_path, sr=16000)
                audios.append((audio_array, sr))
                audio_paths.append(temp_path)
            except OSError as e:
                return jsonify({"error": f"保存音频失败: {e}"}), 500
            except Exception as e:
                return jsonify({"error": f"处理音频 {idx+1} 失败: {e}"}), 400
            logger.debug(f" 音频 {idx}: {len(audio_array)/sr:.2f}秒")

        # 向后兼容: 单个变量
        image = images[0] if images else None
        audio = audios[0] if audios else None
        image_path = image_paths[0] if image_paths else None
        audio_path = audio_paths[0] if audio_paths else None

        # 构建当前消息内容
        content = []
        has_media = (image is not None or audio is not None)
        display_text = text

        if backend == "mps":
            # MPS 模式需要 PyTorch
            if not PYTORCH_AVAILABLE:
                return jsonify({
                    "error": "MPS 后端需要安装 PyTorch (进阶功能)",
                    "hint": "请使用 mmproj 后端，或安装: pip install torch transformers librosa"
                }), 400

            if not has_media:
                # 纯文本消息：添加 dummy_image
                content.append({"type": "image", "image": dummy_image})
                display_text = text
                text = "Ignore the blank image. " + text

            # Gemma 3n 要求每条消息都有图片
            # 如果只有音频没有图片，也需要添加 dummy_image
            if image is not None:
                content.append({"type": "image", "image": image})
            elif audio is not None:
                # 只有音频，添加 dummy_image
                content.append({"type": "image", "image": dummy_image})
                text = "Ignore the blank image. " + text

            if audio is not None:
                content.append({"type": "audio", "audio": audio[0], "sample_rate": audio[1]})

            content.append({"type": "text", "text": text})

        # 生成回复 (传入 session_id 和 has_media)
        # 确定当前媒体类型
        current_media_type = None
        if image is not None:
            current_media_type = "image"
        elif audio is not None:
            current_media_type = "audio"

        # 生成回复 (传入 session_id, has_media, media_type)
        if backend == "mps":
            result = generate_response(
                session["messages"],
                content,
                session_id=session_id,
                has_media=has_media,
                media_type=current_media_type
            )
        elif backend == "llama.cpp":
            # llama.cpp 纯文本模式（使用 llama-server 持久化服务）
            # 注意：只支持文本，不支持图片/音频
            if has_media:
                result = {"error": "llama.cpp 后端暂不支持图片/音频，请切换到 MPS 或 mmproj 后端"}
            else:
                # 构建历史上下文
                history_context = ""
                if session["messages"]:
                    history_parts = []
                    for msg in session["messages"][-MAX_HISTORY_TURNS * 2:]:
                        role = "User" if msg["role"] == "user" else "Assistant"
                        history_parts.append(f"{role}: {msg['text']}")
                    history_context = "\n".join(history_parts)

                # 使用 llama-server 持久化服务（更快）
                result = query_llama_server(text, history_context)
        else:
            # llama.cpp/mmproj 模式
            # 传入历史消息和 session_id，支持多轮对话和 thought signature
            mm_prompt = text or "Please describe what you see/hear."
            # 使用多文件路径 (如有)，否则回退到单文件
            mm_image_paths = image_paths if image_paths else None
            mm_audio_paths = audio_paths if audio_paths else None
            # 日志中不记录用户输入内容
            logger.info(f"API mmproj] images={len(image_paths)}, audios={len(audio_paths)}")

            result = run_llama_mmproj(
                mm_prompt,
                image_path=mm_image_paths,  # 支持单个路径或路径列表
                audio_path=mm_audio_paths,  # 支持单个路径或路径列表
                messages_history=session["messages"],
                session_id=session_id,
                has_media=has_media
            )

        if "error" not in result:
            # 如果有媒体输入，从模型回复中提取理解并存储到 thought signature
            media_ref = None
            if has_media:
                # 模型的回复就是对媒体的理解
                # 将其存储为 "压缩记忆"
                understanding = result["response"]

                # 生成媒体理解签名
                media_ref = generate_media_signature(
                    session_id=session_id,
                    turn_index=turn_index,
                    understanding=understanding[:500]  # 限制长度，只保存摘要
                )
                logger.debug(f"ThoughtSignature: 存储 {current_media_type} 理解: {media_ref}")

            # 保存到历史（只保存文本摘要）
            user_summary = display_text
            if len(images) > 0:
                user_summary = f"[{len(images)}张图片] " + user_summary
            if len(audios) > 0:
                user_summary = f"[{len(audios)}个音频] " + user_summary

            session["messages"].append({
                "role": "user",
                "text": user_summary,
                "has_image": len(images) > 0,
                "has_audio": len(audios) > 0,
                "image_count": len(images),
                "audio_count": len(audios),
                "timestamp": time.time()
            })
            session["messages"].append({
                "role": "assistant",
                "text": result["response"],
                "timestamp": time.time()
            })

            # 限制历史长度
            if len(session["messages"]) > MAX_HISTORY_TURNS * 2:
                session["messages"] = session["messages"][-MAX_HISTORY_TURNS * 2:]

            # 更新标题 (用第一条用户消息)
            if session.get("title") == "新对话" and len(session["messages"]) >= 1:
                first_user_msg = next((m for m in session["messages"] if m["role"] == "user"), None)
                if first_user_msg:
                    session["title"] = first_user_msg["text"][:30] + ("..." if len(first_user_msg["text"]) > 30 else "")

            # 保存到磁盘
            save_session_to_disk(session_id, session)
            # 追加到全局历史
            append_to_history(session_id, display_text)

            response_time_ms = int((time.perf_counter() - request_started) * 1000)
            feedback_data = data.get("feedback")
            implicit_feedback_score = data.get("implicit_feedback_score")
            retrieved_doc_ids = data.get("retrieved_doc_ids")
            embedding = data.get("multimodal_embedding") or data.get("embedding")
            if embedding is None:
                service = get_embedding_service()
                if service is not None:
                    if EVOLUTION_EMBED_MEDIA and image_paths:
                        embedding = service.embed_image_paths(image_paths)
                    elif EVOLUTION_EMBED_TEXT and display_text:
                        embedding = service.embed_text(display_text)
            _record_evolution_interaction(
                session_id=session_id,
                turn_index=turn_index,
                user_text=display_text,
                response_text=result["response"],
                image_paths=image_paths,
                audio_paths=audio_paths,
                backend=backend,
                response_time_ms=response_time_ms,
                thought_signature=media_ref,
                feedback_data=feedback_data,
                implicit_feedback_score=implicit_feedback_score,
                retrieved_doc_ids=retrieved_doc_ids,
                embedding=embedding,
            )

        result["session_id"] = session_id
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    init_storage()

    # 启动缓存清理线程
    start_cache_cleanup_thread()

    # macOS: 请求 sudo 权限用于硬件监控 (可选)
    if platform.system() == "Darwin" and DEFAULT_BACKEND == "mps":
        request_sudo_permission()

    load_model()
    port = int(os.environ.get("WEBUI_PORT", 5001))
    logger.info("=" * 60)
    logger.info("灵空 AI 多模态聊天服务器")
    logger.info("=" * 60)
    logger.info(f"  地址: http://localhost:{port}")
    logger.info(f"  后端: {DEFAULT_BACKEND}")
    if DEFAULT_BACKEND == "mmproj":
        logger.info(f"  模型: {LLAMA_MM_MODEL}")
        logger.info(f"  视觉: {LLAMA_MM_PROJ_IMAGE}")
        logger.info(f"  音频: {LLAMA_MM_PROJ_AUDIO}")
    logger.info(f"  存储: {GEMMA3N_HOME}")
    if DEFAULT_BACKEND == "mmproj":
        logger.info("  提示: 使用 llama.cpp 多模态后端，无需 PyTorch")
        logger.info("  进阶: export GEMMA3N_BACKEND=mps (需要 PyTorch)")
    if sudo_authorized:
        logger.info("  GPU 温度监控: 已启用")
    logger.info("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

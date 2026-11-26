"""
Kaggle Environment Detection and Configuration
===============================================

Provides runtime detection of Kaggle environment, available hardware (GPU, VRAM),
CUDA availability, and filesystem paths. Configures AI models based on detected
hardware with graceful CPU fallback.

This module does NOT modify the original Clips Kitty pipeline. It provides an
adapter layer that can be imported and used to configure the pipeline for Kaggle.
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class ComputeBackend(Enum):
    """Available compute backends for model inference."""
    CUDA = "cuda"
    CPU = "cpu"
    AUTO = "auto"


class GPUType(Enum):
    """Known GPU types in Kaggle."""
    UNKNOWN = "unknown"
    T4 = "T4"           # 16 GB VRAM, sm_75
    P100 = "P100"       # 16 GB VRAM, sm_60
    V100 = "V100"       # 16/32 GB VRAM, sm_70
    A100 = "A100"       # 40/80 GB VRAM, sm_80
    A100_80GB = "A100-80GB"
    L4 = "L4"           # 24 GB VRAM, sm_89


@dataclass
class GPUInfo:
    """Information about detected GPU."""
    name: str = "Unknown"
    type: GPUType = GPUType.UNKNOWN
    vram_gb: float = 0.0
    compute_capability: str = "unknown"
    driver_version: str = "unknown"
    cuda_version: str = "unknown"
    is_available: bool = False
    index: int = 0


@dataclass
class KaggleEnvironment:
    """Complete Kaggle environment detection result."""
    is_kaggle: bool = False
    kernel_type: str = "unknown"  # "notebook", "script", "interactive"
    username: str = "unknown"
    kernel_slug: str = "unknown"

    # GPU info
    gpus: List[GPUInfo] = field(default_factory=list)
    primary_gpu: Optional[GPUInfo] = None
    total_vram_gb: float = 0.0

    # CUDA info
    cuda_available: bool = False
    cuda_version: str = "unknown"
    cublas12_available: bool = False

    # CPU info
    cpu_count: int = 1
    cpu_architecture: str = "unknown"
    total_ram_gb: float = 0.0

    # Filesystem paths
    working_dir: Path = Path.cwd()
    input_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    data_dir: Path = Path("/data")  # Default persistent volume

    # Secrets / API keys (from Kaggle Secrets)
    secrets: Dict[str, str] = field(default_factory=dict)

    # Recommended configuration based on hardware
    recommended_config: Dict[str, Any] = field(default_factory=dict)


def detect_kaggle_environment() -> KaggleEnvironment:
    """
    Detect if running in Kaggle and gather environment information.

    Returns:
        KaggleEnvironment with all detected information and recommendations.
    """
    env = KaggleEnvironment()

    # --- Kaggle detection ---
    env.is_kaggle = _detect_kaggle()
    if env.is_kaggle:
        env.kernel_type = _get_kaggle_kernel_type()
        env.username = _get_kaggle_username()
        env.kernel_slug = _get_kaggle_kernel_slug()

    # --- GPU detection ---
    env.gpus = _detect_gpus()
    if env.gpus:
        env.primary_gpu = env.gpus[0]
        env.total_vram_gb = sum(g.vram_gb for g in env.gpus)
        env.cuda_available = _check_cuda_available()
        env.cuda_version = _get_cuda_version()
        env.cublas12_available = _check_cublas12()

    # --- CPU detection ---
    env.cpu_count = os.cpu_count() or 1
    env.cpu_architecture = _get_cpu_architecture()
    env.total_ram_gb = _get_total_ram_gb()

    # --- Filesystem paths ---
    env.working_dir = Path.cwd()
    env.input_dir = _get_kaggle_input_dir()
    env.output_dir = _get_kaggle_output_dir()
    env.data_dir = _get_data_dir()

    # --- Secrets ---
    env.secrets = _load_kaggle_secrets()

    # --- Generate recommendations ---
    env.recommended_config = _generate_recommendations(env)

    return env


def _detect_kaggle() -> bool:
    """Check if running inside Kaggle kernel."""
    # KAGGLE_KERNEL_RUN_TYPE is set in Kaggle kernels
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return True
    # KAGGLE_URL_BASE is another indicator
    if "KAGGLE_URL_BASE" in os.environ:
        return True
    # Check for typical Kaggle paths
    if Path("/kaggle").exists():
        return True
    return False


def _get_kaggle_kernel_type() -> str:
    """Get Kaggle kernel type (notebook, script, interactive)."""
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "unknown")


def _get_kaggle_username() -> str:
    """Get Kaggle username."""
    return os.environ.get("KAGGLE_USER_NAME", "unknown")


def _get_kaggle_kernel_slug() -> str:
    """Get Kaggle kernel slug."""
    return os.environ.get("KAGGLE_KERNEL_SLUG", "unknown")


def _detect_gpus() -> List[GPUInfo]:
    """Detect NVIDIA GPUs using nvidia-smi and nvidia-ml-py."""
    gpus = []

    # Try nvidia-ml-py first (more reliable in containers)
    try:
        import nvidia_ml_py3 as nvml
        nvml.nvmlInit()
        device_count = nvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            name = nvml.nvmlDeviceGetName(handle).decode("utf-8")
            mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            vram_gb = mem_info.total / (1024 ** 3)

            # Get compute capability
            try:
                cc_major = nvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
                cc_minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
                compute_capability = f"sm_{cc_major}{cc_minor}"
            except Exception:
                compute_capability = "unknown"

            gpu_type = _identify_gpu_type(name, vram_gb, compute_capability)

            gpus.append(GPUInfo(
                name=name,
                type=gpu_type,
                vram_gb=round(vram_gb, 1),
                compute_capability=compute_capability,
                is_available=True,
                index=i
            ))
        nvml.nvmlShutdown()
    except Exception:
        # Fallback to nvidia-smi
        gpus = _detect_gpus_nvidia_smi()

    return gpus


def _identify_gpu_type(name: str, vram_gb: float, compute_capability: str) -> GPUType:
    """Identify GPU type from name, VRAM, and compute capability."""
    name_lower = name.lower()

    if "t4" in name_lower or compute_capability == "sm_75":
        return GPUType.T4
    if "p100" in name_lower or compute_capability == "sm_60":
        return GPUType.P100
    if "v100" in name_lower or compute_capability == "sm_70":
        return GPUType.V100
    if "a100" in name_lower:
        if vram_gb >= 70:
            return GPUType.A100_80GB
        return GPUType.A100
    if "l4" in name_lower or compute_capability == "sm_89":
        return GPUType.L4

    return GPUType.UNKNOWN


def _detect_gpus_nvidia_smi() -> List[GPUInfo]:
    """Fallback GPU detection using nvidia-smi command."""
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    idx = int(parts[0])
                    name = parts[1]
                    vram_mb = int(parts[2])
                    vram_gb = vram_mb / 1024
                    driver = parts[3] if len(parts) > 3 else "unknown"

                    # Try to get compute capability
                    cc = _get_compute_capability(idx)

                    gpu_type = _identify_gpu_type(name, vram_gb, cc)

                    gpus.append(GPUInfo(
                        name=name,
                        type=gpu_type,
                        vram_gb=round(vram_gb, 1),
                        compute_capability=cc,
                        driver_version=driver,
                        is_available=True,
                        index=idx
                    ))
    except Exception:
        pass
    return gpus


def _get_compute_capability(gpu_index: int) -> str:
    """Get compute capability for a specific GPU index."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            cc = result.stdout.strip()
            return f"sm_{cc.replace('.', '')}"
    except Exception:
        pass
    return "unknown"


def _check_cuda_available() -> bool:
    """Check if CUDA is available for PyTorch/TensorFlow."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        try:
            import tensorflow as tf
            return len(tf.config.list_physical_devices("GPU")) > 0
        except Exception:
            return False


def _get_cuda_version() -> str:
    """Get CUDA version."""
    try:
        import torch
        return torch.version.cuda or "unknown"
    except Exception:
        try:
            result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "release":
                            return parts[i + 1].rstrip(",")
        except Exception:
            pass
    return "unknown"


def _check_cublas12() -> bool:
    """Check if cuBLAS 12 is available (required for faster-whisper GPU)."""
    # Check via ctypes (Linux)
    if sys.platform != "win32":
        try:
            import ctypes
            ctypes.CDLL("libcublas.so.12")
            ctypes.CDLL("libcublasLt.so.12")
            return True
        except Exception:
            pass
    else:
        # Windows check
        try:
            import ctypes
            ctypes.WinDLL("cublas64_12.dll")
            ctypes.WinDLL("cublasLt64_12.dll")
            return True
        except Exception:
            pass

    # Also check nvidia-cublas-cu12 package
    try:
        import nvidia.cublas
        return True
    except Exception:
        pass

    return False


def _get_cpu_architecture() -> str:
    """Get CPU architecture."""
    try:
        import platform
        return platform.machine()
    except Exception:
        return "unknown"


def _get_total_ram_gb() -> float:
    """Get total system RAM in GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        # Fallback: read from /proc/meminfo
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 ** 2), 1)
        except Exception:
            pass
    return 0.0


def _get_kaggle_input_dir() -> Optional[Path]:
    """Get Kaggle input directory (/kaggle/input)."""
    path = Path("/kaggle/input")
    return path if path.exists() else None


def _get_kaggle_output_dir() -> Optional[Path]:
    """Get Kaggle output directory (/kaggle/working)."""
    path = Path("/kaggle/working")
    return path if path.exists() else None


def _get_data_dir() -> Path:
    """Get persistent data directory, respecting CLIPS_STUDIO_DATA_DIR."""
    # Priority 1: Environment variable (set by user or Docker)
    env_data = os.environ.get("CLIPS_STUDIO_DATA_DIR")
    if env_data:
        return Path(env_data)

    # Priority 2: Kaggle persistent volume convention
    kaggle_data = Path("/data")
    if kaggle_data.exists() or os.access("/", os.W_OK):
        return kaggle_data

    # Priority 3: Working directory fallback
    return Path.cwd() / "data"


def _load_kaggle_secrets() -> Dict[str, str]:
    """Load secrets from Kaggle Secrets (mounted at /kaggle/secrets)."""
    secrets = {}
    secrets_dir = Path("/kaggle/secrets")
    if secrets_dir.exists():
        for secret_file in secrets_dir.iterdir():
            if secret_file.is_file():
                try:
                    secrets[secret_file.name] = secret_file.read_text().strip()
                except Exception:
                    pass
    return secrets


def _generate_recommendations(env: KaggleEnvironment) -> Dict[str, Any]:
    """Generate recommended configuration based on detected hardware."""
    config = {}

    # --- Transcription ---
    if env.primary_gpu and env.primary_gpu.is_available and env.cublas12_available:
        config["transcription"] = {
            "model": "large-v3-turbo",
            "device": "cuda",
            "compute_type": "float16",
            "batch_size": _get_whisper_batch_size(env.primary_gpu.vram_gb),
        }
    else:
        config["transcription"] = {
            "model": "small",
            "device": "cpu",
            "compute_type": "int8",
            "batch_size": 1,
        }

    # --- Video rendering parallelism ---
    if env.primary_gpu:
        vram = env.primary_gpu.vram_gb
        if vram >= 40:  # A100 40GB/80GB
            config["video"] = {"parallel_renders": 2}
        elif vram >= 24:  # L4
            config["video"] = {"parallel_renders": 2}
        elif vram >= 16:  # T4, P100, V100
            config["video"] = {"parallel_renders": 1}
        else:
            config["video"] = {"parallel_renders": 1}
    else:
        config["video"] = {"parallel_renders": 1}

    # Encoder selection
    if env.primary_gpu and env.primary_gpu.is_available:
        config["video"]["encoder"] = "auto"  # Will pick NVENC
    else:
        config["video"]["encoder"] = "libx264"

    # --- Tracking ---
    if env.primary_gpu and env.primary_gpu.is_available:
        config["tracking"] = {
            "enabled": True,
            "yolo_model": "yolov8n-pose.pt",
            "talknet_model": "pretrain_TalkSet.model",
            "sample_every": 1,
            "device": "cuda",
        }
    else:
        config["tracking"] = {
            "enabled": True,
            "yolo_model": "yolov8n-pose.pt",
            "talknet_model": "pretrain_TalkSet.model",
            "sample_every": 1,
            "device": "cpu",
        }

    # --- LLM backend ---
    # Prefer cloud LLM if API keys available
    has_openrouter = bool(env.secrets.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY"))
    has_openai = bool(env.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(env.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))

    if has_openrouter:
        config["llm"] = {"backend": "openrouter", "model": "anthropic/claude-3.5-sonnet"}
    elif has_openai:
        config["llm"] = {"backend": "openai", "model": "gpt-4o"}
    elif has_anthropic:
        config["llm"] = {"backend": "anthropic", "model": "claude-3-5-sonnet-20241022"}
    else:
        # Fallback to local Ollama (requires sidecar)
        config["llm"] = {"backend": "ollama", "model": "llama3.1"}

    # --- Paths ---
    config["paths"] = {"data_dir": str(env.data_dir)}

    # --- YouTube cookies ---
    if env.secrets.get("YOUTUBE_COOKIES") or (env.data_dir / "youtube_cookies.txt").exists():
        config["youtube"] = {"cookies": str(env.data_dir / "youtube_cookies.txt")}

    # --- General ---
    config["general"] = {"headless": True}

    return config


def _get_whisper_batch_size(vram_gb: float) -> int:
    """Determine optimal Whisper batch size based on VRAM."""
    if vram_gb >= 40:
        return 16
    if vram_gb >= 24:
        return 12
    if vram_gb >= 16:
        return 8
    return 4


def apply_kaggle_config(config: Dict[str, Any], env: Optional[KaggleEnvironment] = None) -> Dict[str, Any]:
    """
    Apply Kaggle-specific configuration overrides to a base config dict.

    Args:
        config: Base configuration dictionary (from settings.yaml)
        env: Pre-detected environment (optional, will detect if not provided)

    Returns:
        Updated configuration dictionary with Kaggle optimizations.
    """
    if env is None:
        env = detect_kaggle_environment()

    # Deep merge recommendations
    for key, value in env.recommended_config.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict) and isinstance(config[key], dict):
            config[key].update(value)
        else:
            config[key] = value

    # Ensure data_dir is absolute
    if "paths" in config and "data_dir" in config["paths"]:
        config["paths"]["data_dir"] = str(Path(config["paths"]["data_dir"]).resolve())

    # Inject secrets as environment variables for downstream libraries
    for secret_name, secret_value in env.secrets.items():
        env_var = secret_name.upper()
        if env_var not in os.environ:
            os.environ[env_var] = secret_value

    return config


def get_optimal_device(env: Optional[KaggleEnvironment] = None) -> ComputeBackend:
    """
    Determine the optimal compute device for the current environment.

    Returns:
        ComputeBackend.CUDA if GPU with cuBLAS12 available, else CPU.
    """
    if env is None:
        env = detect_kaggle_environment()

    if env.primary_gpu and env.primary_gpu.is_available and env.cublas12_available:
        return ComputeBackend.CUDA
    return ComputeBackend.CPU


def print_environment_summary(env: Optional[KaggleEnvironment] = None) -> None:
    """Print a human-readable summary of the detected environment."""
    if env is None:
        env = detect_kaggle_environment()

    print("=" * 60)
    print("KAGGLE ENVIRONMENT DETECTION")
    print("=" * 60)
    print(f"Is Kaggle:        {env.is_kaggle}")
    if env.is_kaggle:
        print(f"Kernel Type:      {env.kernel_type}")
        print(f"Username:         {env.username}")
        print(f"Kernel Slug:      {env.kernel_slug}")
    print(f"Working Dir:      {env.working_dir}")
    print(f"Input Dir:        {env.input_dir}")
    print(f"Output Dir:       {env.output_dir}")
    print(f"Data Dir:         {env.data_dir}")
    print(f"CPU Cores:        {env.cpu_count}")
    print(f"CPU Arch:         {env.cpu_architecture}")
    print(f"System RAM:       {env.total_ram_gb} GB")
    print(f"CUDA Available:   {env.cuda_available}")
    print(f"CUDA Version:     {env.cuda_version}")
    print(f"cuBLAS 12:        {env.cublas12_available}")
    print(f"GPUs Found:       {len(env.gpus)}")
    for gpu in env.gpus:
        print(f"  GPU {gpu.index}: {gpu.name} ({gpu.vram_gb} GB, {gpu.compute_capability})")
    if env.primary_gpu:
        print(f"Primary GPU:      {env.primary_gpu.name} ({env.primary_gpu.vram_gb} GB)")
    print(f"Total VRAM:       {env.total_vram_gb} GB")
    print(f"Optimal Device:   {get_optimal_device(env).value}")
    print(f"Secrets Loaded:   {len(env.secrets)}")
    print("-" * 60)
    print("RECOMMENDED CONFIG:")
    for section, settings in env.recommended_config.items():
        print(f"  {section}:")
        for k, v in settings.items():
            print(f"    {k}: {v}")
    print("=" * 60)


# Module-level singleton for easy access
_environment_cache: Optional[KaggleEnvironment] = None


def get_environment() -> KaggleEnvironment:
    """Get cached environment detection result."""
    global _environment_cache
    if _environment_cache is None:
        _environment_cache = detect_kaggle_environment()
    return _environment_cache


def reset_environment_cache() -> None:
    """Reset the cached environment (useful for testing)."""
    global _environment_cache
    _environment_cache = None
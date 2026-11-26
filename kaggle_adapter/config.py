"""
Kaggle Configuration Adapter
============================

Provides configuration utilities for running Clips Kitty in Kaggle notebooks.
This module creates a bridge between Kaggle environment detection and the
Clips Kitty configuration system (settings.yaml + environment variables).

Does NOT modify the original Clips Kitty pipeline. Provides adapter functions
that can be called before running the pipeline.
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .environment import (
    detect_kaggle_environment,
    KaggleEnvironment,
    apply_kaggle_config,
    get_optimal_device,
    ComputeBackend,
)


@dataclass
class KaggleConfig:
    """Configuration bundle for Kaggle execution."""
    environment: KaggleEnvironment
    settings: Dict[str, Any]
    config_path: Optional[Path] = None


DEFAULT_KAGGLE_SETTINGS = {
    "general": {
        "headless": True,
    },
    "paths": {
        "data_dir": "/data",
    },
    "transcription": {
        "model": "auto",  # Will be overridden by environment detection
        "device": "auto",
        "compute_type": "auto",
        "batch_size": "auto",
    },
    "clips": {
        "vertical": {
            "enabled": True,
            "crop_mode": "track",
            "max_duration": 60,
            "min_duration": 15,
        },
        "scoring": {
            "min_score": 0.6,
        },
        "captions": {
            "enabled": True,
            "font": "Arial",
            "font_size": 84,
            "color": "#FFFFFF",
            "position": "bottom",
            "words_per_caption": 3,
            "uppercase": True,
            "highlight": True,
            "highlight_color": "#FFE600",
            "second_speaker": True,
            "second_speaker_color": "#5CE1FF",
        },
    },
    "video": {
        "encoder": "auto",
        "parallel_renders": 1,
        "preset": "p5",
        "bitrate": "5M",
        "outro": {
            "enabled": True,
            "mascot": "clippy",
        },
    },
    "tracking": {
        "enabled": True,
        "yolo_model": "yolov8n-pose.pt",
        "talknet_model": "pretrain_TalkSet.model",
        "sample_every": 1,
        "device": "auto",
    },
    "llm": {
        "backend": "openrouter",
        "model": "anthropic/claude-3.5-sonnet",
    },
    "youtube": {
        "cookies": "/data/youtube_cookies.txt",
    },
}


def load_base_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load base configuration from settings.yaml or defaults.

    Args:
        config_path: Optional path to settings.yaml. If not provided,
                     searches standard locations.

    Returns:
        Configuration dictionary with defaults applied.
    """
    config = DEFAULT_KAGGLE_SETTINGS.copy()

    # Search for settings.yaml in standard locations
    search_paths = []
    if config_path:
        search_paths.append(Path(config_path))
    else:
        search_paths.extend([
            Path("config/settings.yaml"),
            Path("../config/settings.yaml"),
            Path("/app/config/settings.yaml"),
            Path.home() / ".config" / "clips-kitty" / "settings.yaml",
        ])

    for path in search_paths:
        if path.exists():
            try:
                with open(path, "r") as f:
                    user_config = yaml.safe_load(f) or {}
                config = _deep_merge(config, user_config)
                break
            except Exception:
                continue

    return config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, handling bool-to-dict conversion for clips settings."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        elif key in result and isinstance(result[key], dict) and isinstance(value, bool):
            # Handle original settings.yaml format where clips.vertical/captions/outro are booleans
            # Keep the base dict structure but update the 'enabled' key
            result[key] = result[key].copy()
            result[key]["enabled"] = value
        else:
            result[key] = value
    return result


def create_kaggle_config(
    config_path: Optional[Path] = None,
    environment: Optional[KaggleEnvironment] = None,
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> KaggleConfig:
    """
    Create a complete Kaggle-ready configuration.

    This is the main entry point for configuring Clips Kitty for Kaggle.

    Args:
        config_path: Optional path to base settings.yaml
        environment: Pre-detected environment (will detect if not provided)
        extra_overrides: Additional configuration overrides

    Returns:
        KaggleConfig with environment, merged settings, and config path.
    """
    if environment is None:
        environment = detect_kaggle_environment()

    # Load base config
    settings = load_base_config(config_path)

    # Apply Kaggle-specific optimizations
    settings = apply_kaggle_config(settings, environment)

    # Apply any extra overrides (highest priority)
    if extra_overrides:
        settings = _deep_merge(settings, extra_overrides)

    # Resolve paths
    settings = _resolve_paths(settings, environment)

    return KaggleConfig(
        environment=environment,
        settings=settings,
        config_path=config_path,
    )


def _resolve_paths(config: Dict[str, Any], env: KaggleEnvironment) -> Dict[str, Any]:
    """Resolve all paths to absolute paths."""
    # Data directory
    if "paths" in config and "data_dir" in config["paths"]:
        data_dir = Path(config["paths"]["data_dir"])
        if not data_dir.is_absolute():
            data_dir = env.data_dir / data_dir
        config["paths"]["data_dir"] = str(data_dir.resolve())

    # YouTube cookies
    if "youtube" in config and "cookies" in config["youtube"]:
        cookies_path = Path(config["youtube"]["cookies"])
        if not cookies_path.is_absolute():
            cookies_path = env.data_dir / cookies_path
        config["youtube"]["cookies"] = str(cookies_path.resolve())

    # Model paths - ensure they point to pre-cached models
    if "tracking" in config:
        tracking = config["tracking"]
        # YOLO models from ultralytics cache
        if "yolo_model" in tracking:
            model_path = Path(tracking["yolo_model"])
            if not model_path.is_absolute():
                cached = Path("/app/models") / model_path.name
                if cached.exists():
                    tracking["yolo_model"] = str(cached)
                else:
                    tracking["yolo_model"] = str(Path.home() / ".cache" / "ultralytics" / model_path.name)
        # TalkNet model from third_party/talknet or pre-cached
        if "talknet_model" in tracking:
            model_path = Path(tracking["talknet_model"])
            if not model_path.is_absolute():
                cached = Path("/app/models") / model_path.name
                if cached.exists():
                    tracking["talknet_model"] = str(cached)
                else:
                    # TalkNet is in third_party/talknet/
                    tracking["talknet_model"] = str(Path("third_party") / "talknet" / model_path.name)

    return config


def save_config(config: KaggleConfig, output_path: Path) -> None:
    """Save configuration to YAML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config.settings, f, default_flow_style=False, sort_keys=False)


def get_device_config(env: Optional[KaggleEnvironment] = None) -> Dict[str, str]:
    """
    Get device configuration for transcription and tracking.

    Returns:
        Dict with 'device' and 'compute_type' keys.
    """
    if env is None:
        env = detect_kaggle_environment()

    device = get_optimal_device(env)

    if device == ComputeBackend.CUDA:
        return {
            "device": "cuda",
            "compute_type": "float16",
        }
    else:
        return {
            "device": "cpu",
            "compute_type": "int8",
        }


def get_video_encoder_config(env: Optional[KaggleEnvironment] = None) -> Dict[str, Any]:
    """
    Get video encoder configuration based on available hardware.

    Returns:
        Dict with encoder settings.
    """
    if env is None:
        env = detect_kaggle_environment()

    if env.primary_gpu and env.primary_gpu.is_available:
        return {
            "encoder": "auto",  # Will probe and pick NVENC
            "preset": "p5",
        }
    else:
        return {
            "encoder": "libx264",
            "preset": "medium",
        }


def get_parallel_renders(env: Optional[KaggleEnvironment] = None) -> int:
    """Get recommended number of parallel render workers."""
    if env is None:
        env = detect_kaggle_environment()

    if env.primary_gpu:
        vram = env.primary_gpu.vram_gb
        if vram >= 40:
            return 2
        if vram >= 24:
            return 2
        if vram >= 16:
            return 1
    return 1


def setup_kaggle_environment(
    config_path: Optional[Path] = None,
    extra_overrides: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> KaggleConfig:
    """
    Complete Kaggle environment setup.

    Call this once at the start of your Kaggle notebook to configure
    everything for Clips Kitty.

    Args:
        config_path: Optional path to settings.yaml
        extra_overrides: Additional configuration overrides
        verbose: Print environment summary

    Returns:
        KaggleConfig with all settings ready for pipeline use.
    """
    # Detect environment
    env = detect_kaggle_environment()

    if verbose:
        from .environment import print_environment_summary
        print_environment_summary(env)

    # Create config
    config = create_kaggle_config(
        config_path=config_path,
        environment=env,
        extra_overrides=extra_overrides,
    )

    # Apply environment variables for downstream libraries
    _apply_env_vars(config.settings, env)

    # Ensure data directories exist
    _ensure_directories(config.settings)

    if verbose:
        print(f"\nConfiguration ready. Data dir: {config.settings['paths']['data_dir']}")
        print(f"Device: {get_optimal_device(env).value}")
        print(f"Parallel renders: {config.settings['video']['parallel_renders']}")

    return config


def _apply_env_vars(config: Dict[str, Any], env: KaggleEnvironment) -> None:
    """Apply configuration as environment variables for libraries that read them."""
    # Data directory
    if "paths" in config and "data_dir" in config["paths"]:
        os.environ["CLIPS_STUDIO_DATA_DIR"] = config["paths"]["data_dir"]

    # LLM API keys from secrets
    for secret_name in ["OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "XAI_API_KEY"]:
        if secret_name in env.secrets:
            os.environ[secret_name] = env.secrets[secret_name]

    # YouTube cookies
    if "youtube" in config and "cookies" in config["youtube"]:
        os.environ["YTDLP_COOKIES"] = config["youtube"]["cookies"]

    # CUDA memory management
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _ensure_directories(config: Dict[str, Any]) -> None:
    """Ensure all required directories exist."""
    dirs_to_create = []

    if "paths" in config and "data_dir" in config["paths"]:
        data_dir = Path(config["paths"]["data_dir"])
        dirs_to_create.extend([
            data_dir,
            data_dir / "downloads",
            data_dir / "transcripts",
            data_dir / "clips",
            data_dir / "branding",
            data_dir / "voices",
            data_dir / "state",
        ])

    # Model cache directory
    dirs_to_create.append(Path("/app/models"))

    for d in dirs_to_create:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass  # Ignore permission errors


def get_pipeline_config(config: KaggleConfig) -> Dict[str, Any]:
    """
    Extract the configuration dict for passing to core.pipeline.process_video().

    This is the dict that gets passed as the `config` parameter throughout
    the Clips Kitty pipeline.
    """
    return config.settings


# Convenience function for notebook usage
def quick_setup(
    data_dir: str = "/data",
    use_cloud_llm: bool = True,
    force_cpu: bool = False,
    verbose: bool = True,
) -> KaggleConfig:
    """
    Quick setup for Kaggle notebooks.

    Args:
        data_dir: Persistent data directory
        use_cloud_llm: Use cloud LLM (requires API key in Kaggle Secrets)
        force_cpu: Force CPU mode even if GPU available
        verbose: Print summary

    Returns:
        Ready-to-use KaggleConfig
    """
    extra_overrides = {
        "paths": {"data_dir": data_dir},
    }

    if force_cpu:
        extra_overrides["transcription"] = {"device": "cpu", "compute_type": "int8"}
        extra_overrides["tracking"] = {"device": "cpu"}
        extra_overrides["video"] = {"encoder": "libx264"}

    return setup_kaggle_environment(
        extra_overrides=extra_overrides,
        verbose=verbose,
    )
# TRANSCRIPTION_AUDIT.md — Transcription System (Clips Kitty → Kaggle)

**Repository:** `E:\Clips kitty\clips-kitty-kaggle\`
**Date:** 2026-11-26

---

## 1. Models Used

### Local (faster-whisper / CTranslate2)

| Model | Size | Accuracy | Speed | Bundled | Notes |
|-------|------|----------|-------|---------|-------|
| **large-v3-turbo** | ~1.5 GB | Highest | Fast (4-layer decoder) | ✅ Yes | Default on GPU (`auto` mode) |
| **small** | ~466 MB | Good | Faster | ✅ Yes | Default on CPU (`auto` mode); fallback if GPU fails |
| medium | ~1.5 GB | Higher | 3-5x slower than small | ❌ No | Not bundled; downloads on demand |
| large-v3 | ~3 GB | Highest | Slowest | ❌ No | Not bundled |

**Config (settings.yaml:48-52):**
```yaml
whisper:
  model: auto        # auto = large-v3-turbo on GPU, small on CPU
  device: auto       # cuda if available, else cpu
```

**Model selection logic (`transcription/transcriber.py:_load_model`):**
- `device=auto` + GPU available + cuBLAS 12 → tries `large-v3-turbo` → falls back to `small`
- `device=auto` + no GPU/cuBLAS → `small` (CPU, `compute_type="auto"` = int8)
- `device=cuda` (explicit) → requires cuBLAS 12, no silent CPU fallback
- `model_size` forced (e.g., `medium`) → uses that size on selected device

### Cloud Providers (Optional, opt-in via `transcription.backend`)

| Provider | API Format | Word Timings | Models | Notes |
|----------|------------|--------------|--------|-------|
| **OpenRouter** | `/audio/transcriptions` (Whisper-compatible) | ✅ | `whisper-1`, others | Any OpenRouter STT model |
| **OpenAI** | `/audio/transcriptions` | ✅ | `whisper-1` | Direct OpenAI |
| **xAI** | `/stt` | ✅ | `grok-2-whisper` (example) | xAI Grok STT |

**Config:**
```yaml
transcription:
  backend: local        # local | openrouter | openai | xai
  model: ""             # provider-specific model name
```

**Requirements for cloud STT:**
- Must return `verbose_json` with `timestamp_granularities: ["segment", "word"]`
- Audio chunked at `spec.stt["chunk_seconds"]` (default 300s = 5 min)
- Voice activity detection (Silero) runs locally before sending — silent chunks dropped

---

## 2. GPU Usage

### Local Whisper (faster-whisper)

| Aspect | Detail |
|--------|--------|
| **Engine** | CTranslate2 (CUDA 12) |
| **Compute Type** | `float16` on GPU, `int8` (auto) on CPU |
| **Decoding** | Greedy (`beam_size=1`) — ~2.4x faster than beam 5 |
| **VAD** | Silero (built into faster-whisper) — `vad_filter=True` |
| **Word Timestamps** | Enabled (`word_timestamps=True`) |
| **Condition on Previous Text** | **Disabled** — prevents repetition loops on music/noise |
| **Hotwords** | Supported (sports player names) — passed per-window |

### cuBLAS 12 Dependency (Critical)

**Windows:** Requires `cublas64_12.dll` + `cublasLt64_12.dll`
- Bundled via `nvidia-cublas-cu12` wheel in `requirements.txt`
- Loaded at **first inference** (not model load) via `ctypes.WinDLL`
- Search paths: app bundle → `nvidia.cublas` wheel → `CUDA_PATH` → `PATH` → PyTorch lib

**Linux (Kaggle):** Requires `libcublas.so.12` + `libcublasLt.so.12`
- Install: `pip install nvidia-cublas-cu12` (Linux wheel exists on PyPI)
- Or: `apt-get install cuda-cublas-12` (if CUDA toolkit installed)
- CTranslate2 finds via `ldconfig` / `LD_LIBRARY_PATH`

### GPU Validation (`core/gpu.py:cuda_usable()`)

```python
# Checks:
1. torch.cuda.is_available()
2. Device capability (e.g., sm_86 for T4, sm_80 for A100)
3. PyTorch build arch list (forward-compatible within major version only)
   - sm_8x runs on sm_8y (e.g., sm_86 kernels run on T4 sm_75? No — T4 is sm_75)
   - Wait: T4 = sm_75, A100 = sm_80, P100 = sm_60
   - PyTorch CUDA 12.1 build typically supports sm_60 through sm_90
```

**Kaggle GPUs:**
| GPU | Compute Capability | PyTorch CUDA 12 Support |
|-----|-------------------|------------------------|
| T4 | sm_75 | ✅ Supported |
| P100 | sm_60 | ✅ Supported (older) |
| A100 | sm_80 | ✅ Supported |
| V100 | sm_70 | ✅ Supported |

**Failure modes:**
- `NO_GPU` — no CUDA device → CPU fallback
- `older than` — GPU too old for build → CPU fallback (permanent)
- `newer than` — GPU newer than build → CPU fallback (fix in release)

---

## 3. CPU Fallback

### Automatic Fallback Paths

| Trigger | Behavior |
|---------|----------|
| `device=auto` + no cuBLAS 12 | Prints warning, loads `small` on CPU (`compute_type="auto"`) |
| `device=auto` + GPU load fails | Prints error, retries with `_load_model(model_size, "cpu")` |
| `device=cuda` (explicit) + cuBLAS missing | **Raises `RuntimeError`** — no silent fallback |
| `device=cuda` + GPU inference fails | If library error (cuBLAS/cuDNN/CUDA), retries on CPU |

### CPU Performance

| Model | Compute Type | Relative Speed (vs GPU float16) | VRAM | RAM |
|-------|--------------|--------------------------------|------|-----|
| large-v3-turbo | int8 | ~5-10x slower | 0 | ~2 GB |
| small | int8 | ~3-5x slower | 0 | ~1 GB |
| medium | int8 | ~10-15x slower | 0 | ~3 GB |

**Thread control (`core/pipeline.py:_share_the_cpu`):**
- Called once per process before rendering
- Sets `cv2.setNumThreads(per_worker)` and `torch.set_num_threads(per_worker)`
- `per_worker = max(1, (cores - 2) // parallel_renders)`
- Holds back 2 cores for system responsiveness

---

## 4. Memory Requirements

### GPU VRAM (Local Whisper)

| Model | Compute Type | VRAM (model) | VRAM (activation) | Total Estimate |
|-------|--------------|--------------|-------------------|----------------|
| large-v3-turbo | float16 | ~1.5 GB | ~2-3 GB | **4-5 GB** |
| large-v3-turbo | int8 | ~800 MB | ~1-2 GB | 2-3 GB |
| small | float16 | ~500 MB | ~1 GB | 1.5 GB |
| small | int8 | ~250 MB | ~500 MB | 750 MB |

**Kaggle T4 (16 GB VRAM):** ✅ Comfortable for large-v3-turbo + YOLO + TalkNet
**Kaggle P100 (16 GB VRAM):** ✅ Comfortable
**Kaggle A100 (40 GB VRAM):** ✅ Plenty

### CPU RAM (Local Whisper)

| Model | Compute Type | RAM (model) | RAM (working) | Total |
|-------|--------------|-------------|---------------|-------|
| large-v3-turbo | int8 | ~1.5 GB | ~1 GB | **2.5 GB** |
| small | int8 | ~500 MB | ~300 MB | 800 MB |

### Cloud Transcription Memory

| Component | Memory |
|-----------|--------|
| Audio extraction (FFmpeg) | ~50 MB per chunk (300s = 5 min → ~2.4 MB MP3) |
| Chunk processing | Sequential — one chunk at a time |
| Silero VAD | ~50 MB (model loaded once) |
| **Total** | **< 200 MB** |

---

## 5. Transcription Pipeline Flow

```
transcribe(video_path, video_id, transcript_dir, config)
    │
    ├─► Check cache: transcript_dir/{video_id}.json
    │       └─► Exists → load, collapse repetition loops, return
    │
    ├─► Online backend? (transcription.backend != "local")
    │       └─► transcription/cloud.py:transcribe()
    │             ├─► Chunk video (300s parts, 2s overlap)
    │             ├─► Extract audio per chunk (FFmpeg → 16kHz mono MP3)
    │             ├─► Silero VAD (local) → drop silent chunks
    │             ├─► Send to provider (OpenRouter/OpenAI/xAI)
    │             ├─► Reconstruct segments from word timings
    │             └─► Cache JSON + return
    │
    └─► Local Whisper
          ├─► _load_model(model_size, device)
          │       ├─► Resolve model path (bundled via core.binaries.whisper_model)
          │       ├─► GPU: load cuBLAS 12, WhisperModel(device="cuda", compute_type="float16")
          │       └─► CPU: WhisperModel(device="cpu", compute_type="auto")
          │
          ├─► _run(model, video_path, language, hotwords)
          │       ├─► model.transcribe(
          │       │       language=language,
          │       │       vad_filter=True,
          │       │       beam_size=1,
          │       │       condition_on_previous_text=False,
          │       │       word_timestamps=True,
          │       │       hotwords=hotwords
          │       │   )
          │       ├─► Iterate segments (generator)
          │       │       ├─► cancel.check_active() per segment
          │       │       ├─► Build Segment(start, end, text, words[])
          │       │       └─► progress.emit() every 2% or 5s
          │       └─► Return (segments, info)
          │
          ├─► _collapse_repetition_loops(segments)
          │       └─► Detects: duration>25s, words>40, unique_ratio<0.15
          │
          └─► Cache JSON to transcript_dir/{video_id}.json
```

---

## 6. Kaggle-Specific Configuration

### Recommended `settings.yaml` for Kaggle

```yaml
whisper:
  model: "large-v3-turbo"   # or "small" if VRAM constrained
  device: "cuda"            # force GPU (fails fast if unavailable)

transcription:
  backend: "local"          # or "openrouter" to avoid local GPU deps
  model: ""                 # provider-specific if cloud

# Reduce parallelism to save VRAM for Whisper
video:
  parallel_renders: 1       # T4: 1, A100: 2
```

### Environment Variables

```bash
# Data directory (persistent volume)
export CLIPS_STUDIO_DATA_DIR=/data

# Force CPU if no cuBLAS / want to save VRAM
# export WHISPER_DEVICE=cpu   # not directly used; set in settings.yaml

# Cloud STT (if using)
export OPENROUTER_API_KEY=sk-...
```

### Dockerfile Additions

```dockerfile
# CUDA 12 + cuBLAS 12 for faster-whisper GPU
RUN pip install --no-cache-dir nvidia-cublas-cu12

# Or use CPU-only (smaller image, slower)
# ENV WHISPER_DEVICE=cpu  # handled via settings.yaml
```

### Model Pre-caching

```python
# scripts/precache_transcription.py
from core.binaries import whisper_model
from faster_whisper import WhisperModel

# Pre-load to trigger download if not bundled
for size in ["large-v3-turbo", "small"]:
    path = whisper_model(size)
    if path == size:  # not bundled
        print(f"Downloading {size}...")
        WhisperModel(size, device="cpu", compute_type="int8")
        print(f"  Cached at ~/.cache/huggingface/hub/")
```

---

## 7. Known Issues & Fixes for Kaggle

| Issue | Cause | Fix |
|-------|-------|-----|
| `ImportError: libcublas.so.12` | cuBLAS 12 not installed | `pip install nvidia-cublas-cu12` |
| `RuntimeError: GPU transcription needs NVIDIA's cuBLAS 12` | Windows-only error message; on Linux it's the same root cause | Install cuBLAS or use CPU |
| `CUDA error: no kernel image` | PyTorch build arch mismatch | `core.gpu.cuda_usable()` catches this → CPU fallback |
| Silent multi-GB download on first run | Model not bundled | Pre-cache in Docker build or bundle in image |
| OOM on T4 with parallel_renders > 1 | Whisper + YOLO + TalkNet + NVENC exceed 16 GB | Set `video.parallel_renders=1` |
| Transcription hangs on long video | No progress emit / cancel check | `_run()` emits progress every 2% and checks cancel per segment |

---

## 8. Files to Modify for Kaggle

| File | Change |
|------|--------|
| `config/settings.yaml` | Set `whisper.device: "cuda"`, `video.parallel_renders: 1` |
| `Dockerfile` | Add `pip install nvidia-cublas-cu12` |
| `scripts/precache_models.py` | **New** — pre-download Whisper models |
| `transcription/transcriber.py` | No changes needed — auto-detects cuBLAS |
| `core/gpu.py` | No changes — validates at runtime |

---

## 9. Summary

| Aspect | Kaggle Status |
|--------|---------------|
| **Local Whisper GPU** | ✅ Works with `nvidia-cublas-cu12` |
| **Local Whisper CPU** | ✅ Works out of the box (`small` model, int8) |
| **Cloud STT** | ✅ Works (OpenRouter/OpenAI/xAI) — no GPU needed |
| **Bundled models** | ⚠️ Not in repo — pre-cache in Docker build |
| **cuBLAS 12** | ⚠️ Must install `nvidia-cublas-cu12` (Linux wheel exists) |
| **VRAM on T4** | ✅ large-v3-turbo fits with `parallel_renders=1` |
| **Cancellation** | ✅ Per-segment cancel check |
| **Progress** | ✅ Throttled percent updates |
| **Repetition loops** | ✅ Auto-collapsed (music/noise guard) |
| **Word timestamps** | ✅ Always enabled (captions depend on them) |
| **Hotwords** | ✅ Supported (sports player names) |
| **Language detection** | ✅ Auto or forced via `content_language` config |

**Recommendation:** Use **local Whisper on GPU** with `large-v3-turbo` for best quality/speed. Pre-cache models in Docker image. Set `parallel_renders=1` on T4. Cloud STT is a viable alternative if GPU setup is problematic.
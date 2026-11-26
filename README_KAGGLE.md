# Clips Kitty on Kaggle

Run the full Clips Kitty viral clip extraction pipeline in a Kaggle GPU notebook. Zero modifications to the original processing logic — only a thin adapter layer for Kaggle compatibility.

---

## Quick Start

1. **Open** `Clips_Kitty_Kaggle.ipynb` in a Kaggle GPU notebook (T4, P100, V100, A100, or L4)
2. **Run Cells 1–3** — installs dependencies, detects GPU, configures environment
3. **Run Cell 4** — paste a YouTube/Twitch/Kick URL or upload a local video
4. **Run Cell 5** — configure content mode, clip count, duration limits
5. **Run Cell 6** — start processing with live progress
6. **Cells 7–10** — preview clips, download individually or as ZIP

---

## Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | T4 (16 GB) | A100 40/80 GB, L4 24 GB |
| VRAM | 8 GB | 16+ GB |
| Disk | 10 GB | 20+ GB (`/data` persistent volume) |
| RAM | 16 GB | 30+ GB |
| Internet | Yes (model downloads) | Yes |
| Secrets | Optional | `OPENROUTER_API_KEY` for cloud LLM |

**First run** downloads models (~2–5 min): Whisper `large-v3-turbo`, YOLOv8n-pose, TalkNet. Subsequent runs use cached models in `/app/models/`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    KAGGLE NOTEBOOK                          │
├─────────────────────────────────────────────────────────────┤
│  Clips_Kitty_Kaggle.ipynb                                   │
│  ├── kaggle_adapter.environment   → GPU/VRAM/CUDA detection │
│  ├── kaggle_adapter.config        → Settings + path merge   │
│  ├── kaggle_adapter.pipeline_adapter → Wraps original pipe  │
│  └── kaggle_adapter.progress_monitor → Rich progress UI    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              ORIGINAL CLIPS KITTY PIPELINE (UNMODIFIED)     │
├─────────────────────────────────────────────────────────────┤
│  core.pipeline.process_video()                              │
│  ├── sources.dispatch.download()        → yt-dlp            │
│  ├── transcription.transcriber.transcribe() → faster-whisper│
│  ├── analysis.audio/visual_features    → FFmpeg + numpy     │
│  ├── analysis.fusion.find_clips()      → LLM + multimodal   │
│  ├── video.cutter.cut_clip()           → FFmpeg (NVENC)     │
│  ├── video.cropper.render_vertical()   → YOLOv8 + TalkNet   │
│  ├── video.captions.build_captions()   → ASS subtitles      │
│  └── analysis.metadata.generate_metadata_batch() → LLM      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    KAGGLE FILESYSTEM                        │
├─────────────────────────────────────────────────────────────┤
│  /data/              → Persistent volume (survives restart) │
│  │  ├── downloads/   → Source videos                        │
│  │  ├── transcripts/ → Whisper JSON                        │
│  │  ├── clips/       → Rendered clips (organized)          │
│  │  ├── state.db     → SQLite WAL (crash-resilient)        │
│  │  ├── branding/    → Watermark assets                    │
│  │  └── voices/      → Speaker embeddings                  │
│  /kaggle/working/    → Output for download (clips/, ZIP)   │
│  /kaggle/secrets/    → API keys (OPENROUTER_API_KEY, etc.) │
│  /app/models/        → Pre-cached model weights            │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Content Modes (Cell 5)

| Mode | Crop | Use Case |
|------|------|----------|
| `standard` | track | General Shorts, speaker tracking |
| `vertical_live` | fit_blur | Already-vertical source (TikTok, Reels) |
| `sports` | track | Play-aware, scoreboard detection |
| `gaming` | split | Webcam + gameplay layout |
| `podcast` | track | Multi-speaker, shot-aware |
| `longform` | none (16:9) | Horizontal, chapter markers |

### Key Settings (modify in Cell 5)

```python
CONTENT_MODE = "standard"      # standard, vertical_live, sports, gaming, podcast, longform
CLIP_COUNT = 3                 # 1–10 clips
MIN_DURATION = 15              # seconds
MAX_DURATION = 60              # seconds (Shorts limit)
MIN_SCORE = 0.6                # fusion score threshold
```

### Advanced Config (auto-detected, override if needed)

```python
config.settings["transcription"]["model"]     # "large-v3-turbo", "small", "base"
config.settings["transcription"]["device"]    # "cuda" or "cpu"
config.settings["video"]["parallel_renders"]  # 1 (T4), 2 (A100/L4)
config.settings["video"]["encoder"]           # "auto" (NVENC) or "libx264"
config.settings["clips"]["vertical"]["crop_mode"]  # "track", "split", "fit_blur"
```

---

## GPU-Specific Guidance

| GPU | VRAM | Whisper Model | Batch Size | Parallel Renders | Encoder |
|-----|------|---------------|------------|------------------|---------|
| T4 | 16 GB | large-v3-turbo | 8 | 1 | NVENC |
| P100 | 16 GB | large-v3-turbo | 8 | 1 | NVENC |
| V100 | 16/32 GB | large-v3-turbo | 8–12 | 1–2 | NVENC |
| A100 | 40/80 GB | large-v3-turbo | 12–16 | 2 | NVENC |
| L4 | 24 GB | large-v3-turbo | 12 | 2 | NVENC |
| CPU only | — | small | 1 | 1 | libx264 |

**Force CPU mode** (Cell 3):
```python
config = quick_setup(data_dir="/data", force_cpu=True, verbose=True)
```

---

## Secrets Setup (Cloud LLM)

For best clip scoring/reranking, add API keys to **Kaggle Secrets** (🔑 icon in left sidebar):

| Secret Name | Provider | Example Model |
|-------------|----------|---------------|
| `OPENROUTER_API_KEY` | OpenRouter | `anthropic/claude-3.5-sonnet` |
| `OPENAI_API_KEY` | OpenAI | `gpt-4o` |
| `ANTHROPIC_API_KEY` | Anthropic | `claude-3-5-sonnet-20241022` |
| `XAI_API_KEY` | xAI | `grok-beta` |
| `YOUTUBE_COOKIES` | YouTube | (cookie string for age-restricted) |

Without secrets, falls back to local Ollama (requires sidecar container — not included).

---

## Pipeline Stages & Progress

| Stage | Weight | Description |
|-------|--------|-------------|
| `download` | 15% | yt-dlp fetches video (H.264 preferred) |
| `transcribe` | 35% | faster-whisper GPU (cuBLAS 12 required) |
| `analyze` | 20% | Audio/visual signals + LLM moment scoring |
| `render` | 30% | Cut → track/crop → captions → encode |

Progress shown via tqdm + IPython display. State saved to `/data/state.db` after each stage — **interrupted runs resume automatically**.

---

## Output

### Clip Files
```
/kaggle/working/clips/<video_id>/
├── clip_00015-00045.mp4
├── clip_00120-00155.mp4
└── clip_00300-00345.mp4
```

Each clip: 1080×1920 (9:16), H.264/AAC, 5 Mbps, burned captions, end card (if enabled).

### Metadata (in notebook `result` object)
```python
result.clips[0].candidate.start      # 15.0
result.clips[0].candidate.end        # 45.0
result.clips[0].candidate.score      # 87
result.clips[0].candidate.hook       # "You won't believe what happened next"
result.clips[0].candidate.reason     # "Emotional peak with revelation"
result.clips[0].path                 # Path to .mp4
```

### Download Options
- **Individual**: File browser → `/kaggle/working/clips/<video_id>/` → download
- **ZIP**: Cell 10 creates `<video_id>.zip` in `/kaggle/working/clips/`

---

## Troubleshooting

### CUDA OOM (Out of Memory)
```python
# Cell 3: force CPU
config = quick_setup(data_dir="/data", force_cpu=True)

# Or reduce parallelism
config.settings["video"]["parallel_renders"] = 1

# Or smaller Whisper model
config.settings["transcription"]["model"] = "small"
```

### yt-dlp Download Fails
- Add `YOUTUBE_COOKIES` to Kaggle Secrets (export from browser)
- Or place `cookies.txt` in `/data/youtube_cookies.txt`
- Update: `!pip install -U yt-dlp`

### No Clips Generated
- Lower `MIN_SCORE` to 0.4–0.5
- Increase `MAX_DURATION` to 90
- Verify video has speech (Whisper needs audio)

### Slow Processing
- Ensure GPU detected (Cell 2 output shows `CUDA Available: True`)
- Add `OPENROUTER_API_KEY` for cloud LLM (local Gemma is slow on CPU)
- Models auto-cache on first run

### Interrupted Session
- State in `/data/state.db` — re-run cells 5, 6 to resume
- Already-rendered clips skipped automatically

---

## File Structure

```
clips-kitty-kaggle/
├── Clips_Kitty_Kaggle.ipynb      # Main notebook (23 cells)
├── requirements-kaggle.txt        # Pinned dependencies
├── kaggle_adapter/
│   ├── __init__.py
│   ├── environment.py            # GPU/CUDA/environment detection
│   ├── config.py                 # Config merge + path resolution
│   ├── pipeline_adapter.py       # Wraps core.pipeline.process_video
│   └── progress_monitor.py       # tqdm/IPython progress UI
├── core/                         # Original Clips Kitty (UNMODIFIED)
│   ├── pipeline.py               # Main orchestration
│   ├── state.py                  # SQLite state machine
│   └── ...
├── analysis/                     # Signal extraction, fusion, highlights
├── video/                        # Cutter, cropper, captions, encoding
├── transcription/                # faster-whisper wrapper
├── llm/                          # LLM backends (Ollama, OpenRouter, etc.)
└── sources/                      # YouTube, Twitch, Kick downloaders
```

---

## Verified Original Functions (Adapter Uses These)

| Function | Module | Purpose |
|----------|--------|---------|
| `process_video` | `core.pipeline` | Full 14-stage pipeline |
| `download` | `sources.dispatch` | yt-dlp ingestion |
| `transcribe` | `transcription.transcriber` | faster-whisper GPU/CPU |
| `extract_audio_features` | `analysis.audio_features` | Spike/burst/noise per second |
| `extract_visual_features` | `analysis.visual_features` | Motion/scene cuts/flash per second |
| `find_clips` | `analysis.fusion` | Candidate gen + fusion + rerank |
| `cut_clip` | `video.cutter` | Frame-exact FFmpeg extraction |
| `render_vertical` | `video.cropper` | Track/split/fit_blur 9:16 |
| `build_captions` | `video.captions` | ASS word-synced subtitles |
| `generate_metadata_batch` | `analysis.metadata` | Batched LLM titles/tags |

**No original pipeline code was modified.** The adapter only configures, wraps, and monitors.

---

## License & Credits

- **Original Clips Kitty**: [Clips-Kitty/Clips-Kitty](https://github.com/Clips-Kitty/Clips-Kitty)
- **Kaggle Adapter**: This repository — MIT License
- **Models**: Whisper (OpenAI), YOLOv8 (Ultralytics), TalkNet (MIT), Gemma (Google)

---

## Support

- **Issues**: GitHub Issues on this repository
- **Original Project**: Clips Kitty Discord / GitHub Discussions
- **Kaggle-Specific**: Check notebook Cell 25 for common fixes
# Repository Map — Clips Kitty

**Root:** `E:\Clips kitty\clips-kitty-kaggle\`

---

## Main Entry Points

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point — commands: `process`, `run`, `status`, `serve`, `models`, `channels`, `render-worker`, `auth`, `upload`, `outro-backfill`, `mcp` |
| `_clipskitty_script_host.py` | Internal host for pipeline runs launched from the desktop app (Electron spawns `api.exe src/main.py ...` with `CLIPSKITTY_SCRIPT_HOST=1`) |

---

## Top-Level Directories & Their Purposes

| Directory | Purpose |
|-----------|---------|
| `analysis/` | Clip engine: multimodal signal extraction (audio/visual), candidate generation, fusion scoring, rerank, metadata (titles/hashtags), gaming/sports profiles, intent/direction, chat moments, game audio/vision/text |
| `campaigns/` | Discord-driven campaigns: eligibility, queue, links, webhook, OAuth, state machine |
| `config/` | Configuration files: `settings.yaml` (user-editable), `gaming.yaml`, `sports.yaml`, prompt templates (`prompts/`) |
| `connected_accounts/` | OAuth + EventSub for Twitch, Discord; credential storage |
| `core/` | Pipeline orchestration, SQLite state, models, progress/cancel, GPU detection, binaries resolver, queue/scheduler, preflight, paths |
| `creator/` | Creator intelligence: identity resolution, knowledge extraction, retrieval, preference learning |
| `gaming/` | Gaming/Reaction mode: streamer detection, webcam/game layout, framing, composition (11 layouts) |
| `llm/` | Swappable model layer: `LLMBackend` interface, Ollama backend, cloud providers (OpenRouter, OpenAI, Anthropic, Gemini, xAI, etc.), model manager |
| `longform/` | 16:9 output profiles: short_clips, clips_140, highlights, edited_stream; assembly, downtime detection |
| `multilingual/` | Translated publishing: translation, subtitles, dubbing (Piper TTS), voice catalogue, glossary protection |
| `packaging/` | Windows installer/build assets (MSIX, AppX) |
| `plugins/` | Marketplace plugin system: registry, steps (Understand/Rate/Suggest edits), gaming detector, runner |
| `publish/` | Publishing integrations: YouTube Shorts API, WoopSocial, Upload-Post; scheduling, quota |
| `remote_render/` | Second-PC render worker: TLS gateway, worker protocol, dispatcher |
| `scripts/` | Build/maintenance scripts: fetch binaries (FFmpeg, Ollama, Whisper, YOLO, PANNs, QuickJS), benchmarks, installer build |
| `server/` | Local FastAPI service: API routes, job queue, WebSocket events, automation, MCP server, agent, webhooks, publisher |
| `sources/` | Platform ingestion: YouTube (RSS + yt-dlp), Twitch (VOD + chat replay), Kick, vertical lives, local files; shared yt-dlp helpers |
| `sports/` | Sports mode (opt-in): per-sport packages (soccer), score box OCR, ball detection, moment windows, story reels |
| `stream_workflows/` | Streamer-tool hand-offs: OBS, Streamlabs, etc. |
| `transcription/` | faster-whisper (local, CUDA/CPU), cloud transcription (OpenRouter, etc.), word timestamps, repetition-loop collapse |
| `video/` | Rendering pipeline: YOLOv8 pose tracking, TalkNet active-speaker detection, framing, shot detection, cropper (9:16 modes), captions (ASS), cutter (FFmpeg), filters, hardware encoder selection |
| `video_editor/` | Non-destructive timeline editor: cuts, audio (volume/fades/mutes/music), caption edits, hook overlay, watermark/branding, export |
| `third_party/` | Vendored dependencies: TalkNet-ASD (do not edit) |
| `models/` | Model weights (gitignored at runtime): TalkNet `pretrain_TalkSet.model`, PANNs game sounds, voice models (WeSpeaker + pyannote) |
| `ui/` | Desktop app: Electron main + React/TypeScript/Tailwind renderer |
| `feedback-relay/` | Cloudflare Worker → GitHub Issues (in-app feedback hub) |
| `web/` | Static site / docs |
| `whop-app/` | Whop marketplace integration |
| `examples/` | Example pipelines / usage |
| `tests/` | Test suite |
| `docs/` | Documentation (architecture, API, developers, platform research, hackathon) |

---

## Key Configuration Files

| File | Purpose |
|------|---------|
| `config/settings.yaml` | User settings: model, channel, transcription, clips (scoring, duration, vertical, captions, outro), analysis thresholds, video encoder, tracking, upload, paths |
| `config/gaming.yaml` | Game genres, highlight definitions, callouts, chat reaction classes, sound groups |
| `config/sports.yaml` | Sport definitions: moments, windows, commentary words, score box OCR zones |
| `config/prompts/*.txt` | All LLM prompts as editable files: `score_clips`, `score_windows`, `rerank`, `edit_clip`, `metadata`, `extract_knowledge`, `look_at_game`, `translate_captions`, `translate_metadata` |

---

## Runtime Data (gitignored)

| Path | Purpose |
|------|---------|
| `data/downloads/` | Cached source videos (per video_id) |
| `data/transcripts/` | Whisper JSON caches (per video_id) |
| `data/clips/` | Rendered clips organized by channel/video |
| `data/state.db` | SQLite database: videos, clips, channels, creators, branding, rejections, outcomes |
| `data/branding/` | Watermark assets, branding profiles |
| `data/voices/` | Piper TTS voices (downloaded on demand) |
| `data/youtube_token.json` | YouTube OAuth token |

---

## External Binaries (bundled in installed app, resolved via `core/binaries.py`)

| Binary | Source | Purpose |
|--------|--------|---------|
| `ffmpeg` / `ffprobe` | FFmpeg | All decode, encode, probe, concat |
| `ollama` | Ollama | Local LLM server |
| `qjs` (QuickJS) | QuickJS-ng | YouTube player JS execution for cookies fallback |
| Whisper models | faster-whisper / CTranslate2 | `large-v3-turbo`, `small` (bundled) |
| YOLO weights | ultralytics | `yolov8n-pose.pt` (tracking), `yolov8n.pt` (ball detection) |
| TalkNet | vendored in `third_party/talknet/` | Active-speaker detection |
| PANNs | AudioSet tagger | Game sounds, crowd, whistle |
| Voice models | onnxruntime | pyannote segmentation + WeSpeaker (second speaker caption colour) |

---

## Pipeline Stage Map (from `core/pipeline.py`)

| Stage | Module | Input | Output |
|-------|--------|-------|--------|
| 1. Download | `sources/` | URL | `data/downloads/{id}.mp4` |
| 2. Transcribe | `transcription/transcriber.py` | video | segments + word timestamps (cached) |
| 3. Analyze | `analysis/fusion.py` + LLM | segments + signals | `ClipCandidate[]` with subscores |
| 4. Frame/Track | `video/tracker.py` (standard), `gaming/run.py`, `sports/`, `video/podcast.py` | clip window | crop path / layout |
| 5. Render | `video/cropper.py` + `video/cutter.py` + FFmpeg | crop path + captions | `data/clips/.../clip_N.mp4` |
| 6. Metadata | `analysis/metadata.py` + LLM | candidates + segments | title, description, hashtags |

---

## Concurrency Design

- **Signal extraction** (audio + visual) runs in background thread during transcription
- **Hype fetch** (chat replay / heatmap) runs in background thread during transcription  
- **Game sounds** (PANNs) runs in background thread during transcription (gaming mode)
- **Sports reading** (OCR + sound) runs in background during transcription
- **Creator learning** runs during render stage (Ollama otherwise idle)
- **Render workers** (`video.parallel_renders`, default 3) — one's GPU tracking overlaps another's NVENC encode
- **Download prefetch** fetches next queued video during current processing

---

## GPU/CPU Split

| Component | Device | Notes |
|-----------|--------|-------|
| Whisper (faster-whisper) | CUDA (float16) or CPU | Requires cuBLAS 12 on Windows; auto-fallback |
| YOLOv8 (tracking, ball) | CUDA or CPU | `core/gpu.py` validates kernel compatibility |
| TalkNet | CUDA or CPU | Vendored PyTorch model |
| PANNs | CPU | onnxruntime |
| Voice models | CPU | onnxruntime, one core |
| RapidOCR | CPU | onnxruntime |
| FFmpeg encode | NVENC / AMF / QSV / VideoToolbox / CPU | Hardware encoder probed at startup |
| FFmpeg decode | NVDEC (auto) | `-hwaccel auto` before `-i` |
| LLM (Ollama) | GPU/CPU per Ollama | Independent process |

---

## Platform Compatibility Notes

| Aspect | Windows | Linux (Kaggle/Docker) | macOS |
|--------|---------|----------------------|-------|
| FFmpeg | Bundled | System package (apt) | Homebrew / bundled |
| Ollama | Bundled | System service / container | Bundled |
| CUDA | Native | Container GPU / CPU-only | Metal (VideoToolbox) |
| Hardware encode | NVENC/AMF/QSV | NVENC (if GPU) / CPU | VideoToolbox |
| DPAPI secrets | `core/secrets.py` | keyring (libsecret) | keyring |
| Desktop app | Electron (shipped) | Not in container | Electron (dev) |
| Paths | `core/paths.py` resolves per-platform | `/data` via env | `~/Library/Application Support` |

---

## Features Requiring Desktop App (Electron)

- Video editor UI (`video_editor/` + `ui/src/renderer/`)
- Timeline editing, caption corrections, speaker edits
- Layout buttons (crop mode, split, letterbox, face-safe)
- Branding/watermark visual editor
- AI chat edit assistant
- Preview player with seek/scrub
- Settings UI (AI backends, upload, branding)
- Channel management UI
- Clip review/selection grid
- Story reel builder (Sports)
- Longform profile selector

---

## Features That Work Headless (CLI / API)

- Full pipeline: `python main.py process <url>`
- Daemon: `python main.py run` (RSS polling + scheduler)
- Local API: `python main.py serve` (FastAPI on 127.0.0.1:8765)
- MCP server: `python main.py mcp` (stdio)
- Remote render worker: `python main.py render-worker`
- YouTube auth: `python main.py auth`
- Upload: `python main.py upload`
- Channel management: `python main.py channels add/list/remove`
- Model management: `python main.py models use <tag>`
- Status: `python main.py status`

---

## Dependencies Likely to Need Attention on Kaggle

| Dependency | Issue | Mitigation |
|------------|-------|------------|
| `nvidia-cublas-cu12` | Windows-only wheel; Kaggle uses Linux | Install `nvidia-cublas-cu12` via pip (Linux wheel exists) or use CPU Whisper |
| `opencv-python<5` | Needs `libgl1`, `libglib2.0-0` | `apt-get install -y libgl1 libglib2.0-0` |
| `faster-whisper` | Needs CUDA 12 + cuBLAS 12 for GPU | Kaggle GPUs have CUDA 12; install `nvidia-cublas-cu12` |
| `ultralytics` | Downloads YOLO weights on first use | Pre-download `yolov8n-pose.pt`, `yolov8n.pt` to `models/` |
| `rapidocr-onnxruntime` | Downloads OCR models on first use | Pre-download or allow network on first run |
| `piper-tts` | Downloads voices on demand | Pre-download needed voices to `data/voices/` |
| `yt-dlp` | Frequent updates needed for site changes | Pin to known-working version or allow auto-update |
| FFmpeg | Bundled on Windows; system on Linux | `apt-get install ffmpeg` in container |
| Ollama | Separate service | Run Ollama in sidecar container or use cloud LLM |

---

## Environment Variables for Container/Headless Use

| Variable | Purpose |
|----------|---------|
| `CLIPS_STUDIO_DATA_DIR` | Override data directory (default: `./data`) |
| `CLIPS_STUDIO_OLLAMA_HOST` | Point to Ollama service (default: `http://localhost:11434`) |
| `CLIPS_STUDIO_FFMPEG` / `CLIPS_STUDIO_FFPROBE` | Override bundled binary paths |
| `CLIPS_STUDIO_OLLAMA` | Override Ollama binary path |
| `YTDLP_COOKIES` / `youtube_cookies` in config | YouTube cookies for age-gated/bot-blocked content |
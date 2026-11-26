# EXECUTION_FLOW.md — Runtime Execution Flow (Clips Kitty → Kaggle)

**Repository:** `E:\Clips kitty\clips-kitty-kaggle\`
**Date:** 2026-11-26

---

## 1. Runtime Entry Points

### 1.1 Primary CLI Entry Point: `main.py`

The main entry point for all headless execution is `main.py`, which exposes 10 subcommands via `argparse`:

| Command | Function | Purpose |
|---------|----------|---------|
| `process` | `process_cmd()` | **Primary pipeline** — download → transcribe → analyze → render for a single URL |
| `run` | `run_cmd()` | Background daemon — processes queued jobs from SQLite |
| `status` | `status_cmd()` | Shows queue status and system health |
| `serve` | `serve_cmd()` | Starts FastAPI server (`server/api.py`) for web UI / API |
| `auth` | `auth_cmd()` | YouTube OAuth flow for upload tokens |
| `upload` | `upload_cmd()` | Manual upload of rendered clips |
| `outro-backfill` | `outro_backfill_cmd()` | Adds outro card to existing clips missing it |
| `channels` | `channels_cmd()` | Manage watched channels (add/list/remove) |
| `models` | `models_cmd()` | List/manage Whisper/LLM models |
| `render-worker` | `render_worker_cmd()` | Standalone render worker for distributed rendering |
| `mcp` | `mcp_cmd()` | Model Context Protocol server (experimental) |

**Execution flow for `process`:**
```
main.py:process_cmd(url, args)
    → load_config() → merged settings
    → StateDB(data_dir) → SQLite connection
    → core.pipeline.process_video(url, config, db, force, ...)
    → Full 14-stage pipeline execution
```

### 1.2 Frozen Application Entry Point: `_clipskitty_script_host.py`

When packaged via PyInstaller (Windows desktop app), this is the **true runtime entry point** for the frozen executable (`api.exe`):

```python
# _clipskitty_script_host.py:main()
# Called by PyInstaller bootloader
```

**Modes of operation:**
| Mode | Trigger | Behavior |
|------|---------|----------|
| `script` | `api.exe script <file>` | Executes a Python script file in isolated namespace |
| `module` | `api.exe -m <module>` | Runs module with `runpy` (used by Electron) |
| `code` | `api.exe -c "code"` | Executes inline code string |
| `fork` | `api.exe fork <module> <func> <json>` | Spawns worker process, calls function with JSON args |

**EngineGuard isolation:** Each execution creates a fresh `EngineGuard` context that:
- Snapshots `sys.modules` before import
- Restores only allowlisted modules after execution
- Prevents pipeline state leakage between runs (critical for Electron multi-job)

**Electron integration:** The Electron UI (`desktop/`) spawns `api.exe fork pipeline.process_video ...` for each video, achieving true process isolation.

### 1.3 API Server Entry Point: `server/api.py`

Started via `python main.py serve` (or `api.exe serve` when frozen):

```python
# server/api.py:create_app()
app = FastAPI()
# Mounts: /api/*, /ws/*, static files for web UI
uvicorn.run(app, host="127.0.0.1", port=8080)
```

**API routes** (partial):
- `POST /api/process` — queue a video for processing
- `GET /api/jobs` — list queue
- `WS /ws/progress` — real-time progress updates
- `POST /api/upload` — trigger upload

### 1.4 Daemon Entry Point: `core/scheduler.py`

Background worker started by `python main.py run`:

```python
# core/scheduler.py:run_daemon()
while True:
    job = db.claim_next_job()        # Atomic claim from SQLite
    if job:
        process_job(job, db, config) # Full pipeline per job
    else:
        sleep(5)                     # Idle backoff
```

**Job types handled:**
- `type="process"` — Full video pipeline (download → render)
- `type="render"` — Re-render existing clip with new settings
- `type="translate"` — Generate multilingual captions

### 1.5 Entry Point Summary for Kaggle

| Environment | Entry Point | Command |
|-------------|-------------|---------|
| **Kaggle Notebook (CLI)** | `main.py` | `python main.py process <url>` |
| **Kaggle Notebook (API)** | `server/api.py` | `python main.py serve` + HTTP calls |
| **Kaggle Background** | `core/scheduler.py` | `python main.py run` (in background cell) |
| **Windows Desktop (Frozen)** | `_clipskitty_script_host.py` | `api.exe fork pipeline.process_video ...` |

**For Kaggle:** Use `main.py process` for single videos, `main.py serve` for batch via API, or `main.py run` for continuous daemon processing.

---

## 2. Startup Sequence

### 2.1 CLI Startup (`main.py` → `process_cmd`)

```python
def process_cmd(args):
    # 1. UTF-8 stdout fix (Windows)
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer)

    # 2. Load configuration (see Section 3)
    config = load_config(args.config)

    # 3. Resolve data directory
    data_dir = paths.resolve_data_dir(config)

    # 4. Initialize StateDB (SQLite)
    db = StateDB(data_dir / "state.db")

    # 5. Crash recovery: mark stuck videos as failed
    recovered = db.recover_stuck_videos()
    if recovered:
        print(f"  Recovered {recovered} video(s) left mid-pipeline")

    # 6. Crash recovery: re-queue interrupted jobs
    requeued = db.recover_interrupted_jobs()
    if requeued:
        print(f"  Re-queued {requeued} interrupted job(s)")

    # 7. Execute pipeline
    process_video(
        url=args.url,
        config=config,
        db=db,
        force=args.force,
        vertical=args.vertical,
        profile=args.profile,
        # ... other args
    )
```

### 2.2 Frozen App Startup (`_clipskitty_script_host.py`)

```python
def main():
    # 1. Parse frozen app arguments
    #    sys.argv[1] = mode: "script" | "module" | "code" | "fork"

    # 2. If fork mode (Electron worker):
    #    - Parse module, function, JSON payload
    #    - Create EngineGuard isolation context
    #    - Import module, call function(payload)
    #    - Return JSON result to parent process

    # 3. If script/module/code mode:
    #    - Execute in isolated namespace
    #    - EngineGuard snapshots/restores sys.modules
```

### 2.3 API Server Startup (`server/api.py` → `create_app`)

```python
def create_app():
    # 1. Load config (same as CLI)
    config = load_config()

    # 2. Initialize StateDB
    db = StateDB(paths.resolve_data_dir(config) / "state.db")

    # 3. Crash recovery (same as CLI)

    # 4. Build FastAPI app with lifespan
    @app.on_event("startup")
    async def startup():
        # Start background job worker thread
        threading.Thread(target=job_worker_loop, args=(db, config), daemon=True).start()

    @app.on_event("shutdown")
    async def shutdown():
        # Signal worker to stop, wait for current job
        pass

    # 5. Mount routes
    app.include_router(api_router)
    app.mount("/ws", websocket_router)
    app.mount("/", StaticFiles(directory="web/dist"), name="web")

    return app
```

### 2.4 Daemon Worker Startup (`core/scheduler.py`)

```python
def job_worker_loop(db, config):
    # 1. Initialize hardware detection (once per process)
    from core.gpu import cuda_usable
    gpu_ok = cuda_usable()
    print(f"  GPU available: {gpu_ok}")

    # 2. Initialize encoder probe (once per process)
    from video.encoding import pick_encoder
    encoder = pick_encoder(config.get("video", {}))
    print(f"  Encoder: {encoder}")

    # 3. Thread CPU affinity (once per process)
    from core.pipeline import _share_the_cpu
    _share_the_cpu(config)  # Sets cv2/torch thread counts

    # 4. Main loop
    while not shutdown_event.is_set():
        job = db.claim_next_job()
        if job:
            process_job(job, db, config)
        else:
            time.sleep(5)
```

### 2.5 Startup Sequence Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      PROCESS STARTUP                            │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌──────────┐        ┌──────────────┐    ┌──────────┐
    │  CLI     │        │  API Server  │    │  Daemon  │
    │ main.py  │        │ server/api.py│    │scheduler │
    └────┬─────┘        └──────┬───────┘    └────┬─────┘
         │                     │                 │
         └─────────────────────┼─────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │  load_config()       │
                    │  (Section 3)         │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  paths.resolve_data  │
                    │  _dir(config)        │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  StateDB(data_dir/   │
                    │  state.db)           │
                    │  - WAL mode          │
                    │  - Schema migration  │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Crash Recovery      │
                    │  - recover_stuck_    │
                    │    videos()          │
                    │  - recover_          │
                    │    interrupted_jobs()│
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Hardware Detection  │
                    │  (daemon/worker only)│
                    │  - cuda_usable()     │
                    │  - pick_encoder()    │
                    │  - _share_the_cpu()  │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  PIPELINE EXECUTION  │
                    │  (Section 4)         │
                    └──────────────────────┘
```

---

## 3. Configuration Loading Sequence

### 3.1 Configuration Sources (Priority Order)

```
HIGHEST PRIORITY
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. CLI Arguments (--vertical, --profile, --model, etc.)     │
│    → Direct overrides in process_cmd() before config load   │
├─────────────────────────────────────────────────────────────┤
│ 2. Environment Variables                                     │
│    → CLIPS_STUDIO_DATA_DIR, OPENROUTER_API_KEY, etc.        │
├─────────────────────────────────────────────────────────────┤
│ 3. User Config File: settings.yaml                           │
│    → Frozen: %LOCALAPPDATA%\Clips Studio\settings.yaml      │
│    → Dev:  <repo_root>\config\settings.yaml                 │
├─────────────────────────────────────────────────────────────┤
│ 4. Bundled Config: config/settings.yaml (shipped with app)  │
│    → Read-only defaults, used as fallback                   │
├─────────────────────────────────────────────────────────────┤
│ 5. Hardcoded Defaults (in Python code)                      │
│    → Final fallback for any missing key                     │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
LOWEST PRIORITY
```

### 3.2 `load_config()` Implementation (`core/config.py`)

```python
def load_config(config_path: str | None = None) -> dict:
    # 1. Start with bundled defaults
    bundled = BUNDLED_CONFIG_PATH  # config/settings.yaml in package
    config = yaml.safe_load(bundled.read_text())

    # 2. Overlay user config (if exists)
    if config_path:
        user_path = Path(config_path)
    else:
        user_path = paths.user_config_path(bundled)  # Handles frozen vs dev

    if user_path.exists():
        user_config = yaml.safe_load(user_path.read_text())
        config = deep_merge(config, user_config)

    # 3. Apply environment variable overrides
    config = apply_env_overrides(config)

    # 4. Resolve relative paths to absolute (data_dir, model paths, etc.)
    config = resolve_paths(config)

    return config
```

### 3.3 Environment Variable Overrides

| Environment Variable | Config Path | Description |
|---------------------|-------------|-------------|
| `CLIPS_STUDIO_DATA_DIR` | `paths.data_dir` | Override data directory (critical for Kaggle `/data`) |
| `OPENROUTER_API_KEY` | `llm.api_key` | Cloud LLM provider key |
| `OPENAI_API_KEY` | `llm.api_key` | OpenAI API key |
| `ANTHROPIC_API_KEY` | `llm.api_key` | Anthropic API key |
| `XAI_API_KEY` | `llm.api_key` | xAI Grok API key |
| `YTDLP_COOKIES` | `youtube_cookies` | Path to cookies.txt for YouTube |
| `CLIPS_STUDIO_FFMPEG` | `binaries.ffmpeg` | Custom FFmpeg path |
| `CLIPS_STUDIO_FFPROBE` | `binaries.ffprobe` | Custom ffprobe path |

**Application:** `core/config.py:apply_env_overrides()` walks a mapping and sets nested keys via dot notation.

### 3.4 Path Resolution (`core/paths.py`)

```python
def resolve_data_dir(config: dict) -> Path:
    """Resolves data_dir to absolute path."""
    raw = Path(str((config.get("paths") or {}).get("data_dir") or "data"))
    
    if raw.is_absolute():
        return raw
    
    if getattr(sys, "frozen", False):
        # Installed app: %LOCALAPPDATA%\Clips Studio\data
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "Clips Studio" / raw
    
    # Dev checkout: <repo_root>/data
    return _REPO_ROOT / raw
```

### 3.5 Kaggle-Specific Configuration

**Required for Kaggle:**
```yaml
# config/settings.yaml (or via env vars)
paths:
  data_dir: "/data"                    # Persistent volume mount

youtube_cookies: "file"                # Use /data/youtube_cookies.txt

whisper:
  model: "large-v3-turbo"              # Or "small" for VRAM savings
  device: "cuda"                       # Force GPU

video:
  parallel_renders: 1                  # T4: 1, A100: 2
  encoder: "auto"                      # NVENC on Kaggle GPU
```

**Environment variables in Kaggle notebook:**
```bash
export CLIPS_STUDIO_DATA_DIR=/data
export OPENROUTER_API_KEY=sk-...       # If using cloud LLM
```

### 3.6 Config Access Pattern in Pipeline

```python
# All modules access config via passed dict, never global
def process_video(url, config, db, ...):
    # Download
    download_config = config.get("download", {})
    
    # Transcription
    whisper_config = config.get("whisper", {})
    model_size = whisper_config.get("model", "auto")
    device = whisper_config.get("device", "auto")
    
    # Analysis
    clips_config = config.get("clips", {})
    max_clips = clips_config.get("max_count", 10)
    
    # Rendering
    video_config = config.get("video", {})
    encoder = video_config.get("encoder", "auto")
    parallel = video_config.get("parallel_renders", 1)
```
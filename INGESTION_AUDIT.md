# INGESTION_AUDIT.md — Downloading & Ingestion (Clips Kitty → Kaggle)

**Repository:** `E:\Clips kitty\clips-kitty-kaggle\`
**Date:** 2026-11-26

---

## 1. Entry Points

| Entry Point | Purpose | Called From |
|-------------|---------|-------------|
| `sources.dispatch.download(url, output_dir, vertical=False)` | **Primary download function** — routes to platform-specific downloader | `core/pipeline.py:_cached_or_download()` |
| `sources.dispatch.identify(url)` | Returns `(source_name, video_id)` for any supported URL | Pipeline, channel feed, preview frames |
| `sources.youtube.download(url, output_dir, vertical)` | YouTube VOD/Shorts download (H.264 selector) | `dispatch.download()` |
| `sources.twitch.download(url, output_dir, vertical)` | Twitch VOD/clip download | `dispatch.download()` |
| `sources.kick.download(url, output_dir, vertical)` | Kick VOD download | `dispatch.download()` |
| `sources.channel_feed.poll_channel(channel_id)` | RSS polling for YouTube new uploads | Daemon (`core/scheduler.py`) |
| `sources.channel_feed.latest(platform, channel_key)` | List newest videos for a watched channel | Daemon (`core/scheduler.py`) |
| `sources.channel_feed.readiness(url)` | Check if video is clippable (not live, not too short) | Daemon |
| `sources.vod_finder.find_stream_vod(platform, channel, start, end)` | Match OBS stream to published VOD | OBS plugin / stream workflows |
| `sources.preview_frames.frame(cache_dir, at, url|path)` | Extract preview frame for Gaming split setup | Electron UI (Generate bar) |
| `main.py:process` CLI command | `python main.py process <url>` — end-to-end pipeline | User / Kaggle notebook |
| `core.pipeline.process_video(url, config, db, force)` | Pipeline orchestration (download → transcribe → analyze → render) | CLI, API, daemon |

---

## 2. Core Functions by Module

### `sources/ytdlp_common.py` — Shared yt-dlp Infrastructure

| Function | Purpose |
|----------|---------|
| `extract_info(url, opts, download, process)` | **Central yt-dlp call** — retries IPv4/IPv6, handles bot checks, applies cookies |
| `progress_opts(video_id)` | yt-dlp options with progress hook, 6 concurrent fragments, 10 retries, 10MB chunks, exponential backoff |
| `_ffmpeg_dir()` | Returns bundled FFmpeg directory for yt-dlp merge (critical for YouTube DASH) |
| `games_from_info(info, platform)` | Extract game metadata (Twitch chapters, Kick category, YouTube tags) |
| `failure_note(error)` | Classifies yt-dlp errors: extractor bug (version note), network, or access |
| `set_cookies(value, data_dir)` / `cookies()` | YouTube cookies setting (firefox/file/off) |
| `_signed_in(opts)` | Context manager for signed-in YouTube requests (QuickJS for player JS) |
| `installed_version()` | Cached yt-dlp version string for diagnostics |

### `sources/youtube.py` — YouTube Source

| Function | Purpose |
|----------|---------|
| `download(url, output_dir, vertical)` | **Main entry** — probes for live, selects H.264 ≤1080p + m4a, merges to MP4 |
| `extract_video_id(url)` | Regex extraction from watch/shorts/live/youtu.be URLs |
| `resolve_channel(query)` | `@handle`, URL, or UC...id → `{channel_id, name}` via yt-dlp |
| `poll_channel(channel_id)` | RSS feed fetch → `[{video_id, title, url, published, short}]` |
| `_friendly_errors()` / `_friendly_message()` | Human-readable errors: 403 (rate limit), age-gate, bot check |

### `sources/twitch.py` — Twitch Source

| Function | Purpose |
|----------|---------|
| `download(url, output_dir, vertical)` | VOD/clips only (rejects live channels); `best[ext=mp4]/best` format |
| `extract_vod_id(url)` / `extract_clip_id(url)` | `tw_<numeric>` / `tw_clip_<slug>` prefixes |
| `is_twitch_url(url)` | Host-based check (`twitch.tv`) |

### `sources/kick.py` — Kick Source

| Function | Purpose |
|----------|---------|
| `download(url, output_dir, vertical)` | VOD only; Cloudflare impersonation via `curl_cffi` |
| `extract_vod_id(url)` | `kick_<uuid>` prefix |
| `_impersonation()` | Returns `{"impersonate": "chrome"}` for yt-dlp if `curl_cffi` available |
| `_from_web_api(ydl, url)` | Fallback to Kick's newer API (`web.kick.com/api/v1/...`) when yt-dlp extractor 404s |
| `_gone_or_raise()` | Translates 404 → "VOD deleted" message |

### `sources/dispatch.py` — Router & Utilities

| Function | Purpose |
|----------|---------|
| `download(url, output_dir, vertical)` | Routes to platform downloader; clears stale `.part` fragments first |
| `identify(url)` | `(source, video_id)` — `local:`, Twitch, Kick, YouTube |
| `platform_of_link(url)` | Platform from hostname for attribution |
| `game_info(url)` | Game metadata without download (for cached files) |
| `description(url)` | Video description without download (sports hotwords) |
| `metadata(url)` | `(title, channel)` without download (creator tagging) |

### `sources/channel_feed.py` — Channel Watching (Daemon)

| Function | Purpose |
|----------|---------|
| `resolve(platform, text)` | Input → `Channel(platform, channel_key, name)` |
| `latest(platform, channel_key)` | Newest 15 videos → `NewSourceVideo[]` |
| `readiness(url, min_seconds)` | `Readiness(state="ready|not_yet|skip", reason, orientation)` |
| `_youtube_latest` | RSS primary, uploads playlist fallback |
| `_kick_latest` | `kick.com/api/v2/channels/<slug>/videos` |
| `_twitch_listing` | `twitch.tv/<handle>/videos?filter=archives&sort=time` |
| `_orientation(info)` | Detects vertical formats from metadata (no download) |

### `sources/vod_finder.py` — OBS Stream → VOD Matching

| Function | Purpose |
|----------|---------|
| `find_stream_vod(platform, channel, started_at, ended_at)` | Matches stream timestamps to published VOD (±20 min tolerance) |
| `pick(entries, started_at, ended_at)` | Best match by start-time gap; rejects <50% stream length |
| `listing_url(platform, channel)` | Channel VOD listing URL |
| `_extract(url, flat, size)` | yt-dlp metadata extraction (flat for listing, full for candidates) |

### `sources/preview_frames.py` — Preview Frames (UI Only)

| Function | Purpose |
|----------|---------|
| `frame(cache_dir, at, url|path)` | JPEG at fractional position (0-1); cached; uses FFmpeg seek |
| `_probe_link(url)` | yt-dlp media URL + duration + headers; YouTube relayed via IPv4 loopback |
| `_Relay` | ThreadingHTTPServer on 127.0.0.1 — forwards FFmpeg range requests over IPv4 |
| `made_once(folder, name, make)` | Atomic file creation with per-file locking (prevents Windows replace failure) |

### `sources/vertical.py` — Vertical Live Download

| Constant/Function | Purpose |
|-------------------|---------|
| `SUFFIX = "__vertical"` | Filename suffix to separate from horizontal download |
| `YOUTUBE_FORMAT` | Portrait H.264 + m4a, avoids AV1, sorts by `res:1080` |
| `HLS_FORMAT` | Twitch/Kick: `best[aspect_ratio<1][ext=mp4]` |
| `no_vertical_version(platform)` | User-facing message when vertical format missing |
| `is_missing_format(error)` | Detects yt-dlp "Requested format is not available" |

### `sources/urlmatch.py` — Host-Based URL Routing

| Function | Purpose |
|----------|---------|
| `host_matches(url, domain)` | True if host == domain or subdomain (label-boundary safe) |

---

## 3. Dependencies

### Python Packages (from `requirements.txt`)

| Package | Version | Purpose |
|---------|---------|---------|
| `yt-dlp` | Latest (bundled in build) | All platform downloads |
| `curl_cffi` | Optional | Kick Cloudflare impersonation (Chrome TLS fingerprint) |
| `defusedxml` | - | Safe XML parsing for YouTube RSS |
| `requests` | - | HTTP for RSS, Kick API, YouTube channel resolve |
| `ffmpeg` / `ffprobe` | System/bundled | Media probing, merging, frame extraction |
| `quickjs` | Bundled binary | YouTube player JS execution (signed-in cookies) |

### External Binaries (resolved via `core/binaries.py`)

| Binary | Source | Kaggle Note |
|--------|--------|-------------|
| `ffmpeg` / `ffprobe` | Bundled (Windows) / System (Linux) | `apt-get install ffmpeg` |
| `ollama` | Bundled / System service | Not needed for ingestion; sidecar for LLM |
| `qjs` (QuickJS) | Bundled | Only for YouTube signed-in cookies; skip if using cloud LLM |

### Platform-Specific Requirements

| Platform | Special Requirements |
|----------|---------------------|
| **YouTube** | H.264 format selector (avoids AV1); RSS feed (no API quota); bot-check retry (IPv4/IPv6); optional cookies (firefox/cookies.txt) |
| **Twitch** | VODs only; numeric IDs prefixed `tw_`; `best[ext=mp4]` format |
| **Kick** | Cloudflare → needs `curl_cffi` for Chrome impersonation; fallback to web API; UUID IDs prefixed `kick_` |

---

## 4. Kaggle Issues & Mitigations

### 4.1 Blocking Issues (Must Fix)

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **No bundled FFmpeg on Linux** | yt-dlp merge fails for YouTube DASH (separate video+audio) | `apt-get install ffmpeg` in Dockerfile; ensure on PATH |
| **`curl_cffi` not installed** | Kick downloads 403 (Cloudflare) | `pip install curl_cffi` OR accept Kick may fail |
| **YouTube bot check** | IPv6 routes often flagged as bot on cloud IPs | Set `CLIPS_STUDIO_DATA_DIR` with `youtube_cookies.txt` (Netscape format) |
| **QuickJS binary missing** | Signed-in YouTube cookies need JS execution | Use cloud LLM + cookies.txt; skip QuickJS |
| **yt-dlp version frozen at build** | Site changes break extractors | Pin `yt-dlp` in requirements OR allow `pip install -U yt-dlp` at runtime |

### 4.2 Performance Issues

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **No NVDEC for download** | Download uses software decode only | Not critical — download is I/O bound; NVDEC used later in pipeline |
| **Concurrent fragment downloads (6)** | May hit rate limits on shared Kaggle egress | Reduce `concurrent_fragment_downloads` to 3 in `ytdlp_common.progress_opts` |
| **10 MB HTTP chunks** | Good for resilience; keep as-is | No change needed |

### 4.3 Functional Gaps (Headless/Container)

| Feature | Status | Workaround |
|---------|--------|------------|
| **Firefox cookies** (`cookiesfrombrowser: ("firefox",)`) | **Won't work** — no browser in container | Use `cookies.txt` file via `youtube_cookies` config |
| **QuickJS for player JS** | Binary not on Linux | Not needed if cookies.txt provided |
| **Channel RSS polling (daemon)** | Works headless | Run `python main.py run` in background |
| **Preview frames** | Needs FFmpeg + loopback HTTP relay | Works in container (127.0.0.1 only); UI not available |
| **VOD finder (OBS integration)** | Works if timestamps provided | Not needed for batch processing |

### 4.4 Configuration for Kaggle

```yaml
# config/settings.yaml additions for container
paths:
  data_dir: "/data"                    # Persistent volume

youtube_cookies: "file"                # Use cookies.txt in data_dir
# Place youtube_cookies.txt in /data/ (Netscape format)

# Optional: reduce parallelism for shared egress
# (in sources/ytdlp_common.py progress_opts)
# "concurrent_fragment_downloads": 3,
```

### 4.5 Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `CLIPS_STUDIO_DATA_DIR` | **Yes** | Override data dir to persistent volume (`/data`) |
| `YTDLP_COOKIES` / `youtube_cookies` in config | Recommended | YouTube cookies for age-gated/bot-blocked content |
| `CLIPS_STUDIO_FFMPEG` / `CLIPS_STUDIO_FFPROBE` | No | Override if FFmpeg not on PATH |

---

## 5. Data Flow Diagram

```
┌─────────────┐
│   URL Input │  (CLI, API, Daemon RSS, OBS webhook)
└──────┬──────┘
       ▼
┌──────────────────┐
│ dispatch.identify│  → (source, video_id)
└──────┬───────────┘
       ▼
┌──────────────────┐
│ dispatch.download│  → clears stale .part files
└──────┬───────────┘
       ▼
┌──────────────────┐
│ Platform Download│  youtube.py / twitch.py / kick.py
│  (yt-dlp)        │
│  - format select │
│  - progress hook │
│  - IPv4/IPv6 retry│
│  - cookies       │
│  - FFmpeg merge  │
└──────┬───────────┘
       ▼
┌──────────────────┐
│ DownloadedVideo  │  {video_id, title, path, duration, channel, games}
└──────┬───────────┘
       ▼
┌──────────────────┐
│ core/pipeline.py │  → transcribe → analyze → render
│ process_video()  │
└──────────────────┘
```

---

## 6. Key Files to Modify for Kaggle

| File | Change |
|------|--------|
| `Dockerfile` | `apt-get install ffmpeg curl` + `pip install curl_cffi` |
| `sources/ytdlp_common.py` | Reduce `concurrent_fragment_downloads` to 3 (optional) |
| `config/settings.yaml` | Set `data_dir: "/data"`, `youtube_cookies: "file"` |
| `scripts/precache_models.py` | **New** — not needed for ingestion (no ML models) |
| `requirements.txt` | Add `curl_cffi` (optional but recommended for Kick) |

---

## 7. Test Checklist for Kaggle

| Test | Command | Expected |
|------|---------|----------|
| YouTube download | `python main.py process "https://youtu.be/<id>"` | MP4 in `/data/downloads/<id>.mp4` |
| YouTube Shorts | `python main.py process "https://youtube.com/shorts/<id>"` | Works (treated as normal video) |
| Twitch VOD | `python main.py process "https://twitch.tv/videos/<id>"` | MP4 with `tw_` prefix |
| Kick VOD | `python main.py process "https://kick.com/video/<uuid>"` | MP4 with `kick_` prefix (needs `curl_cffi`) |
| Age-gated YouTube | With `youtube_cookies.txt` in `/data/` | Downloads successfully |
| Channel RSS poll | `python main.py channels add @handle && python main.py run` | Detects new uploads |
| Vertical Live | `python main.py process <vertical_url> --vertical` | Downloads portrait format only |

---

## 8. Summary

**Ingestion is fully headless-compatible.** The download pipeline (`sources/`) has:
- ✅ No GUI dependencies
- ✅ No display server needed
- ✅ Pure Python + FFmpeg + yt-dlp
- ✅ Works in Linux containers

**Only 3 things needed for Kaggle:**
1. **System FFmpeg** (`apt-get install ffmpeg`)
2. **YouTube cookies** (`youtube_cookies.txt` in persistent volume)
3. **`curl_cffi`** for Kick (optional: `pip install curl_cffi`)

The rest of the pipeline (transcription, analysis, render) is documented in `KAGGLE_ARCHITECTURE_AUDIT.md`.
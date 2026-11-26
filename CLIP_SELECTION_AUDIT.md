# CLIP_SELECTION_AUDIT.md — Clip Selection Pipeline (Clips Kitty → Kaggle)

**Repository:** `E:\Clips kitty\clips-kitty-kaggle\`
**Date:** 2026-11-26

---

## 1. Overview

The clip selection pipeline is the core intelligence layer that transforms a transcribed video into ranked, deduplicated clip candidates ready for rendering. It lives primarily in `analysis/` and operates in three phases:

```
Phase 1: Candidate Generation
  ├─ Transcript windows (LLM-scored)
  ├─ Signal peaks (audio/visual/reaction/hype)
  └─ Gaming/Sports/Intent augmentations

Phase 2: Fusion Scoring & Deduplication
  ├─ Multimodal fusion (5 subscores → composite)
  ├─ Adaptive weighting per profile
  ├─ Bonus system (creator, action, hype, game, intent, sport, menu penalty)
  ├─ 3-pass deduplication (timestamp, transcript, segment reuse)
  └─ LLM rerank (relative judgment in batches)

Phase 3: Selection & Metadata
  ├─ Selection v1 (greedy) / v2 (budget-aware)
  ├─ Boundary fitting to sentence ends
  └─ Metadata generation (titles, descriptions, hashtags)
```

---

## 2. Phase 1: Candidate Generation

### 2.1 Transcript Windows (`analysis/highlights.py`)

**Entry:** `find_highlights(segments, video_id, duration, config, hotwords, ...)`

| Step | Function | Detail |
|------|----------|--------|
| 1 | Chunking | Transcript split into overlapping chunks (~10 min each, 30s overlap) |
| 2 | LLM Scoring | Each chunk → `score_windows()` → LLM rates 30s windows 0-100 |
| 3 | Candidate Building | Windows with score ≥ threshold → `ClipCandidate` objects |
| 4 | Sentence Fitting | `_fit_to_segments()` snaps boundaries to segment ends |
| 5 | Dedupe | `_select_unique()` removes overlaps / similar transcripts / segment reuse |

**LLM Prompt (`config/prompts/score_windows.txt`):**
- Input: Transcript chunks with timestamps
- Output: JSON array of `{start, end, score, reason}` for each 30s window
- Model: Configurable via `llm.backend` (Ollama, OpenRouter, OpenAI, etc.)

**Chunking Parameters:**
```python
CHUNK_SECONDS = 600      # 10 minutes per LLM call
OVERLAP_SECONDS = 30     # Overlap to catch boundary moments
MIN_WINDOW = 15          # Minimum clip length (seconds)
MAX_WINDOW = 60          # Maximum clip length (seconds)
```

**`_select_unique()` — 3-pass deduplication:**
1. **Timestamp Overlap:** Skip if `IoU > 0.5` with any kept candidate
2. **Transcript Similarity:** Skip if Jaccard(word_sets) > 0.7
3. **Segment Reuse:** Skip if >50% of underlying Whisper segments already used

---

### 2.2 Signal Peaks (`analysis/fusion.py`)

**Entry:** `find_clips(segments, signals, video_id, duration, profile, config, ...)`

Signal curves are per-second 0-1 percentile-ranked arrays from:
| Signal | Source | Module |
|--------|--------|--------|
| **Audio** | Loudness, spike, burst, noisiness | `audio_features.py:extract_audio_features()` |
| **Visual** | Motion, scene_cut, flash | `visual_features.py:extract_visual_features()` |
| **Reaction** | YOLOv8 pose (streamer excitement) | `visual_features.py:reaction_for_window()` |
| **Hype** | Chat replay (Twitch/YouTube) | `hype.py:audience_curve()` |
| **Game Audio** | PANNs (crowd, whistle, etc.) | `game_audio.py` (gaming profile) |
| **Game Vision** | YOLO ball detection | `game_vision.py` (gaming profile) |
| **Sports OCR** | Scorebox reading | `sports/` (sports profile) |

**Peak Detection:**
```python
def _peaks(curve: np.ndarray, min_gap: int = 8) -> list[int]:
    """Local maxima with minimum gap between peaks."""
    # Finds indices where curve[i] > curve[i-1] and curve[i] > curve[i+1]
    # Enforces min_gap seconds between peaks
```

**Candidate Pool Construction:**
- Each peak → candidate window centered at peak, grown to `target_duration` (25s default)
- Windows clamped to video boundaries
- Transcript windows + signal peaks merged (deduped by timestamp IoU)

---

### 2.3 Profile-Specific Augmentations

| Profile | Augmentation | Module |
|---------|-------------|--------|
| **Gaming** | Streamer face tracking, webcam/game layout detection, 11 composition layouts, callouts, game sounds (PANNs), ball detection (YOLO) | `gaming/run.py`, `game_audio.py`, `game_vision.py` |
| **Sports** | Scorebox OCR (RapidOCR), ball tracking, commentary keywords, moment windows, story reels | `sports/` |
| **Intent** | LLM direction classification (tutorial, review, story, etc.) | `intent.py` |
| **Direction** | Camera movement / framing analysis | `direction.py` |

---

## 3. Phase 2: Fusion Scoring

### 3.1 Five Subscores (`analysis/fusion.py:_score_candidate()`)

Each candidate gets 5 normalized subscores (0-1), then combined:

| Subscore | Source | Computation |
|----------|--------|-------------|
| **Text** | LLM window score | Normalized to 0-1 (divide by 100) |
| **Visual** | Visual signal curve | Mean of `visual[window]` |
| **Reaction** | YOLO pose reaction | Mean of `reaction[window]` (top-k streamer boxes) |
| **Audio** | Audio signal curve | Mean of `audio[window]` |
| **Engagement** | Hype curve | Mean of `hype[window]` |

**Adaptive Weighting:**
```python
# Base weights (configurable per profile)
weights = {
    "text": 0.35,
    "visual": 0.20,
    "reaction": 0.15,
    "audio": 0.15,
    "engagement": 0.15,
}

# Gaming profile shifts weight toward reaction + visual
# Sports profile shifts toward engagement + audio (crowd)
# Podcast/intent profile shifts toward text
```

**Fusion Formula:**
```python
composite = sum(w * s for w, s in zip(weights.values(), subscores))
# Then bonuses applied (see 3.2)
final_score = min(1.0, composite + sum(bonuses))
```

---

### 3.2 Bonus System

| Bonus | Trigger | Max Value | Notes |
|-------|---------|-----------|-------|
| **Creator Context** | Creator knowledge matches content | +0.10 | From `creator/` learning |
| **Action** | High visual motion + reaction | +0.08 | Gaming: combat, clutch moments |
| **Hype** | Chat peak in window | +0.07 | Capped; unique chatters only |
| **Game** | Game sound/vision detection | +0.06 | PANNs crowd/whistle + YOLO ball |
| **Intent** | LLM direction matches profile | +0.05 | Tutorial→educational, story→narrative |
| **Sport** | Score change / key moment | +0.08 | OCR-detected score events |
| **Menu Penalty** | Static UI / menu detected | -0.15 | Negative bonus (penalty) |

**Bonus Application:**
- Bonuses are additive but **hard-capped** so text/visual context stays dominant
- Total bonus ≤ 0.25 (prevents signal-only clips from outranking substantive content)
- Menu penalty can push score below threshold → candidate rejected

---

### 3.3 Deduplication (3 Checks)

After scoring, candidates sorted by score descending, then:

| Pass | Check | Threshold | Action |
|------|-------|-----------|--------|
| 1 | **Timestamp IoU** | > 0.5 | Reject lower-scored |
| 2 | **Transcript Jaccard** | > 0.7 | Reject lower-scored |
| 3 | **Segment Reuse** | > 50% segments used | Reject lower-scored |

**Segment Reuse Logic:**
- Tracks which Whisper segments (by index) have been "consumed"
- Prevents the same spoken content from appearing in multiple clips
- Critical for long videos with repetitive commentary

---

### 3.4 LLM Rerank (`analysis/fusion.py:_rerank()`)

**Purpose:** Relative judgment — LLM compares candidates pairwise to fix fusion artifacts.

**Process:**
1. Take top-K candidates (default K=20)
2. Batch into groups of 5
3. LLM prompt (`config/prompts/rerank.txt`): "Rank these 5 clips by viral potential"
4. Merge rankings → final order

**Batching:**
```python
BATCH_SIZE = 5
TOP_K_RERANK = 20
```

**Why Relative Judgment?**
- Absolute scoring (0-100) has calibration drift across videos
- Relative comparison within a video is more stable
- Corrects for: over-scored quiet moments, under-scored chaotic moments

---

## 4. Phase 3: Selection & Metadata

### 4.1 Selection Strategies

| Strategy | Function | Logic |
|----------|----------|-------|
| **v1 (Greedy)** | `_select_v1()` | Take highest-scored until `max_clips` or `min_score` threshold |
| **v2 (Budget-Aware)** | `_select_v2()` | Allocate "clip budget" across video; prefer diversity; respect min_gap |

**v2 Budget Allocation:**
```python
# Target: N clips across duration D
# Ideal spacing: D / N
# But allows clustering around genuine peaks
# Enforces minimum gap between selected clips (default 30s)
```

**Config (settings.yaml):**
```yaml
clips:
  max_count: 10           # Maximum clips per video
  min_score: 0.6          # Fusion score threshold
  min_gap_seconds: 30     # Minimum time between clips
  selection_version: 2    # 1 or 2
```

---

### 4.2 Boundary Fitting (`analysis/highlights.py:_fit_to_segments()`)

**Purpose:** Snap clip boundaries to sentence/segment ends for natural cuts.

**Algorithm:**
1. For each candidate (start, end), find overlapping Whisper segments
2. Extend start backward to previous segment boundary (if < 2s)
3. Extend end forward to next segment boundary (if < 2s)
4. Clamp to `MIN_DURATION` / `MAX_DURATION`
5. Re-score if boundaries shifted significantly

---

### 4.3 Metadata Generation (`analysis/metadata.py`)

**Entry:** `generate_metadata(candidates, segments, video_info, config)` or `generate_metadata_batch()`

| Field | Source |
|-------|--------|
| **Title** | LLM prompt (`config/prompts/metadata.txt`) — viral-style, <100 chars |
| **Description** | LLM — 2-3 sentences, hooks, CTAs |
| **Hashtags** | LLM — 3-5 relevant tags, platform-aware |
| **Sport Scoreboard** | Sports profile — OCR score state at clip time |

**Batch Processing:**
- Multiple candidates sent in single LLM call for efficiency
- Prompt includes all candidate transcripts + context
- Returns structured JSON for each clip

**Fallback:**
- If LLM fails: template-based title from first sentence + generic description
- Hashtags from video tags / game name / sport

---

## 5. Key Configuration (settings.yaml)

```yaml
# Clip selection
clips:
  max_count: 10
  min_score: 0.6
  min_duration: 15
  max_duration: 60
  min_gap_seconds: 30
  selection_version: 2        # 1=greedy, 2=budget-aware
  vertical:
    enabled: true
    crop_mode: "track"        # track | split | fit_blur

# Fusion weights (profile-specific overrides in gaming.yaml, sports.yaml)
fusion:
  weights:
    text: 0.35
    visual: 0.20
    reaction: 0.15
    audio: 0.15
    engagement: 0.15
  bonuses:
    creator_context: 0.10
    action: 0.08
    hype: 0.07
    game: 0.06
    intent: 0.05
    sport: 0.08
    menu_penalty: -0.15
  bonus_cap: 0.25

# LLM scoring
llm:
  backend: "openrouter"
  model: "anthropic/claude-3.5-sonnet"
  temperature: 0.3
  max_tokens: 2000
```

---

## 6. GPU/CPU Requirements

| Component | GPU Path | CPU Fallback | Kaggle Notes |
|-----------|----------|--------------|--------------|
| **YOLOv8 Pose (reaction)** | CUDA (ultralytics) | `device="cpu"` | Pre-download `yolov8n-pose.pt` to `models/` |
| **YOLOv8 Ball (gaming)** | CUDA | `device="cpu"` | Pre-download `yolov8n.pt` |
| **TalkNet (speaker tracking)** | CUDA (PyTorch) | `device="cpu"` | Vendored in `third_party/talknet/` |
| **PANNs (game audio)** | — | onnxruntime CPU | Model auto-downloads; pre-cache |
| **RapidOCR (sports)** | — | onnxruntime CPU | Models auto-download; pre-cache |
| **LLM Scoring** | Cloud API (preferred) | Ollama local | Use cloud to avoid sidecar |
| **FFmpeg Decode** | NVDEC (`-hwaccel auto`) | Software | Works in container |
| **Whisper** | See TRANSCRIPTION_AUDIT.md | | |

---

## 7. Memory Profile (Kaggle T4 16GB)

| Component | VRAM |
|-----------|------|
| Whisper large-v3-turbo | ~6 GB |
| YOLOv8n-pose (tracking) | ~500 MB |
| YOLOv8n (ball, gaming) | ~300 MB |
| TalkNet | ~1.5 GB |
| PANNs / RapidOCR / Voice | CPU RAM only |
| NVENC encode (1 stream) | ~500 MB |
| **Total (1 render worker)** | **~8.8 GB** ✅ |
| **Total (2 render workers)** | **~12 GB** ⚠️ |

**Recommendation:** `video.parallel_renders = 1` on T4.

---

## 8. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIP SELECTION PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: Whisper segments + Signal curves + Video metadata       │
│  - segments: [{start, end, text, words[]}]                      │
│  - signals: {audio, visual, reaction, hype, game_audio, ...}    │
│  - profile: "standard" | "gaming" | "sports" | "podcast"        │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
         ┌─────────────────┐       ┌─────────────────┐
         │ Transcript      │       │ Signal Peaks    │
         │ Windows (LLM)   │       │ (per-modality)  │
         │ score_windows() │       │ _peaks()        │
         └────────┬────────┘       └────────┬────────┘
                  │                         │
                  └───────────┬─────────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Candidate Pool      │
                    │ (merged, deduped    │
                    │  by timestamp IoU)  │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Fusion Scoring      │
                    │ 5 subscores +       │
                    │ adaptive weights +  │
                    │ bonus system        │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ 3-Pass Dedupe       │
                    │ 1. Timestamp IoU    │
                    │ 2. Transcript Jacc  │
                    │ 3. Segment Reuse    │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ LLM Rerank          │
                    │ (relative judgment  │
                    │  in batches of 5)   │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Selection v1/v2     │
                    │ + Boundary Fit      │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Metadata Gen        │
                    │ (title, desc, tags) │
                    └─────────────────────┘
```

---

## 9. Files to Modify for Kaggle

| File | Change |
|------|--------|
| `config/settings.yaml` | Set fusion weights, selection version, clip limits |
| `config/gaming.yaml` | Game genres, highlight defs, sound groups (if using gaming) |
| `config/sports.yaml` | Sport definitions, OCR zones (if using sports) |
| `config/prompts/*.txt` | All LLM prompts editable — tune for your content |
| `analysis/fusion.py` | No changes needed — config-driven |
| `analysis/highlights.py` | No changes needed — config-driven |
| `analysis/metadata.py` | No changes needed — config-driven |
| `Dockerfile` | Pre-download YOLO, TalkNet, PANNs, OCR models |

---

## 10. Pre-caching Script for Models

```python
# scripts/precache_clip_models.py
import os
from ultralytics import YOLO
import shutil

MODELS_DIR = "/app/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# YOLOv8 pose (reaction tracking)
print("Downloading yolov8n-pose.pt...")
YOLO("yolov8n-pose.pt")
shutil.copy(os.path.expanduser("~/.cache/ultralytics/yolov8n-pose.pt"), MODELS_DIR)

# YOLOv8 ball (gaming)
print("Downloading yolov8n.pt...")
YOLO("yolov8n.pt")
shutil.copy(os.path.expanduser("~/.cache/ultralytics/yolov8n.pt"), MODELS_DIR)

# TalkNet (vendored — copy from third_party)
# PANNs, RapidOCR, Voice models — download on first run or pre-cache similarly
```

---

## 11. Test Checklist for Kaggle

| Test | Command | Expected |
|------|---------|----------|
| Standard profile | `python main.py process <url>` | Clips in `/data/clips/...` |
| Gaming profile | `python main.py process <url> --profile gaming` | Streamer-tracked clips |
| Sports profile | `python main.py process <url> --profile sports` | Score-aware clips |
| Vertical render | `python main.py process <url> --vertical` | 9:16 crops |
| Metadata quality | Check clip JSON sidecars | Title, desc, hashtags present |
| Dedupe verification | Long video (2h+) | No duplicate content across clips |
| Boundary fitting | Inspect clip timestamps | Align to sentence ends |

---

## 12. Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| **Candidate Generation** | ✅ Complete | Transcript LLM + signal peaks |
| **Fusion Scoring (5 subscores)** | ✅ Complete | Adaptive weights per profile |
| **Bonus System** | ✅ Complete | 7 bonuses + menu penalty, capped |
| **3-Pass Deduplication** | ✅ Complete | Timestamp, transcript, segment reuse |
| **LLM Rerank** | ✅ Complete | Relative judgment, batched |
| **Selection v1/v2** | ✅ Complete | Budget-aware v2 recommended |
| **Boundary Fitting** | ✅ Complete | Sentence-aware |
| **Metadata Generation** | ✅ Complete | Batch LLM, sport scoreboard |
| **GPU Dependencies** | ⚠️ YOLO/TalkNet | Pre-cache models; CPU fallback works |
| **Headless Compatible** | ✅ Yes | Zero display deps in analysis/ |
| **Config-Driven** | ✅ Yes | All weights/thresholds in YAML |

**The clip selection pipeline is fully headless, config-driven, and Kaggle-ready.** The only GPU dependencies are YOLOv8 (pose + ball) and TalkNet — both have reliable CPU fallbacks and pre-cachable weights.
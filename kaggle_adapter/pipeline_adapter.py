"""
Kaggle Pipeline Adapter
=======================

Provides a thin adapter layer around the original Clips Kitty pipeline
for Kaggle notebook execution. This module does NOT modify the original
pipeline — it wraps and configures it for the Kaggle environment.

All processing uses the ORIGINAL Clips Kitty functions:
- Ingestion: sources.dispatch.download() -> sources.youtube/twitch/kick.download()
- Transcription: transcription.transcriber.transcribe()
- Signal extraction: analysis.audio_features/visual_features.extract_*()
- Clip selection: analysis.fusion.find_clips() + LLM rerank
- Rendering: video.cutter.cut_clip() + video.cropper.render_vertical() + video.encoding

Original entry point: core.pipeline.process_video()
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# Original Clips Kitty imports (these are the REAL pipeline functions)
from core.models import ClipCandidate, RenderedClip
from core.pipeline import process_video
from core.state import StateDB
from kaggle_adapter.config import KaggleConfig, quick_setup, get_pipeline_config


@dataclass
class PipelineProgress:
    """Progress state for a pipeline run."""
    stage: str = "idle"
    message: str = ""
    video_id: str = ""
    current: int = 0
    total: int = 0
    percent: float = 0.0
    elapsed_seconds: float = 0.0
    details: dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""
    success: bool
    clips: list[RenderedClip] = field(default_factory=list)
    video_id: str = ""
    video_title: str = ""
    error: Optional[str] = None
    processing_time_seconds: float = 0.0
    stages_completed: list[str] = field(default_factory=list)


class PipelineAdapter:
    """
    Adapter for running the original Clips Kitty pipeline in Kaggle.

    This class wraps core.pipeline.process_video() with:
    - Kaggle-optimized configuration (via kaggle_adapter.config)
    - Progress monitoring callbacks
    - Error handling and recovery
    - Notebook-friendly output paths
    """

    # Verified original functions called by this adapter
    VERIFIED_FUNCTIONS = {
        "process_video": {
            "source": "core.pipeline",
            "function": "process_video",
            "verified": True,
            "notes": "Main pipeline entry point - runs full 14-stage pipeline"
        },
        "download": {
            "source": "sources.dispatch",
            "function": "download",
            "verified": True,
            "notes": "Routes to youtube/twitch/kick downloaders via yt-dlp"
        },
        "transcribe": {
            "source": "transcription.transcriber",
            "function": "transcribe",
            "verified": True,
            "notes": "Uses faster-whisper with CTranslate2; GPU requires cuBLAS 12"
        },
        "extract_audio_features": {
            "source": "analysis.audio_features",
            "function": "extract_audio_features",
            "verified": True,
            "notes": "Runs in background thread during transcription"
        },
        "extract_visual_features": {
            "source": "analysis.visual_features",
            "function": "extract_visual_features",
            "verified": True,
            "notes": "Runs in background thread during transcription"
        },
        "find_clips": {
            "source": "analysis.fusion",
            "function": "find_clips",
            "verified": True,
            "notes": "Candidate generation + fusion scoring + LLM rerank"
        },
        "cut_clip": {
            "source": "video.cutter",
            "function": "cut_clip",
            "verified": True,
            "notes": "Frame-exact cuts via FFmpeg + ShotCuts detection"
        },
        "render_vertical": {
            "source": "video.cropper",
            "function": "render_vertical",
            "verified": True,
            "notes": "YOLOv8 + TalkNet tracking with 3 vertical modes"
        },
        "generate_metadata_batch": {
            "source": "analysis.metadata",
            "function": "generate_metadata_batch",
            "verified": True,
            "notes": "Batched LLM calls for titles/descriptions/hashtags"
        },
    }

    def __init__(
        self,
        config: Optional[KaggleConfig] = None,
        data_dir: str = "/data",
        use_cloud_llm: bool = True,
        force_cpu: bool = False,
        progress_callback: Optional[Callable[[PipelineProgress], None]] = None,
        verbose: bool = True,
    ):
        """
        Initialize the pipeline adapter.

        Args:
            config: Pre-configured KaggleConfig (optional, will create if not provided)
            data_dir: Persistent data directory (default: /data)
            use_cloud_llm: Use cloud LLM backend (requires API key in Kaggle Secrets)
            force_cpu: Force CPU mode even if GPU available
            progress_callback: Optional callback for progress updates
            verbose: Print progress to console
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.use_cloud_llm = use_cloud_llm
        self.force_cpu = force_cpu
        self.progress_callback = progress_callback
        self.verbose = verbose

        self._db: Optional[StateDB] = None
        self._current_video_id: Optional[str] = None
        self._start_time: float = 0.0
        self._cancelled = False
        self._lock = threading.Lock()

        # Stage tracking for progress
        self._stages = [
            "download", "transcribe", "analyze", "render"
        ]
        self._stage_weights = {
            "download": 0.15,
            "transcribe": 0.35,
            "analyze": 0.20,
            "render": 0.30,
        }
        self._completed_stages: set[str] = set()

    def _setup_config(self) -> KaggleConfig:
        """Create or reuse Kaggle configuration."""
        if self.config is None:
            self.config = quick_setup(
                data_dir=str(self.data_dir),
                use_cloud_llm=self.use_cloud_llm,
                force_cpu=self.force_cpu,
                verbose=self.verbose,
            )
        return self.config

    def _get_db(self) -> StateDB:
        """Get or create StateDB instance."""
        if self._db is None:
            db_path = self.data_dir / "state.db"
            self._db = StateDB(db_path)
        return self._db

    def _emit_progress(self, progress: PipelineProgress) -> None:
        """Emit progress to callback and console."""
        progress.elapsed_seconds = time.monotonic() - self._start_time

        if self.progress_callback:
            try:
                self.progress_callback(progress)
            except Exception:
                pass  # Progress callback must never break pipeline

        if self.verbose:
            bar = self._progress_bar(progress.percent)
            print(f"\r  {bar} {progress.stage}: {progress.message}", end="", flush=True)
            if progress.stage == "done" or progress.error:
                print()  # Newline on completion

    def _progress_bar(self, percent: float, width: int = 30) -> str:
        """Create a simple text progress bar."""
        filled = int(width * percent / 100)
        return "[" + "=" * filled + " " * (width - filled) + "]"

    def _calculate_overall_percent(self, stage: str, stage_percent: float = 0) -> float:
        """Calculate overall progress percentage."""
        total = 0.0
        for s in self._stages:
            weight = self._stage_weights.get(s, 0)
            if s in self._completed_stages:
                total += weight * 100
            elif s == stage:
                total += weight * stage_percent
        return min(total, 100.0)

    def _progress_handler(self, event: dict) -> None:
        """Handle progress events from the original pipeline."""
        # Map pipeline stages to our progress stages
        stage_map = {
            "download": "download",
            "downloaded": "download",
            "transcribe": "transcribe",
            "analyze": "analyze",
            "render": "render",
            "done": "done",
        }

        stage = event.get("stage", "")
        mapped_stage = stage_map.get(stage, stage)

        if mapped_stage in self._stage_weights and mapped_stage not in self._completed_stages:
            if stage in ("downloaded", "transcribed", "analyzed", "done"):
                self._completed_stages.add(mapped_stage)

        video_id = event.get("video_id", "")
        message = event.get("message", "")
        clip = event.get("clip", 0)
        total = event.get("total", 0)

        if mapped_stage == "render" and total > 0:
            stage_percent = (clip / total) * 100 if clip > 0 else 0
        else:
            stage_percent = 0

        overall = self._calculate_overall_percent(mapped_stage, stage_percent)

        progress = PipelineProgress(
            stage=mapped_stage,
            message=message or f"Processing {mapped_stage}...",
            video_id=video_id,
            current=clip,
            total=total,
            percent=overall,
            details=event,
        )
        self._emit_progress(progress)

    def _install_progress_handler(self) -> None:
        """Install the progress handler for the pipeline."""
        from core import progress
        progress.set_handler(self._progress_handler)

    def _remove_progress_handler(self) -> None:
        """Remove the progress handler."""
        from core import progress
        progress.set_handler(None)

    def cancel(self) -> None:
        """Request cancellation of the current pipeline run."""
        with self._lock:
            self._cancelled = True
            if self._current_video_id:
                from core import cancel
                cancel.request_cancel(self._current_video_id)

    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        with self._lock:
            return self._cancelled

    def process_url(
        self,
        url: str,
        force: bool = False,
        vertical: bool = False,
        profile: Optional[str] = None,
    ) -> PipelineResult:
        """
        Process a YouTube/Twitch/Kick URL through the full pipeline.

        This invokes the ORIGINAL Clips Kitty pipeline:
        core.pipeline.process_video()

        Args:
            url: Video URL (YouTube, Twitch, Kick, or local:video_id)
            force: Re-process even if already done
            vertical: Request vertical version (for Vertical Live)
            profile: Longform rendering profile

        Returns:
            PipelineResult with clips and metadata
        """
        self._start_time = time.monotonic()
        self._completed_stages.clear()
        self._cancelled = False
        self._install_progress_handler()

        result = PipelineResult(success=False)

        try:
            # Setup configuration
            config = self._setup_config()
            pipeline_config = get_pipeline_config(config)
            db = self._get_db()

            # Extract video_id for cancellation tracking
            from sources.dispatch import identify
            source, video_id = identify(url)
            self._current_video_id = video_id

            # Initial progress
            self._emit_progress(PipelineProgress(
                stage="download",
                message=f"Starting pipeline for {url}",
                video_id=video_id,
                percent=0,
            ))

            # Run the ORIGINAL pipeline
            clips = process_video(
                url=url,
                config=pipeline_config,
                db=db,
                force=force,
            )

            # Final progress
            self._emit_progress(PipelineProgress(
                stage="done",
                message=f"Completed: {len(clips)} clip(s) created",
                video_id=video_id,
                percent=100,
            ))

            result.success = True
            result.clips = clips
            result.video_id = video_id
            if clips:
                result.video_title = clips[0].candidate.hook
            result.processing_time_seconds = time.monotonic() - self._start_time
            result.stages_completed = list(self._completed_stages)

        except Exception as e:
            error_msg = str(e)
            self._emit_progress(PipelineProgress(
                stage="error",
                message=f"Pipeline failed: {error_msg}",
                video_id=self._current_video_id or "",
                error=error_msg,
            ))
            result.error = error_msg
            result.processing_time_seconds = time.monotonic() - self._start_time
            result.stages_completed = list(self._completed_stages)

        finally:
            self._remove_progress_handler()
            self._current_video_id = None

        return result

    def process_local_file(
        self,
        file_path: str,
        title: str = "",
        channel: str = "",
        force: bool = False,
    ) -> PipelineResult:
        """
        Process a local video file through the pipeline.

        Uses the original ingestion path for local files via
        sources.dispatch.download() with "local:" prefix.

        Args:
            file_path: Path to local video file
            title: Optional title (extracted from filename if not provided)
            channel: Optional channel name
            force: Re-process even if already done

        Returns:
            PipelineResult with clips and metadata
        """
        from sources.dispatch import download as dispatch_download
        from core.paths import cached_source

        self._start_time = time.monotonic()
        self._completed_stages.clear()
        self._cancelled = False
        self._install_progress_handler()

        result = PipelineResult(success=False)

        try:
            config = self._setup_config()
            pipeline_config = get_pipeline_config(config)
            db = self._get_db()

            # Import local file to downloads directory
            source_path = Path(file_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Video file not found: {file_path}")

            # Generate a video_id from filename
            video_id = f"local_{source_path.stem}"

            # Copy to downloads directory (original pipeline expects it there)
            downloads_dir = self.data_dir / "downloads"
            downloads_dir.mkdir(parents=True, exist_ok=True)
            target_path = downloads_dir / source_path.name

            import shutil
            if not target_path.exists():
                shutil.copy2(source_path, target_path)

            # Create local URL for pipeline
            local_url = f"local:{video_id}"
            self._current_video_id = video_id

            self._emit_progress(PipelineProgress(
                stage="download",
                message=f"Processing local file: {source_path.name}",
                video_id=video_id,
                percent=0,
            ))

            # Run the ORIGINAL pipeline (handles local: URLs specially)
            clips = process_video(
                url=local_url,
                config=pipeline_config,
                db=db,
                force=force,
            )

            self._emit_progress(PipelineProgress(
                stage="done",
                message=f"Completed: {len(clips)} clip(s) created",
                video_id=video_id,
                percent=100,
            ))

            result.success = True
            result.clips = clips
            result.video_id = video_id
            result.video_title = title or source_path.stem
            result.processing_time_seconds = time.monotonic() - self._start_time
            result.stages_completed = list(self._completed_stages)

        except Exception as e:
            error_msg = str(e)
            self._emit_progress(PipelineProgress(
                stage="error",
                message=f"Pipeline failed: {error_msg}",
                video_id=self._current_video_id or "",
                error=error_msg,
            ))
            result.error = error_msg
            result.processing_time_seconds = time.monotonic() - self._start_time
            result.stages_completed = list(self._completed_stages)

        finally:
            self._remove_progress_handler()
            self._current_video_id = None

        return result

    def get_clips_for_video(self, video_id: str) -> list[RenderedClip]:
        """Get all rendered clips for a video from the database."""
        db = self._get_db()
        clips = []
        for row in db.clips_for_video(video_id):
            # Reconstruct RenderedClip from database row
            from core.models import ClipCandidate
            candidate = ClipCandidate(
                start=row["start_s"],
                end=row["end_s"],
                score=row["score"],
                hook=row["hook"] or "",
            )
            clips.append(RenderedClip(
                source_video_id=video_id,
                candidate=candidate,
                path=Path(row["path"]),
            ))
        return clips

    def get_clip_paths(self, video_id: str) -> list[Path]:
        """Get file paths for all clips of a video."""
        clips = self.get_clips_for_video(video_id)
        return [clip.path for clip in clips if clip.path.exists()]

    def export_clips_to_working(self, video_id: str, output_dir: Optional[str] = None) -> list[Path]:
        """
        Copy clips to Kaggle working directory for download.

        Args:
            video_id: Video ID to export clips for
            output_dir: Target directory (default: /kaggle/working/clips)

        Returns:
            List of copied clip paths
        """
        if output_dir is None:
            output_dir = "/kaggle/working/clips"

        target_dir = Path(output_dir) / video_id
        target_dir.mkdir(parents=True, exist_ok=True)

        clip_paths = self.get_clip_paths(video_id)
        copied = []

        for clip_path in clip_paths:
            target = target_dir / clip_path.name
            import shutil
            shutil.copy2(clip_path, target)
            copied.append(target)

        return copied

    def close(self) -> None:
        """Clean up resources."""
        if self._db:
            self._db.close()
            self._db = None


def run_pipeline(
    url: str,
    data_dir: str = "/data",
    use_cloud_llm: bool = True,
    force_cpu: bool = False,
    force: bool = False,
    progress_callback: Optional[Callable[[PipelineProgress], None]] = None,
    verbose: bool = True,
) -> PipelineResult:
    """
    Convenience function to run the pipeline with minimal setup.

    Example:
        result = run_pipeline("https://youtu.be/...")
        if result.success:
            print(f"Created {len(result.clips)} clips")
            adapter.export_clips_to_working(result.video_id)

    This uses the ORIGINAL Clips Kitty pipeline via core.pipeline.process_video()
    """
    adapter = PipelineAdapter(
        data_dir=data_dir,
        use_cloud_llm=use_cloud_llm,
        force_cpu=force_cpu,
        progress_callback=progress_callback,
        verbose=verbose,
    )
    try:
        return adapter.process_url(url, force=force)
    finally:
        adapter.close()
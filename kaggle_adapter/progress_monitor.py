"""
Kaggle Progress Monitor
=======================

Provides rich progress monitoring for Clips Kitty pipeline runs
in Kaggle notebooks. Supports tqdm, IPython display, and custom callbacks.

This module does NOT modify the original pipeline — it only observes
progress events emitted by the adapter.
"""

from __future__ import annotations

import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional
from collections import deque

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from IPython.display import display, HTML, clear_output
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False


@dataclass
class StageProgress:
    """Progress for a single pipeline stage."""
    name: str
    display_name: str
    weight: float
    completed: bool = False
    current: int = 0
    total: int = 0
    message: str = ""
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        if self.end_time > 0:
            return self.end_time - self.start_time
        if self.start_time > 0:
            return time.monotonic() - self.start_time
        return 0.0

    @property
    def percent(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(100.0, (self.current / self.total) * 100)


# Pipeline stages with weights (matching PipelineAdapter)
PIPELINE_STAGES = [
    StageProgress("download", "Downloading", 0.15),
    StageProgress("transcribe", "Transcribing", 0.35),
    StageProgress("analyze", "Analyzing", 0.20),
    StageProgress("render", "Rendering", 0.30),
]

# Stage names that indicate completion (emitted by PipelineAdapter._progress_handler)
# PipelineAdapter emits: "downloaded", "transcribed", "analyzed", "done"
# but the stage keys are: "download", "transcribe", "analyze", "render"
STAGE_COMPLETION_MAP = {
    "downloaded": "download",
    "transcribed": "transcribe",
    "analyzed": "analyze",
    "done": "render",
}


class ProgressMonitor:
    """
    Rich progress monitor for Kaggle notebooks.

    Features:
    - tqdm progress bars (if available)
    - IPython display updates (if in notebook)
    - Console fallback
    - Stage-by-stage timing
    - ETA estimation
    - Callback support for custom UIs
    """

    def __init__(
        self,
        use_tqdm: bool = True,
        use_ipython: bool = True,
        callback: Optional[Callable[[dict], None]] = None,
        refresh_rate: float = 0.5,
    ):
        """
        Initialize the progress monitor.

        Args:
            use_tqdm: Use tqdm for progress bars
            use_ipython: Use IPython display for notebook updates
            callback: Optional custom callback for progress events
            refresh_rate: Minimum seconds between updates
        """
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE
        self.use_ipython = use_ipython and IPYTHON_AVAILABLE
        self.callback = callback
        self.refresh_rate = refresh_rate

        self._stages = {s.name: s for s in PIPELINE_STAGES}
        self._stage_order = [s.name for s in PIPELINE_STAGES]
        self._current_stage_idx = 0
        self._start_time = 0.0
        self._last_update = 0.0
        self._video_id = ""
        self._total_clips = 0
        self._completed_clips = 0
        self._lock = threading.Lock()

        # tqdm bars
        self._overall_bar: Optional[tqdm] = None
        self._stage_bar: Optional[tqdm] = None

        # History for ETA
        self._stage_durations: dict[str, float] = {}

    def handle_progress(self, progress) -> None:
        """Handle a progress event from the pipeline adapter."""
        # Import PipelineProgress here to avoid circular import
        from kaggle_adapter.pipeline_adapter import PipelineProgress

        if not isinstance(progress, PipelineProgress):
            # Try to convert dict to PipelineProgress
            if isinstance(progress, dict):
                progress = PipelineProgress(**progress)
            else:
                return

        with self._lock:
            now = time.monotonic()

            # Rate limiting
            if now - self._last_update < self.refresh_rate:
                return
            self._last_update = now

            # Track video ID
            if progress.video_id:
                self._video_id = progress.video_id

            # Handle stage transitions
            stage = progress.stage

            # Check if this stage marks completion of a tracked stage
            completed_stage_key = STAGE_COMPLETION_MAP.get(stage)
            if completed_stage_key and completed_stage_key in self._stages:
                stage_obj = self._stages[completed_stage_key]
                if not stage_obj.completed:
                    stage_obj.completed = True
                    stage_obj.end_time = now
                    self._stage_durations[completed_stage_key] = stage_obj.duration
                    self._current_stage_idx = min(
                        self._current_stage_idx + 1,
                        len(self._stage_order) - 1
                    )

            # Update current stage progress (if it's a tracked stage)
            if stage in self._stages:
                stage_obj = self._stages[stage]
                if stage_obj.start_time == 0:
                    stage_obj.start_time = now
                if progress.total > 0:
                    stage_obj.current = progress.current
                    stage_obj.total = progress.total
                stage_obj.message = progress.message

            # Update overall progress
            self._update_display(progress)

            # Call custom callback
            if self.callback:
                try:
                    self.callback({
                        "stage": stage,
                        "percent": progress.percent,
                        "message": progress.message,
                        "video_id": progress.video_id,
                        "elapsed": progress.elapsed_seconds,
                        "current_clip": progress.current,
                        "total_clips": progress.total,
                    })
                except Exception:
                    pass

    def _update_display(self, progress) -> None:
        """Update all active displays."""
        if self.use_tqdm:
            self._update_tqdm(progress)
        if self.use_ipython:
            self._update_ipython(progress)
        else:
            self._update_console(progress)

    def _update_tqdm(self, progress) -> None:
        """Update tqdm progress bars."""
        # Initialize bars on first call
        if self._overall_bar is None:
            self._overall_bar = tqdm(
                total=100,
                desc="Overall",
                position=0,
                leave=True,
                bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}% [{elapsed}<{remaining}]",
            )

        if self._stage_bar is None:
            self._stage_bar = tqdm(
                total=100,
                desc=f"Stage: {progress.stage}",
                position=1,
                leave=False,
                bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}% [{elapsed}<{remaining}]",
            )

        # Update overall bar
        self._overall_bar.n = progress.percent
        self._overall_bar.refresh()

        # Update stage bar
        stage_obj = self._stages.get(progress.stage)
        if stage_obj and stage_obj.total > 0:
            self._stage_bar.total = stage_obj.total
            self._stage_bar.n = stage_obj.current
            self._stage_bar.set_description(f"Stage: {progress.stage} - {progress.message[:40]}")
            self._stage_bar.refresh()
        elif stage_obj:
            self._stage_bar.n = progress.percent
            self._stage_bar.set_description(f"Stage: {progress.stage}")
            self._stage_bar.refresh()

        # Close bars on completion
        if progress.stage == "done" or progress.error:
            if self._stage_bar:
                self._stage_bar.close()
                self._stage_bar = None
            if self._overall_bar:
                self._overall_bar.close()
                self._overall_bar = None

    def _update_ipython(self, progress) -> None:
        """Update IPython display with HTML progress."""
        # Build HTML progress display
        html = self._build_html(progress)
        clear_output(wait=True)
        display(HTML(html))

    def _update_console(self, progress) -> None:
        """Update console output (fallback)."""
        bar = self._console_bar(progress.percent)
        stage_info = f" [{progress.current}/{progress.total}]" if progress.total > 0 else ""
        msg = f"\r{bar} {progress.stage}: {progress.message}{stage_info}"
        sys.stdout.write(msg)
        sys.stdout.flush()

        if progress.stage == "done" or progress.error:
            sys.stdout.write("\n")

    def _console_bar(self, percent: float, width: int = 40) -> str:
        """Create console progress bar."""
        filled = int(width * percent / 100)
        return "[" + "=" * filled + " " * (width - filled) + f"] {percent:5.1f}%"

    def _build_html(self, progress) -> str:
        """Build HTML progress display for IPython."""
        elapsed = progress.elapsed_seconds
        elapsed_str = self._format_time(elapsed)

        # Overall progress
        overall_pct = progress.percent
        overall_bar = self._html_bar(overall_pct, "overall")

        # Stage progress bars
        stage_bars = ""
        for stage_name in self._stage_order:
            stage = self._stages[stage_name]
            if stage.completed:
                pct = 100
                status = "✓"
                color = "#4CAF50"
            elif stage_name == progress.stage:
                pct = stage.percent if stage.total > 0 else overall_pct
                status = "⟳"
                color = "#2196F3"
            else:
                pct = 0
                status = "○"
                color = "#9E9E9E"

            stage_bars += f"""
            <div style="margin: 8px 0;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="width: 24px; text-align: center; color: {color}; font-weight: bold;">{status}</span>
                    <span style="width: 120px;">{stage.display_name}</span>
                    <div style="flex: 1; background: #e0e0e0; border-radius: 4px; height: 20px; overflow: hidden;">
                        <div style="width: {pct}%; background: {color}; height: 100%; transition: width 0.3s;"></div>
                    </div>
                    <span style="width: 50px; text-align: right;">{pct:.0f}%</span>
                </div>
            </div>
            """

        # Current status
        current_msg = progress.message
        if progress.error:
            current_msg = f"❌ Error: {progress.error}"
            status_color = "#f44336"
        elif progress.stage == "done":
            current_msg = f"✅ Completed: {progress.total} clip(s) created"
            status_color = "#4CAF50"
        else:
            status_color = "#2196F3"

        # Clip counter
        clip_info = ""
        if progress.total > 0:
            clip_info = f"<br><strong>Clips:</strong> {progress.current} / {progress.total}"

        return f"""
        <div style="font-family: monospace; padding: 16px; background: #fafafa; border-radius: 8px; border: 1px solid #e0e0e0;">
            <div style="margin-bottom: 16px;">
                <strong>Video ID:</strong> {progress.video_id or '—'}<br>
                <strong>Elapsed:</strong> {elapsed_str}<br>
                <strong>Overall Progress:</strong>
                <div style="background: #e0e0e0; border-radius: 4px; height: 24px; margin-top: 4px; overflow: hidden;">
                    <div style="width: {overall_pct}%; background: {status_color}; height: 100%; transition: width 0.3s;"></div>
                </div>
                <span style="float: right;">{overall_pct:.1f}%</span>
            </div>
            <div style="margin-bottom: 16px;">
                <strong>Stages:</strong>
                {stage_bars}
            </div>
            <div style="padding: 12px; background: white; border-radius: 4px; border-left: 4px solid {status_color};">
                {current_msg}{clip_info}
            </div>
        </div>
        """

    def _html_bar(self, percent: float, label: str) -> str:
        """Create HTML progress bar."""
        return f"""
        <div style="margin: 8px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span>{label.capitalize()}</span>
                <span>{percent:.1f}%</span>
            </div>
            <div style="background: #e0e0e0; border-radius: 4px; height: 20px; overflow: hidden;">
                <div style="width: {percent}%; background: #2196F3; height: 100%; transition: width 0.3s;"></div>
            </div>
        </div>
        """

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    def start(self) -> None:
        """Mark monitoring as started."""
        self._start_time = time.monotonic()
        for stage in self._stages.values():
            stage.start_time = 0
            stage.end_time = 0
            stage.completed = False
            stage.current = 0
            stage.total = 0
            stage.message = ""
        self._current_stage_idx = 0

    def finish(self, success: bool = True, message: str = "") -> None:
        """Mark monitoring as finished."""
        # Close tqdm bars
        if self._stage_bar:
            self._stage_bar.close()
            self._stage_bar = None
        if self._overall_bar:
            self._overall_bar.close()
            self._overall_bar = None

        # Final IPython display
        if self.use_ipython and IPYTHON_AVAILABLE:
            from kaggle_adapter.pipeline_adapter import PipelineProgress
            final_progress = PipelineProgress(
                stage="done" if success else "error",
                message=message or ("Completed" if success else "Failed"),
                video_id=self._video_id,
                current=self._completed_clips,
                total=self._total_clips,
                percent=100 if success else 0,
                elapsed_seconds=time.monotonic() - self._start_time,
            )
            self._update_ipython(final_progress)

    def get_summary(self) -> dict:
        """Get a summary of the run."""
        return {
            "video_id": self._video_id,
            "total_time": time.monotonic() - self._start_time,
            "stages": {
                name: {
                    "completed": s.completed,
                    "duration": s.duration,
                    "message": s.message,
                }
                for name, s in self._stages.items()
            },
            "stage_durations": self._stage_durations,
        }


def create_progress_callback(monitor: ProgressMonitor) -> Callable:
    """Create a callback function compatible with PipelineAdapter."""
    def callback(progress):
        monitor.handle_progress(progress)
    return callback


# Convenience function for quick notebook usage
def monitor_pipeline(
    adapter,
    url: str,
    **kwargs
):
    """
    Run pipeline with progress monitoring.

    Example:
        from kaggle_adapter.pipeline_adapter import PipelineAdapter, run_pipeline
        from kaggle_adapter.progress_monitor import ProgressMonitor, create_progress_callback

        monitor = ProgressMonitor()
        adapter = PipelineAdapter(progress_callback=create_progress_callback(monitor))
        result = adapter.process_url("https://youtu.be/...")
        monitor.finish(result.success)
    """
    monitor = ProgressMonitor(**kwargs)
    monitor.start()

    callback = create_progress_callback(monitor)

    # If adapter not provided, create one
    if not hasattr(adapter, 'process_url'):
        from kaggle_adapter.pipeline_adapter import PipelineAdapter
        adapter = PipelineAdapter(progress_callback=callback, **kwargs)

    try:
        result = adapter.process_url(url)
        monitor.finish(result.success, result.error or "")
        return result, monitor.get_summary()
    except Exception as e:
        monitor.finish(False, str(e))
        raise
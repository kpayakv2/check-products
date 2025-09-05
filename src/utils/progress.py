"""
Progress reporting implementations.

This module provides implementations of the ProgressReporter interface
for different progress tracking and reporting strategies.
"""

import time
import sys
from typing import Optional, Dict, Any, Callable
from abc import ABC, abstractmethod

from ..core.interfaces import ProgressReporter


class ConsoleProgressReporter(ProgressReporter):
    """Console-based progress reporter with text output."""
    
    def __init__(self, 
                 show_percentage: bool = True,
                 show_eta: bool = True,
                 show_rate: bool = True,
                 update_interval: float = 0.1):
        """
        Initialize console progress reporter.
        
        Args:
            show_percentage: Whether to show percentage completed
            show_eta: Whether to show estimated time remaining
            show_rate: Whether to show processing rate
            update_interval: Minimum seconds between updates
        """
        self.show_percentage = show_percentage
        self.show_eta = show_eta
        self.show_rate = show_rate
        self.update_interval = update_interval
        
        self._start_time: Optional[float] = None
        self._last_update: float = 0
        self._last_progress: int = 0
    
    def start(self, total: int, description: str = "Processing") -> None:
        """Start progress tracking."""
        self.total = total
        self.description = description
        self.current = 0
        self._start_time = time.time()
        self._last_update = self._start_time
        self._last_progress = 0
        
        print(f"{self.description}... 0/{total}")
    
    def update(self, progress: int, message: Optional[str] = None) -> None:
        """Update progress."""
        self.current = progress
        current_time = time.time()
        
        # Throttle updates
        if current_time - self._last_update < self.update_interval and progress < self.total:
            return
        
        self._last_update = current_time
        
        # Build progress message
        parts = []
        
        if self.show_percentage and self.total > 0:
            percentage = (progress / self.total) * 100
            parts.append(f"{percentage:.1f}%")
        
        parts.append(f"{progress}/{self.total}")
        
        if self.show_rate and self._start_time:
            elapsed = current_time - self._start_time
            if elapsed > 0:
                rate = progress / elapsed
                parts.append(f"{rate:.1f}/s")
        
        if self.show_eta and self._start_time and progress > 0:
            elapsed = current_time - self._start_time
            if elapsed > 0 and progress < self.total:
                eta_seconds = (elapsed / progress) * (self.total - progress)
                eta_str = self._format_time(eta_seconds)
                parts.append(f"ETA: {eta_str}")
        
        progress_str = " | ".join(parts)
        
        if message:
            progress_str += f" | {message}"
        
        # Clear line and print progress
        print(f"\r{self.description}... {progress_str}", end="", flush=True)
        
        if progress >= self.total:
            print()  # New line when complete
    
    def finish(self, message: Optional[str] = None) -> None:
        """Finish progress tracking."""
        if self._start_time:
            elapsed = time.time() - self._start_time
            elapsed_str = self._format_time(elapsed)
            
            final_message = f"{self.description} completed in {elapsed_str}"
            if message:
                final_message += f" | {message}"
            
            print(f"\r{final_message}")
    
    def _format_time(self, seconds: float) -> str:
        """Format time duration as human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def get_config(self) -> Dict[str, Any]:
        """Get reporter configuration."""
        return {
            "type": "ConsoleProgressReporter",
            "show_percentage": self.show_percentage,
            "show_eta": self.show_eta,
            "show_rate": self.show_rate,
            "update_interval": self.update_interval
        }


class SilentProgressReporter(ProgressReporter):
    """Silent progress reporter that doesn't output anything."""
    
    def __init__(self):
        """Initialize silent progress reporter."""
        pass
    
    def start(self, total: int, description: str = "Processing") -> None:
        """Start progress tracking (silent)."""
        self.total = total
        self.current = 0
    
    def update(self, progress: int, message: Optional[str] = None) -> None:
        """Update progress (silent)."""
        self.current = progress
    
    def finish(self, message: Optional[str] = None) -> None:
        """Finish progress tracking (silent)."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get reporter configuration."""
        return {"type": "SilentProgressReporter"}


class CallbackProgressReporter(ProgressReporter):
    """Progress reporter that calls user-provided callbacks."""
    
    def __init__(self, 
                 start_callback: Optional[Callable[[int, str], None]] = None,
                 update_callback: Optional[Callable[[int, Optional[str]], None]] = None,
                 finish_callback: Optional[Callable[[Optional[str]], None]] = None):
        """
        Initialize callback progress reporter.
        
        Args:
            start_callback: Callback for start events
            update_callback: Callback for update events
            finish_callback: Callback for finish events
        """
        self.start_callback = start_callback
        self.update_callback = update_callback
        self.finish_callback = finish_callback
        
        self._start_time: Optional[float] = None
    
    def start(self, total: int, description: str = "Processing") -> None:
        """Start progress tracking."""
        self.total = total
        self.current = 0
        self.description = description
        self._start_time = time.time()
        
        if self.start_callback:
            self.start_callback(total, description)
    
    def update(self, progress: int, message: Optional[str] = None) -> None:
        """Update progress."""
        self.current = progress
        
        if self.update_callback:
            self.update_callback(progress, message)
    
    def finish(self, message: Optional[str] = None) -> None:
        """Finish progress tracking."""
        if self.finish_callback:
            self.finish_callback(message)
    
    def get_config(self) -> Dict[str, Any]:
        """Get reporter configuration."""
        return {
            "type": "CallbackProgressReporter",
            "has_start_callback": self.start_callback is not None,
            "has_update_callback": self.update_callback is not None,
            "has_finish_callback": self.finish_callback is not None
        }


class MultiProgressReporter(ProgressReporter):
    """Progress reporter that forwards to multiple other reporters."""
    
    def __init__(self, reporters: list[ProgressReporter]):
        """
        Initialize multi progress reporter.
        
        Args:
            reporters: List of progress reporters to forward to
        """
        self.reporters = reporters
    
    def start(self, total: int, description: str = "Processing") -> None:
        """Start progress tracking on all reporters."""
        self.total = total
        self.current = 0
        
        for reporter in self.reporters:
            try:
                reporter.start(total, description)
            except Exception:
                pass  # Don't let one reporter break others
    
    def update(self, progress: int, message: Optional[str] = None) -> None:
        """Update progress on all reporters."""
        self.current = progress
        
        for reporter in self.reporters:
            try:
                reporter.update(progress, message)
            except Exception:
                pass  # Don't let one reporter break others
    
    def finish(self, message: Optional[str] = None) -> None:
        """Finish progress tracking on all reporters."""
        for reporter in self.reporters:
            try:
                reporter.finish(message)
            except Exception:
                pass  # Don't let one reporter break others
    
    def get_config(self) -> Dict[str, Any]:
        """Get reporter configuration."""
        return {
            "type": "MultiProgressReporter",
            "reporters": [reporter.get_config() for reporter in self.reporters]
        }


class TQDMProgressReporter(ProgressReporter):
    """Progress reporter using tqdm library (if available)."""
    
    def __init__(self, **tqdm_kwargs):
        """
        Initialize tqdm progress reporter.
        
        Args:
            **tqdm_kwargs: Additional arguments for tqdm
        """
        self.tqdm_kwargs = tqdm_kwargs
        self._tqdm = None
        
        # Check if tqdm is available
        try:
            import tqdm
            self._tqdm_available = True
            self._tqdm_module = tqdm
        except ImportError:
            self._tqdm_available = False
            self._fallback = ConsoleProgressReporter()
    
    def start(self, total: int, description: str = "Processing") -> None:
        """Start progress tracking."""
        self.total = total
        self.current = 0
        
        if self._tqdm_available:
            self._tqdm = self._tqdm_module.tqdm(
                total=total,
                desc=description,
                **self.tqdm_kwargs
            )
        else:
            self._fallback.start(total, description)
    
    def update(self, progress: int, message: Optional[str] = None) -> None:
        """Update progress."""
        if self._tqdm_available and self._tqdm:
            # Update tqdm with the difference
            delta = progress - self.current
            if delta > 0:
                self._tqdm.update(delta)
            
            if message:
                self._tqdm.set_postfix_str(message)
        else:
            self._fallback.update(progress, message)
        
        self.current = progress
    
    def finish(self, message: Optional[str] = None) -> None:
        """Finish progress tracking."""
        if self._tqdm_available and self._tqdm:
            if message:
                self._tqdm.set_postfix_str(message)
            self._tqdm.close()
        else:
            self._fallback.finish(message)
    
    def get_config(self) -> Dict[str, Any]:
        """Get reporter configuration."""
        return {
            "type": "TQDMProgressReporter",
            "tqdm_available": self._tqdm_available,
            "tqdm_kwargs": self.tqdm_kwargs
        }


def create_progress_reporter(reporter_type: str, **kwargs) -> ProgressReporter:
    """
    Factory function to create progress reporters.
    
    Args:
        reporter_type: Type of reporter ('console', 'silent', 'callback', 'multi', 'tqdm')
        **kwargs: Additional arguments for the reporter
        
    Returns:
        Configured progress reporter instance
    """
    if reporter_type.lower() == "console":
        return ConsoleProgressReporter(**kwargs)
    elif reporter_type.lower() == "silent":
        return SilentProgressReporter(**kwargs)
    elif reporter_type.lower() == "callback":
        return CallbackProgressReporter(**kwargs)
    elif reporter_type.lower() == "multi":
        return MultiProgressReporter(**kwargs)
    elif reporter_type.lower() == "tqdm":
        return TQDMProgressReporter(**kwargs)
    else:
        raise ValueError(f"Unknown progress reporter type: {reporter_type}")


def create_default_progress_reporter(verbose: bool = True) -> ProgressReporter:
    """
    Create a default progress reporter.
    
    Args:
        verbose: Whether to show progress or use silent reporter
        
    Returns:
        Progress reporter instance
    """
    if verbose:
        # Try tqdm first, fall back to console
        try:
            return TQDMProgressReporter(
                unit="items",
                unit_scale=True,
                dynamic_ncols=True
            )
        except:
            return ConsoleProgressReporter(
                show_percentage=True,
                show_eta=True,
                show_rate=True
            )
    else:
        return SilentProgressReporter()

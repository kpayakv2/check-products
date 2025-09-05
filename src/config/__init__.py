"""Configuration module - minimal version."""

# Import only what we need for basic functionality
from .settings import Settings, load_settings

__all__ = ["Settings", "load_settings"]

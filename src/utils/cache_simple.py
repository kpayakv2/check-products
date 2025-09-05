"""
Simplified caching implementations.
"""

import pickle
import json
import hashlib
import time
from typing import Any, Dict, Optional, Union
from pathlib import Path
import threading


class SimpleCacheManager:
    """Simple in-memory cache manager."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of cached items
        """
        self.max_size = max_size
        self._cache = {}
        self._access_times = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                return self._cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        with self._lock:
            # If cache is full, remove least recently used item
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            self._cache[key] = value
            self._access_times[key] = time.time()
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        del self._cache[lru_key]
        del self._access_times[lru_key]
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    def contains(self, key: str) -> bool:
        """Check if key is in cache."""
        return key in self._cache


class FileCacheManager:
    """File-based cache manager."""
    
    def __init__(self, cache_dir: Path, max_age_seconds: int = 3600):
        """
        Initialize file cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            max_age_seconds: Maximum age of cached files in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.max_age_seconds = max_age_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        # Check if cache is expired
        if time.time() - cache_path.stat().st_mtime > self.max_age_seconds:
            cache_path.unlink()
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            # If we can't load the cache, remove it
            cache_path.unlink()
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in file cache."""
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            raise ValueError(f"Failed to cache value: {e}")
    
    def clear(self) -> None:
        """Clear all cached files."""
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()
    
    def contains(self, key: str) -> bool:
        """Check if key is in cache."""
        return self.get(key) is not None


def create_cache_key(*args, **kwargs) -> str:
    """Create a cache key from arguments."""
    key_data = {
        'args': args,
        'kwargs': kwargs
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()

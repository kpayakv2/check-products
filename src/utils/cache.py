"""
Caching implementations.

This module provides implementations of the CacheManager interface
for different caching strategies.
"""

import pickle
import json
import hashlib
import time
from typing import Any, Dict, Optional, Union
from pathlib import Path
import sqlite3
import threading

from ..core.interfaces import CacheManager


class InMemoryCacheManager(CacheManager):
    """Simple in-memory cache manager."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[int] = None):
        """
        Initialize in-memory cache.
        
        Args:
            max_size: Maximum number of items to cache
            ttl_seconds: Time-to-live for cache entries (None for no expiration)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        
        return time.time() - entry.get('timestamp', 0) > self.ttl_seconds
    
    def _evict_if_needed(self) -> None:
        """Evict old entries if cache is full."""
        if len(self._cache) >= self.max_size:
            # Remove oldest entry (LRU-style)
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].get('timestamp', 0)
            )
            del self._cache[oldest_key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            if self._is_expired(entry):
                del self._cache[key]
                return None
            
            # Update access time for LRU
            entry['timestamp'] = time.time()
            return entry['value']
    
    def put(self, key: str, value: Any) -> None:
        """Put item into cache."""
        with self._lock:
            self._evict_if_needed()
            
            self._cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
    
    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            # Remove expired entries
            current_time = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if self._is_expired(entry)
            ]
            for key in expired_keys:
                del self._cache[key]
            
            return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "type": "InMemoryCache",
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "memory_usage": f"{sum(len(str(entry)) for entry in self._cache.values())} chars"
            }


class FileCacheManager(CacheManager):
    """File-based cache manager using pickle."""
    
    def __init__(self, 
                 cache_dir: Union[str, Path],
                 max_files: int = 10000,
                 ttl_seconds: Optional[int] = None):
        """
        Initialize file cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_files: Maximum number of cache files
            ttl_seconds: Time-to-live for cache entries
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_files = max_files
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use hash to create valid filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _is_expired(self, file_path: Path) -> bool:
        """Check if cache file is expired."""
        if self.ttl_seconds is None:
            return False
        
        if not file_path.exists():
            return True
        
        return time.time() - file_path.stat().st_mtime > self.ttl_seconds
    
    def _evict_if_needed(self) -> None:
        """Evict old files if cache is full."""
        cache_files = list(self.cache_dir.glob("*.cache"))
        
        if len(cache_files) >= self.max_files:
            # Remove oldest files
            cache_files.sort(key=lambda f: f.stat().st_mtime)
            to_remove = len(cache_files) - self.max_files + 1
            
            for file_path in cache_files[:to_remove]:
                try:
                    file_path.unlink()
                except OSError:
                    pass  # File might have been removed by another process
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        file_path = self._get_file_path(key)
        
        if not file_path.exists() or self._is_expired(file_path):
            return None
        
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except (OSError, pickle.PickleError):
            # Remove corrupted file
            try:
                file_path.unlink()
            except OSError:
                pass
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item into cache."""
        with self._lock:
            self._evict_if_needed()
            
            file_path = self._get_file_path(key)
            
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
            except (OSError, pickle.PickleError) as e:
                raise RuntimeError(f"Failed to cache item: {e}")
    
    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        file_path = self._get_file_path(key)
        
        if file_path.exists():
            try:
                file_path.unlink()
                return True
            except OSError:
                pass
        
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        for file_path in self.cache_dir.glob("*.cache"):
            try:
                file_path.unlink()
            except OSError:
                pass
    
    def size(self) -> int:
        """Get current cache size."""
        return len(list(self.cache_dir.glob("*.cache")))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.cache"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "type": "FileCache",
            "size": len(cache_files),
            "max_files": self.max_files,
            "ttl_seconds": self.ttl_seconds,
            "cache_dir": str(self.cache_dir),
            "disk_usage_bytes": total_size,
            "disk_usage_mb": round(total_size / (1024 * 1024), 2)
        }


class SQLiteCacheManager(CacheManager):
    """SQLite-based cache manager."""
    
    def __init__(self, 
                 db_path: Union[str, Path],
                 max_entries: int = 100000,
                 ttl_seconds: Optional[int] = None):
        """
        Initialize SQLite cache.
        
        Args:
            db_path: Path to SQLite database file
            max_entries: Maximum number of cache entries
            ttl_seconds: Time-to-live for cache entries
        """
        self.db_path = Path(db_path)
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        
        # Create database and table
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database and create table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    timestamp REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON cache(timestamp)")
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        
        return time.time() - timestamp > self.ttl_seconds
    
    def _cleanup_expired(self, conn: sqlite3.Connection) -> None:
        """Remove expired entries."""
        if self.ttl_seconds is not None:
            cutoff_time = time.time() - self.ttl_seconds
            conn.execute("DELETE FROM cache WHERE timestamp < ?", (cutoff_time,))
    
    def _evict_if_needed(self, conn: sqlite3.Connection) -> None:
        """Evict old entries if cache is full."""
        # Clean up expired entries first
        self._cleanup_expired(conn)
        
        # Check if we still need to evict
        cursor = conn.execute("SELECT COUNT(*) FROM cache")
        count = cursor.fetchone()[0]
        
        if count >= self.max_entries:
            # Remove oldest entries
            to_remove = count - self.max_entries + 1
            conn.execute("""
                DELETE FROM cache 
                WHERE key IN (
                    SELECT key FROM cache 
                    ORDER BY timestamp ASC 
                    LIMIT ?
                )
            """, (to_remove,))
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT value, timestamp FROM cache WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if row is None:
                    return None
                
                value_blob, timestamp = row
                
                if self._is_expired(timestamp):
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    return None
                
                # Update timestamp for LRU
                conn.execute(
                    "UPDATE cache SET timestamp = ? WHERE key = ?",
                    (time.time(), key)
                )
                
                return pickle.loads(value_blob)
                
        except (sqlite3.Error, pickle.PickleError):
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item into cache."""
        try:
            value_blob = pickle.dumps(value)
            timestamp = time.time()
            
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    self._evict_if_needed(conn)
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO cache (key, value, timestamp)
                        VALUES (?, ?, ?)
                    """, (key, value_blob, timestamp))
                    
        except (sqlite3.Error, pickle.PickleError) as e:
            raise RuntimeError(f"Failed to cache item: {e}")
    
    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                return cursor.rowcount > 0
        except sqlite3.Error:
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache")
        except sqlite3.Error:
            pass
    
    def size(self) -> int:
        """Get current cache size."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Clean up expired entries first
                self._cleanup_expired(conn)
                
                cursor = conn.execute("SELECT COUNT(*) FROM cache")
                return cursor.fetchone()[0]
        except sqlite3.Error:
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                self._cleanup_expired(conn)
                
                cursor = conn.execute("SELECT COUNT(*) FROM cache")
                count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT SUM(LENGTH(value)) FROM cache")
                total_size = cursor.fetchone()[0] or 0
                
                return {
                    "type": "SQLiteCache",
                    "size": count,
                    "max_entries": self.max_entries,
                    "ttl_seconds": self.ttl_seconds,
                    "db_path": str(self.db_path),
                    "db_size_bytes": total_size,
                    "db_size_mb": round(total_size / (1024 * 1024), 2)
                }
        except sqlite3.Error:
            return {
                "type": "SQLiteCache",
                "error": "Failed to get statistics"
            }


def create_cache_manager(cache_type: str, **kwargs) -> CacheManager:
    """
    Factory function to create cache managers.
    
    Args:
        cache_type: Type of cache ('memory', 'file', 'sqlite')
        **kwargs: Additional arguments for the cache manager
        
    Returns:
        Configured cache manager instance
    """
    if cache_type.lower() == "memory":
        return InMemoryCacheManager(**kwargs)
    elif cache_type.lower() == "file":
        return FileCacheManager(**kwargs)
    elif cache_type.lower() == "sqlite":
        return SQLiteCacheManager(**kwargs)
    else:
        raise ValueError(f"Unknown cache manager type: {cache_type}")


def create_default_cache_manager(cache_dir: Union[str, Path]) -> CacheManager:
    """
    Create a default cache manager optimized for embedding caching.
    
    Args:
        cache_dir: Directory for cache storage
        
    Returns:
        File-based cache manager with sensible defaults
    """
    return FileCacheManager(
        cache_dir=cache_dir,
        max_files=50000,  # Allow many cached embeddings
        ttl_seconds=7 * 24 * 3600  # 1 week TTL
    )

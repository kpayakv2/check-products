#!/usr/bin/env python3
"""
Global Model Cache Manager
==========================
จัดการ model caching และ memory management สำหรับ web server
แก้ปัญหา memory leaks และปรับปรุงประสิทธิภาพ
"""

import threading
import time
from typing import Dict, Optional, Any, Tuple
import gc
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

# Optional psutil import
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ModelCacheEntry:
    """Entry สำหรับ model cache"""
    model: Any
    config: Dict[str, Any]
    created_at: datetime
    last_used: datetime
    usage_count: int
    memory_size_mb: float


class GlobalModelCache:
    """
    Global model cache ที่จัดการ memory อย่างมีประสิทธิภาพ
    Singleton pattern เพื่อให้ทั้ง application ใช้ cache เดียวกัน
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._models: Dict[str, ModelCacheEntry] = {}
        self._cache_lock = threading.RLock()
        self._max_memory_mb = 2048  # 2GB limit
        self._cleanup_interval = 300  # 5 minutes
        self._max_unused_time = 1800  # 30 minutes
        self._initialized = True
        
        # เริ่ม background cleanup thread
        self._start_cleanup_thread()
        
        logger.info("🗄️ Global Model Cache initialized")
    
    def _start_cleanup_thread(self):
        """เริ่ม background thread สำหรับ cleanup"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self._cleanup_interval)
                    self._cleanup_unused_models()
                except Exception as e:
                    logger.error(f"Cleanup thread error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("🧹 Cleanup thread started")
    
    def _get_cache_key(self, model_type: str, config: Dict[str, Any]) -> str:
        """สร้าง cache key จาก model type และ config"""
        # สร้าง key ที่ unique จาก config
        config_str = "_".join([f"{k}:{v}" for k, v in sorted(config.items())])
        return f"{model_type}_{hash(config_str)}"
    
    def _get_memory_usage_mb(self) -> float:
        """ดูการใช้ memory ของ process ปัจจุบัน"""
        if not PSUTIL_AVAILABLE:
            return 0.0
            
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _estimate_model_size_mb(self, model: Any) -> float:
        """ประเมินขนาด memory ของ model (rough estimate)"""
        try:
            if hasattr(model, 'get_model_info'):
                info = model.get_model_info()
                # ประมาณการจาก model dimension
                if 'dimension' in info:
                    dim = info['dimension']
                    # SentenceTransformer โดยประมาณ
                    if dim == 384:  # MiniLM
                        return 120
                    elif dim == 768:  # MPNET
                        return 280
            return 100  # default estimate
        except Exception:
            return 100
    
    def get_model(self, model_type: str, config: Dict[str, Any]) -> Optional[Any]:
        """ดึง model จาก cache หรือสร้างใหม่"""
        cache_key = self._get_cache_key(model_type, config)
        
        with self._cache_lock:
            # ตรวจสอบใน cache
            if cache_key in self._models:
                entry = self._models[cache_key]
                entry.last_used = datetime.now()
                entry.usage_count += 1
                
                logger.info(f"📋 Model cache HIT: {model_type} (used {entry.usage_count} times)")
                return entry.model
            
            # ไม่มีใน cache - ต้องสร้างใหม่
            logger.info(f"📋 Model cache MISS: {model_type} - creating new model")
            return None
    
    def store_model(self, model: Any, model_type: str, config: Dict[str, Any]) -> bool:
        """เก็บ model ใน cache"""
        cache_key = self._get_cache_key(model_type, config)
        
        with self._cache_lock:
            # ตรวจสอบ memory limit
            current_memory = self._get_memory_usage_mb()
            model_size = self._estimate_model_size_mb(model)
            
            if current_memory + model_size > self._max_memory_mb:
                logger.warning(f"⚠️ Memory limit would be exceeded. Current: {current_memory:.1f}MB, Model: {model_size:.1f}MB")
                self._force_cleanup()
                
                # ตรวจสอบอีกครั้งหลัง cleanup
                current_memory = self._get_memory_usage_mb()
                if current_memory + model_size > self._max_memory_mb:
                    logger.error(f"❌ Cannot cache model - memory limit exceeded")
                    return False
            
            # เก็บ model ใน cache
            entry = ModelCacheEntry(
                model=model,
                config=config.copy(),
                created_at=datetime.now(),
                last_used=datetime.now(),
                usage_count=1,
                memory_size_mb=model_size
            )
            
            self._models[cache_key] = entry
            
            logger.info(f"💾 Model cached: {model_type} ({model_size:.1f}MB)")
            logger.info(f"📊 Cache status: {len(self._models)} models, {current_memory:.1f}MB total")
            
            return True
    
    def _cleanup_unused_models(self):
        """ทำความสะอาด models ที่ไม่ได้ใช้งาน"""
        with self._cache_lock:
            if not self._models:
                return
            
            now = datetime.now()
            to_remove = []
            
            for cache_key, entry in self._models.items():
                unused_time = (now - entry.last_used).total_seconds()
                if unused_time > self._max_unused_time:
                    to_remove.append((cache_key, unused_time))
            
            if to_remove:
                for cache_key, unused_time in to_remove:
                    entry = self._models[cache_key]
                    del self._models[cache_key]
                    del entry.model  # Explicit deletion
                    
                    logger.info(f"🧹 Cleaned up unused model: {cache_key} (unused for {unused_time/60:.1f} min)")
                
                # Force garbage collection
                gc.collect()
                
                current_memory = self._get_memory_usage_mb()
                logger.info(f"🧹 Cleanup complete: {len(to_remove)} models removed, {current_memory:.1f}MB memory")
    
    def _force_cleanup(self):
        """บังคับ cleanup models เก่า"""
        with self._cache_lock:
            if not self._models:
                return
            
            # เรียงตาม last_used (เก่าที่สุดก่อน)
            sorted_models = sorted(
                self._models.items(),
                key=lambda x: x[1].last_used
            )
            
            # ลบ 50% ของ models เก่า
            remove_count = max(1, len(sorted_models) // 2)
            
            for i in range(remove_count):
                cache_key, entry = sorted_models[i]
                del self._models[cache_key]
                del entry.model
                
                logger.info(f"🧹 Force cleanup: removed {cache_key}")
            
            gc.collect()
            
            current_memory = self._get_memory_usage_mb()
            logger.info(f"🧹 Force cleanup complete: {remove_count} models removed, {current_memory:.1f}MB memory")
    
    def get_stats(self) -> Dict[str, Any]:
        """ดูสถิติของ cache"""
        with self._cache_lock:
            total_memory = sum(entry.memory_size_mb for entry in self._models.values())
            current_memory = self._get_memory_usage_mb()
            
            return {
                "cached_models": len(self._models),
                "total_cache_memory_mb": total_memory,
                "process_memory_mb": current_memory,
                "memory_limit_mb": self._max_memory_mb,
                "memory_utilization": (current_memory / self._max_memory_mb) * 100,
                "models": [
                    {
                        "key": key,
                        "created_at": entry.created_at.isoformat(),
                        "last_used": entry.last_used.isoformat(),
                        "usage_count": entry.usage_count,
                        "memory_mb": entry.memory_size_mb
                    }
                    for key, entry in self._models.items()
                ]
            }
    
    def clear_cache(self):
        """ล้าง cache ทั้งหมด"""
        with self._cache_lock:
            count = len(self._models)
            
            for entry in self._models.values():
                del entry.model
            
            self._models.clear()
            gc.collect()
            
            logger.info(f"🧹 Cache cleared: {count} models removed")
    
    def set_memory_limit_mb(self, limit_mb: int):
        """ตั้งค่า memory limit"""
        self._max_memory_mb = limit_mb
        logger.info(f"📊 Memory limit set to {limit_mb}MB")


# Global instance
_global_cache = None

def get_global_cache() -> GlobalModelCache:
    """ดึง global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = GlobalModelCache()
    return _global_cache


def create_cached_model(model_type: str, config: Dict[str, Any]) -> Optional[Any]:
    """
    สร้าง model โดยใช้ cache
    
    Args:
        model_type: ประเภท model ('sentence-bert', 'tfidf', etc.)
        config: การตั้งค่า model
        
    Returns:
        Model instance หรือ None ถ้าสร้างไม่ได้
    """
    cache = get_global_cache()
    
    # ลองดึงจาก cache ก่อน
    model = cache.get_model(model_type, config)
    if model is not None:
        return model
    
    # ไม่มีใน cache - สร้างใหม่
    try:
        if model_type == 'sentence-bert':
            from advanced_models import get_offline_ready_model
            model = get_offline_ready_model("multilingual")
            
        elif model_type == 'tfidf':
            from advanced_models import create_optimized_tfidf
            model = create_optimized_tfidf()
            
        elif model_type == 'optimized-tfidf':
            from advanced_models import create_optimized_tfidf
            model = create_optimized_tfidf(
                max_features=config.get('max_features', 10000),
                dimension=config.get('dimension', 1000)
            )
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
        
        if model is not None:
            # เก็บใน cache
            cache.store_model(model, model_type, config)
            return model
            
    except Exception as e:
        logger.error(f"Error creating model {model_type}: {e}")
        return None
    
    return None


def clear_model_cache():
    """ล้าง model cache ทั้งหมด"""
    cache = get_global_cache()
    cache.clear_cache()


def get_cache_stats() -> Dict[str, Any]:
    """ดูสถิติของ model cache"""
    cache = get_global_cache()
    return cache.get_stats()
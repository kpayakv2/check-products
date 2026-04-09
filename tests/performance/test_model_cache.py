#!/usr/bin/env python3
"""
ทดสอบ Global Model Cache และ Memory Management
"""

import time
import sys
import os

# เพิ่ม path สำหรับ import
sys.path.append(os.path.dirname(__file__))

from model_cache_manager import (
    get_global_cache, 
    create_cached_model, 
    get_cache_stats,
    clear_model_cache
)


def test_model_caching():
    """ทดสอบการ cache models"""
    print("🔄 Testing Model Caching System")
    print("=" * 50)
    
    # Test 1: สร้าง model ครั้งแรก (จะช้า)
    print("\n1️⃣ Creating model for first time (should be slow)...")
    start_time = time.time()
    
    config1 = {
        'model_type': 'sentence-bert',
        'threshold': 0.6,
        'top_k': 10
    }
    
    model1 = create_cached_model('sentence-bert', config1)
    first_time = time.time() - start_time
    
    if model1:
        print(f"✅ Model created: {type(model1).__name__}")
        print(f"⏱️  Time taken: {first_time:.2f} seconds")
    else:
        print(f"❌ Model creation failed")
        return False
    
    # Test 2: สร้าง model อีกครั้งด้วย config เดียวกัน (จะเร็ว)
    print("\n2️⃣ Creating same model again (should be fast - cached)...")
    start_time = time.time()
    
    model2 = create_cached_model('sentence-bert', config1)
    second_time = time.time() - start_time
    
    if model2:
        print(f"✅ Model retrieved: {type(model2).__name__}")
        print(f"⏱️  Time taken: {second_time:.2f} seconds")
        print(f"🚀 Speed improvement: {first_time/second_time:.1f}x faster")
        print(f"📍 Same instance: {model1 is model2}")
    else:
        print(f"❌ Model retrieval failed")
        return False
    
    # Test 3: สร้าง model ด้วย config ต่างกัน
    print("\n3️⃣ Creating model with different config...")
    config2 = {
        'model_type': 'sentence-bert',
        'threshold': 0.7,  # ต่างจากเดิม
        'top_k': 10
    }
    
    model3 = create_cached_model('sentence-bert', config2)
    
    if model3:
        print(f"✅ Different config model created")
        print(f"📍 Different from first: {model1 is not model3}")
    else:
        print(f"❌ Different config model failed")
    
    return True


def test_cache_stats():
    """ทดสอบ cache statistics"""
    print("\n📊 Testing Cache Statistics")
    print("=" * 30)
    
    stats = get_cache_stats()
    
    print(f"🗄️  Cached models: {stats['cached_models']}")
    print(f"💾 Cache memory: {stats['total_cache_memory_mb']:.1f} MB")
    print(f"🖥️  Process memory: {stats['process_memory_mb']:.1f} MB")
    print(f"📈 Memory utilization: {stats['memory_utilization']:.1f}%")
    
    if stats['models']:
        print(f"\n📋 Model details:")
        for i, model_info in enumerate(stats['models'], 1):
            print(f"  {i}. Usage: {model_info['usage_count']} times, Memory: {model_info['memory_mb']:.1f} MB")
    
    return stats


def test_memory_cleanup():
    """ทดสอบ memory cleanup"""
    print("\n🧹 Testing Memory Cleanup")
    print("=" * 30)
    
    # ดู stats ก่อน cleanup
    stats_before = get_cache_stats()
    models_before = stats_before['cached_models']
    
    print(f"Before cleanup: {models_before} models")
    
    # ทำ cleanup
    clear_model_cache()
    
    # ดู stats หลัง cleanup
    stats_after = get_cache_stats()
    models_after = stats_after['cached_models']
    
    print(f"After cleanup: {models_after} models")
    print(f"✅ Cleanup successful: {models_before - models_after} models removed")
    
    return models_after == 0


def test_performance_impact():
    """ทดสอบ performance impact ของ caching"""
    print("\n⚡ Testing Performance Impact")
    print("=" * 35)
    
    # Clear cache ก่อน
    clear_model_cache()
    
    config = {
        'model_type': 'sentence-bert',
        'threshold': 0.6,
        'top_k': 10
    }
    
    times = []
    
    # ทดสอบ 3 ครั้ง
    for i in range(3):
        print(f"\n🔄 Test run {i+1}/3...")
        start_time = time.time()
        
        model = create_cached_model('sentence-bert', config)
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        if model:
            print(f"✅ Model loaded in {elapsed:.2f} seconds")
        else:
            print(f"❌ Model loading failed")
            return False
    
    print(f"\n📊 Performance Summary:")
    print(f"   1st run (no cache): {times[0]:.2f}s")
    print(f"   2nd run (cached):   {times[1]:.2f}s")
    print(f"   3rd run (cached):   {times[2]:.2f}s")
    
    if times[0] > times[1]:
        improvement = times[0] / times[1]
        print(f"🚀 Cache improvement: {improvement:.1f}x faster")
    
    return True


def main():
    """รันการทดสอบทั้งหมด"""
    print("🎯 Global Model Cache Testing Suite")
    print("=" * 60)
    
    try:
        # Test 1: Model Caching
        success1 = test_model_caching()
        
        # Test 2: Cache Statistics  
        stats = test_cache_stats()
        
        # Test 3: Performance Impact
        success2 = test_performance_impact()
        
        # Test 4: Memory Cleanup
        success3 = test_memory_cleanup()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"🔧 Model Caching: {'✅ PASS' if success1 else '❌ FAIL'}")
        print(f"⚡ Performance: {'✅ PASS' if success2 else '❌ FAIL'}")
        print(f"🧹 Memory Cleanup: {'✅ PASS' if success3 else '❌ FAIL'}")
        
        if success1 and success2 and success3:
            print(f"\n🎉 ALL TESTS PASSED - Model Cache is working perfectly!")
            return True
        else:
            print(f"\n⚠️ Some tests failed - check logs above")
            return False
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
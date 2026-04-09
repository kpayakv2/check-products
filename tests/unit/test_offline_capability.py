#!/usr/bin/env python3
"""
ทดสอบความสามารถ Offline ของ SentenceTransformer
"""

from advanced_models import (
    check_offline_models, 
    get_offline_ready_model, 
    ensure_offline_capability,
    create_sentence_transformer
)


def test_offline_models():
    """ทดสอบการตรวจสอบ offline models"""
    print("🔍 Testing Offline Model Detection")
    print("=" * 50)
    
    # 1. ตรวจสอบ models ที่มีใน cache
    available_models = check_offline_models()
    print(f"\n📊 Available offline models: {available_models}")
    
    # 2. ทดสอบความพร้อมใช้งาน offline
    offline_ready = ensure_offline_capability()
    print(f"📱 Offline capability: {'✅ Ready' if offline_ready else '❌ Not ready'}")
    
    return available_models, offline_ready


def test_offline_model_loading():
    """ทดสอบการโหลด model แบบ offline"""
    print("\n🤖 Testing Offline Model Loading")
    print("=" * 50)
    
    try:
        # ทดสอบโหลด model แบบ offline
        model = get_offline_ready_model("multilingual")
        
        if model:
            print(f"✅ Offline model loaded successfully!")
            print(f"📏 Model info: {model.get_model_info()}")
            
            # ทดสอบ encoding
            test_texts = [
                "ลิปสติกสีแดง",
                "รองเท้าผ้าใบสีขาว", 
                "เสื้อยืดผู้ชาย"
            ]
            
            print(f"\n🔄 Testing encoding with {len(test_texts)} texts...")
            embeddings = model.encode(test_texts)
            print(f"✅ Encoding successful! Shape: {embeddings.shape}")
            
            return True, model
        else:
            print(f"❌ No offline model available")
            return False, None
            
    except Exception as e:
        print(f"❌ Error loading offline model: {e}")
        return False, None


def test_force_offline_mode():
    """ทดสอบ force offline mode"""
    print("\n🔌 Testing Force Offline Mode")
    print("=" * 50)
    
    try:
        # สร้าง model และบังคับใช้ offline mode
        model = create_sentence_transformer("multilingual")
        print(f"📍 Model created")
        
        # เปิด offline mode
        model.enable_offline_mode(True)
        print(f"🔌 Offline mode enabled")
        
        # ทดสอบ encoding
        test_texts = ["สินค้าทดสอบ", "test product"]
        embeddings = model.encode(test_texts)
        print(f"✅ Offline encoding successful! Shape: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Force offline mode failed: {e}")
        return False


def main():
    """รันการทดสอบทั้งหมด"""
    print("🎯 SentenceTransformer Offline Capability Test")
    print("=" * 60)
    
    # Test 1: ตรวจสอบ offline models
    available_models, offline_ready = test_offline_models()
    
    if not offline_ready:
        print("\n❌ Cannot proceed - no offline models available")
        print("💡 Run 'python download_models.py' first")
        return
    
    # Test 2: ทดสอบ offline model loading
    model_loaded, model = test_offline_model_loading()
    
    # Test 3: ทดสอบ force offline mode  
    force_offline_success = test_force_offline_mode()
    
    # สรุปผลการทดสอบ
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"🔍 Offline models detected: {'✅' if offline_ready else '❌'}")
    print(f"🤖 Offline model loading: {'✅' if model_loaded else '❌'}")
    print(f"🔌 Force offline mode: {'✅' if force_offline_success else '❌'}")
    
    if offline_ready and model_loaded and force_offline_success:
        print(f"\n🎉 ALL TESTS PASSED - SentenceTransformer can work offline!")
    else:
        print(f"\n⚠️ Some tests failed - offline capability may be limited")


if __name__ == "__main__":
    main()
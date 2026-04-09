#!/usr/bin/env python3
"""
Download and Cache SentenceTransformer Models
============================================

Pre-download SentenceTransformer models to local cache for offline usage.
"""

import os
import sys
from pathlib import Path

def download_model(model_name: str, cache_dir: str = None):
    """Download and cache a SentenceTransformer model."""
    print(f"🔄 Downloading model: {model_name}")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Set cache directory
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Cache directory: {cache_path.absolute()}")
            
            # Download model to specific cache directory
            model = SentenceTransformer(model_name, cache_folder=str(cache_path))
        else:
            # Use default cache directory
            model = SentenceTransformer(model_name)
            cache_path = Path.home() / ".cache" / "huggingface" / "sentence_transformers"
            print(f"📁 Default cache directory: {cache_path}")
        
        # Get model info
        model_info = {
            "model_name": model_name,
            "dimension": model.get_sentence_embedding_dimension(),
            "max_seq_length": getattr(model, 'max_seq_length', 'unknown'),
            "device": str(model.device)
        }
        
        print(f"✅ Model downloaded successfully!")
        print(f"   Dimension: {model_info['dimension']}")
        print(f"   Max sequence length: {model_info['max_seq_length']}")
        print(f"   Device: {model_info['device']}")
        
        # Test encoding
        print(f"🧪 Testing model...")
        test_texts = ["ทดสอบโมเดล", "Test model", "เสื้อยืด Nike"]
        embeddings = model.encode(test_texts, convert_to_numpy=True)
        print(f"   Test embeddings shape: {embeddings.shape}")
        print(f"   Sample embedding (first 5 dims): {embeddings[0][:5]}")
        
        return model, model_info
        
    except ImportError:
        print("❌ Error: sentence-transformers not installed")
        print("   Install with: pip install sentence-transformers")
        return None, None
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None, None

def download_recommended_models():
    """Download recommended models for Thai product similarity."""
    
    # Create local cache directory
    project_dir = Path(__file__).parent
    cache_dir = project_dir / "model_cache"
    
    recommended_models = [
        "paraphrase-multilingual-MiniLM-L12-v2",    # Primary model (118MB)
        "paraphrase-multilingual-mpnet-base-v2",     # Alternative (278MB)  
        "distiluse-base-multilingual-cased"          # Lightweight (135MB)
    ]
    
    print("🚀 Downloading Recommended SentenceTransformer Models")
    print("=" * 60)
    print(f"📁 Local cache directory: {cache_dir.absolute()}")
    
    downloaded_models = []
    
    for i, model_name in enumerate(recommended_models, 1):
        print(f"\n📦 [{i}/{len(recommended_models)}] {model_name}")
        print("-" * 50)
        
        model, info = download_model(model_name, str(cache_dir))
        if model and info:
            downloaded_models.append(info)
            print(f"✅ Successfully cached: {model_name}")
        else:
            print(f"❌ Failed to download: {model_name}")
    
    print(f"\n🎉 Download Summary:")
    print(f"   Successfully downloaded: {len(downloaded_models)}/{len(recommended_models)} models")
    print(f"   Cache location: {cache_dir.absolute()}")
    
    if downloaded_models:
        print(f"\n📊 Model Details:")
        for info in downloaded_models:
            print(f"   - {info['model_name']}: {info['dimension']}D")
    
    return downloaded_models, cache_dir

def test_cached_models(cache_dir: Path):
    """Test that cached models can be loaded quickly."""
    print(f"\n🧪 Testing Cached Models")
    print("=" * 40)
    
    try:
        from sentence_transformers import SentenceTransformer
        import time
        
        # Test primary model
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        
        print(f"Loading {model_name} from cache...")
        start_time = time.time()
        
        model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        
        # Quick test
        test_texts = ["เสื้อยืด Nike", "Nike เสื้อยืด", "กางเกงยีนส์"]
        embeddings = model.encode(test_texts)
        
        print(f"✅ Test encoding successful: {embeddings.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing cached model: {e}")
        return False

def update_config_for_local_cache():
    """Update advanced_models.py to use local cache by default."""
    
    config_update = '''
# Update your fresh_architecture.py Config class:
class Config:
    def __init__(self):
        self.similarity_threshold = 0.6
        self.model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        # Add local cache path
        self.model_cache_dir = str(Path(__file__).parent / "model_cache")
'''
    
    print(f"\n⚙️ Configuration Update Suggestion:")
    print("=" * 45)
    print(config_update)
    
    print(f"\n💡 Usage in your code:")
    print('''
from fresh_implementations import ComponentFactory
from pathlib import Path

# Use local cache
cache_dir = Path(__file__).parent / "model_cache"
model = ComponentFactory.create_embedding_model(
    "sentence-bert", 
    cache_dir=str(cache_dir)
)
''')

if __name__ == "__main__":
    print("🚀 SentenceTransformer Model Downloader")
    print("=" * 50)
    
    # Check if sentence-transformers is installed
    try:
        import sentence_transformers
        print(f"✅ sentence-transformers version: {sentence_transformers.__version__}")
    except ImportError:
        print("❌ sentence-transformers not installed")
        print("   Installing...")
        os.system("pip install sentence-transformers")
    
    # Download models
    models, cache_dir = download_recommended_models()
    
    if models:
        # Test cached models
        success = test_cached_models(cache_dir)
        
        if success:
            print(f"\n🎉 All models successfully cached and tested!")
            print(f"📁 Models are stored in: {cache_dir.absolute()}")
            
            # Show configuration update
            update_config_for_local_cache()
            
            print(f"\n🚀 Your models are ready for offline usage!")
        
    else:
        print(f"\n❌ No models were successfully downloaded.")
        print("   Check your internet connection and try again.")
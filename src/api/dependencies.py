import time
import uuid
import os
import logging
from typing import Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Import our Phase 4 pipeline
from src.core.fresh_architecture import ProductSimilarityPipeline
from src.core.fresh_implementations import ComponentFactory
from src.main import Phase4Config

# Import embedding model
from src.core.advanced_models import SentenceTransformerModel

# Import API config
from src.api.models import APIConfig

try:
    from supabase import create_client, Client
except ImportError:
    print("WARNING: Supabase not installed. Category classification will be unavailable.")
    Client = None

# Global state management
app_state = {
    "start_time": time.time(),
    "jobs": {},  # job_id -> JobStatus
    "config": APIConfig(),
    "pipeline": None,  # Lazy initialization
    "embedding_model": None,  # Lazy initialization for embeddings
    "taxonomy_service": None,
    "supabase_client": None,
    "ml_learning_system": None
}

def initialize_pipeline():
    """Initialize the product similarity pipeline."""
    if app_state["pipeline"] is None:
        print("[PIPELINE] Initializing Phase 4 pipeline...")
        
        # Create configuration
        config = Phase4Config()
        config.model_name = "tfidf"
        config.similarity_method = "cosine"
        config.enable_performance_tracking = True
        config.include_metadata = True
        config.include_confidence_scores = True
        
        # Create components
        data_source = ComponentFactory.create_data_source("csv")
        data_sink = ComponentFactory.create_data_sink("csv")
        text_processor = ComponentFactory.create_text_processor("thai", remove_all_spaces=True)
        embedding_model = ComponentFactory.create_embedding_model("tfidf")
        similarity_calculator = ComponentFactory.create_similarity_calculator("cosine")
        
        # Create matcher
        from src.core.fresh_architecture import ProductMatcher
        matcher = ProductMatcher(
            embedding_model=embedding_model,
            similarity_calculator=similarity_calculator,
            text_processor=text_processor,
            config=config
        )
        
        # Create pipeline
        pipeline = ProductSimilarityPipeline(
            data_source=data_source,
            data_sink=data_sink,
            product_matcher=matcher
        )
        
        app_state["pipeline"] = pipeline
        print("[PIPELINE] Pipeline initialized successfully!")
    
    return app_state["pipeline"]


def get_supabase_client() -> Optional[Client]:
    """Initialize and return the Supabase Client."""
    if "supabase_client" not in app_state or app_state["supabase_client"] is None:
        try:
            # โหลด .env.local ถ้ายังไม่ได้โหลด
            try:
                from dotenv import load_dotenv
                from pathlib import Path
                env_path = Path(__file__).parent.parent.parent / "taxonomy-app/.env.local"
                if env_path.exists():
                    load_dotenv(env_path, override=False)
            except ImportError:
                pass

            # ลำดับความสำคัญ: SUPABASE_URL (absolute) > NEXT_PUBLIC_SUPABASE_URL
            url = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
            key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")

            # ถ้า URL เป็น relative path ให้ใช้ local URL แทน
            if url and not url.startswith("http"):
                url = "http://127.0.0.1:54331"
                logger.warning(f"⚠️ SUPABASE_URL เป็น relative path ใช้ fallback: {url}")

            if not url or not key:
                logger.warning("⚠️ Supabase credentials missing.")
                return None

            app_state["supabase_client"] = create_client(url, key)
            logger.info(f"✅ Supabase connected: {url}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase: {e}", exc_info=True)
            return None

    return app_state["supabase_client"]

def get_taxonomy_service() -> Optional[Any]:
    """Initialize and return the TaxonomyService."""
    if "taxonomy_service" not in app_state or app_state["taxonomy_service"] is None:
        try:
            supabase = get_supabase_client()
            if not supabase:
                return None
                
            from src.services.taxonomy_service import TaxonomyService
            app_state["taxonomy_service"] = TaxonomyService(supabase)
            logger.info("✅ TaxonomyService initialized!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize TaxonomyService: {e}", exc_info=True)
            return None
            
    return app_state["taxonomy_service"]

def get_ml_learning_system() -> Optional[Any]:
    """Initialize and return the ContinuousLearningSystem."""
    if "ml_learning_system" not in app_state or app_state["ml_learning_system"] is None:
        try:
            supabase = get_supabase_client()
            if not supabase:
                logger.warning("⚠️ Cannot initialize ContinuousLearningSystem: Supabase client is None")
                return None
            from src.services.ml_feedback_learning import ContinuousLearningSystem
            app_state["ml_learning_system"] = ContinuousLearningSystem(supabase)
            logger.info("✅ ContinuousLearningSystem initialized!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ContinuousLearningSystem: {e}", exc_info=True)
            return None
    return app_state["ml_learning_system"]


def create_job_id() -> str:
    """Create a unique job ID."""
    return str(uuid.uuid4())


def initialize_embedding_model():
    """Initialize the sentence transformer model (paraphrase-multilingual-MiniLM-L12-v2)."""
    if app_state["embedding_model"] is None:
        print("[MODEL] Loading Sentence Transformer model (384-dim)...")
        try:
            model = SentenceTransformerModel(
                model_name="paraphrase-multilingual-MiniLM-L12-v2"
            )
            app_state["embedding_model"] = model
            print(f"[MODEL] Model loaded! Dimension: {model.get_dimension()}")
        except Exception as e:
            print(f"[MODEL] Failed to load model: {e}")
            raise
    
    return app_state["embedding_model"]

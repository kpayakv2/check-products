#!/usr/bin/env python3
"""
Phase 4 Performance Optimization - Updated Main Production
=========================================================

Enhanced main.py with Phase 4 advanced features that actually work.
"""

import sys
import argparse
import time
import json
import pandas as pd
import traceback
from pathlib import Path
from typing import List, Dict, Any
from types import SimpleNamespace

# Import fresh architecture
from src.core.fresh_architecture import ProductMatcher, ProductSimilarityPipeline, Config
from src.core.fresh_implementations import ComponentFactory

# Import advanced features
from src.core.advanced_models import ensure_offline_capability, check_offline_models
from src.core.model_cache_manager import create_cached_model, get_cache_stats
from src.utils.product_data_utils import load_products_from_file
from src.core.scoring_logic import enhance_matches

# =============================================================================
# PHASE 4 CONFIGURATION
# =============================================================================

class Phase4Config(Config):
    """Configuration extension for Phase 4."""
    def __init__(self):
        super().__init__()
        self.enable_performance_tracking = True
        self.export_detailed_report = True
        self.include_confidence_scores = True
        self.include_metadata = True
        self.model_cache_dir = "model_cache"
        self.output_dir = "output"
        self.similarity_threshold = 0.6
        self.top_k = 10
        self.model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        self.similarity_method = "cosine"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_directories(config: Phase4Config):
    """Ensure necessary directories exist."""
    Path(config.model_cache_dir).mkdir(exist_ok=True)
    Path(config.output_dir).mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

def generate_performance_report(start_time, end_time, matches, config, args):
    """Create a detailed performance summary."""
    duration = end_time - start_time
    report = {
        "summary": {
            "total_matches": len(matches),
            "execution_time_sec": round(duration, 2),
            "matches_per_second": round(len(matches) / duration, 2) if duration > 0 else 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "configuration": {
            "model": args.model,
            "similarity": args.similarity,
            "threshold": config.similarity_threshold
        },
        "stats": {
            "very_high_confidence": len([m for m in matches if m.get('confidence_level') == 'very_high']),
            "high_confidence": len([m for m in matches if m.get('confidence_level') == 'high']),
            "medium_confidence": len([m for m in matches if m.get('confidence_level') == 'medium']),
            "low_confidence": len([m for m in matches if m.get('confidence_level') == 'low'])
        }
    }
    return report

def create_enhanced_pipeline(args) -> tuple:
    """Create the similarity pipeline with the requested components."""
    config = Phase4Config()
    config.similarity_threshold = args.threshold
    config.top_k = args.top_k
    config.model_name = args.model
    config.similarity_method = args.similarity

    # Setup core components
    data_source = ComponentFactory.create_data_source("csv")
    data_sink = ComponentFactory.create_data_sink("csv")
    text_processor = ComponentFactory.create_text_processor("thai")
    
    # Use caching for embedding model
    embedding_model = create_cached_model(args.model, {
        "model_name": args.model,
        "cache_dir": config.model_cache_dir
    })
    
    similarity_calculator = ComponentFactory.create_similarity_calculator(args.similarity)
    
    # Build matcher
    matcher = ProductMatcher(
        embedding_model=embedding_model,
        similarity_calculator=similarity_calculator,
        text_processor=text_processor,
        config=config
    )
    
    # Build pipeline
    pipeline = ProductSimilarityPipeline(
        data_source=data_source,
        data_sink=data_sink,
        product_matcher=matcher
    )
    
    return pipeline, config

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 4 Product Similarity Optimizer')
    parser.add_argument('--query', '-q', required=True, help='Query products file')
    parser.add_argument('--ref', '-r', required=True, help='Reference products file')
    parser.add_argument('--model', '-m', default='tfidf', choices=['tfidf', 'sentence-bert', 'optimized-tfidf', 'mock'], help='Embedding model')
    parser.add_argument('--similarity', '-s', default='cosine', choices=['cosine', 'dot_product'], help='Similarity method')
    parser.add_argument('--threshold', '-t', type=float, default=0.6, help='Similarity threshold')
    parser.add_argument('--top-k', '-k', type=int, default=10, help='Top K results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("🚀 Phase 4 Optimizer - Starting Pipeline")
    print("=" * 50)
    
    try:
        # 1. Initialize
        pipeline, config = create_enhanced_pipeline(args)
        setup_directories(config)
        
        # 2. Load Data
        query_products = load_products_from_file(args.query)
        ref_products = load_products_from_file(args.ref)
        
        # 3. Execute Matching
        print(f"🔍 Finding matches (Threshold: {config.similarity_threshold})...")
        start_time = time.time()
        
        matches = pipeline.product_matcher.find_matches(
            query_products=query_products,
            reference_products=ref_products
        )
        
        # 4. Post-process & Enhance
        enhanced_matches = enhance_matches(matches)
        end_time = time.time()
        
        # 5. Reporting
        report = generate_performance_report(start_time, end_time, enhanced_matches, config, args)
        
        print(f"\n✅ Processing Complete in {report['summary']['execution_time_sec']}s")
        print(f"📊 Found {len(enhanced_matches)} matches above threshold")
        
        # 6. Save Results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"output/matches_{args.model}_{timestamp}.csv"
        
        pipeline.data_sink.save(enhanced_matches, output_file)
        
        with open(f"output/report_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
            
        print(f"💾 Results saved to: {output_file}")
        print(f"📄 Report saved to: output/report_{timestamp}.json")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

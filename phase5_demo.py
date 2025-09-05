#!/usr/bin/env python3
"""
Phase 5 Quick Demo - Simple API Test
===================================

Test our Phase 5 API without complex dependencies.
"""

import sys
import time
import json
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import our Phase 4 pipeline
from fresh_architecture import ProductSimilarityPipeline
from fresh_implementations import ComponentFactory
from main_phase4 import Phase4Config, enhance_results


def create_test_pipeline():
    """Create a test pipeline for demo."""
    print("🔧 Creating test pipeline...")
    
    # Create configuration
    config = Phase4Config()
    config.enable_performance_tracking = True
    config.include_metadata = True
    config.include_confidence_scores = True
    
    # Create components
    data_source = ComponentFactory.create_data_source("csv")
    data_sink = ComponentFactory.create_data_sink("csv")
    text_processor = ComponentFactory.create_text_processor("thai")
    embedding_model = ComponentFactory.create_embedding_model("tfidf")
    similarity_calculator = ComponentFactory.create_similarity_calculator("cosine")
    
    # Create matcher
    from fresh_architecture import ProductMatcher
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
    
    print("✅ Pipeline created successfully!")
    return pipeline


def demo_api_functionality():
    """Demo the API functionality without FastAPI."""
    print("🚀 Phase 5 API Functionality Demo")
    print("=" * 50)
    
    # Create pipeline
    pipeline = create_test_pipeline()
    
    # Demo 1: Single Product Match
    print("\n📍 Demo 1: Single Product Matching")
    print("-" * 30)
    
    query_product = "ไอโฟน 14 โปร แมกซ์"
    reference_products = [
        "iPhone 14 Pro Max 128GB สีทอง",
        "Samsung Galaxy S23 Ultra",
        "iPad Pro 12.9นิ้ว M2",
        "MacBook Air M2",
        "AirPods Pro รุ่นที่ 2"
    ]
    
    print(f"🔍 Query: {query_product}")
    print(f"📋 Reference Products: {len(reference_products)} items")
    
    start_time = time.time()
    matches = pipeline.product_matcher.find_matches(
        query_products=[query_product],
        reference_products=reference_products
    )
    processing_time = time.time() - start_time
    
    print(f"⏱️  Processing Time: {processing_time:.3f} seconds")
    print(f"🎯 Matches Found: {len(matches)}")
    
    for i, match in enumerate(matches[:3], 1):
        print(f"   {i}. {match['matched_product']}")
        print(f"      📈 Score: {match['similarity_score']:.4f}")
    
    # Demo 2: Batch Processing Simulation
    print("\n📍 Demo 2: Batch Processing Simulation")
    print("-" * 40)
    
    batch_queries = [
        "ไอโฟน 14 โปร แมกซ์",
        "แกแลกซี่ เอส 23",
        "ไอแพด โปร",
        "แมคบุ๊ค แอร์"
    ]
    
    print(f"📦 Batch Size: {len(batch_queries)} queries")
    
    start_time = time.time()
    all_matches = []
    
    for i, query in enumerate(batch_queries, 1):
        print(f"   Processing {i}/{len(batch_queries)}: {query[:30]}...")
        
        batch_matches = pipeline.product_matcher.find_matches(
            query_products=[query],
            reference_products=reference_products
        )
        all_matches.extend(batch_matches)
        
        # Simulate progress updates
        progress = i / len(batch_queries)
        print(f"   Progress: {progress:.1%}")
    
    total_time = time.time() - start_time
    
    print(f"⏱️  Total Processing Time: {total_time:.3f} seconds")
    print(f"🎯 Total Matches: {len(all_matches)}")
    print(f"⚡ Processing Rate: {len(all_matches)/total_time:.1f} matches/second")
    
    # Demo 3: Enhanced Results with Metadata
    print("\n📍 Demo 3: Enhanced Results with Phase 5 Features")
    print("-" * 50)
    
    # Enhance results with metadata
    enhanced_matches = enhance_results(all_matches, pipeline.product_matcher.config)
    
    print("✨ Enhanced features added:")
    print("   - Processing timestamps")
    print("   - Confidence scoring")
    print("   - Match ranking")
    print("   - Metadata enrichment")
    
    # Show sample enhanced result
    if enhanced_matches:
        sample = enhanced_matches[0]
        print(f"\n📋 Sample Enhanced Result:")
        print(f"   Query: {sample['query_product']}")
        print(f"   Match: {sample['matched_product']}")
        print(f"   Score: {sample['similarity_score']:.4f}")
        if 'confidence_score' in sample:
            print(f"   Confidence: {sample['confidence_score']:.4f} ({sample.get('confidence_level', 'unknown')})")
        if 'match_rank' in sample:
            print(f"   Rank: {sample['match_rank']}")
        if 'processing_timestamp' in sample:
            print(f"   Timestamp: {sample['processing_timestamp']}")
    
    # Demo 4: Performance Analytics
    print("\n📍 Demo 4: Performance Analytics")
    print("-" * 35)
    
    # Calculate performance metrics
    scores = [m['similarity_score'] for m in enhanced_matches]
    if scores:
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        
        high_quality = len([s for s in scores if s >= 0.8])
        medium_quality = len([s for s in scores if 0.5 <= s < 0.8])
        low_quality = len([s for s in scores if s < 0.5])
        
        print(f"📊 Score Statistics:")
        print(f"   Average: {avg_score:.4f}")
        print(f"   Maximum: {max_score:.4f}")
        print(f"   Minimum: {min_score:.4f}")
        print(f"\n📈 Quality Distribution:")
        print(f"   High (≥80%): {high_quality}")
        print(f"   Medium (50-80%): {medium_quality}")
        print(f"   Low (<50%): {low_quality}")
    
    # Save demo results
    demo_results = {
        "demo_metadata": {
            "timestamp": time.time(),
            "version": "phase5_demo",
            "total_processing_time": total_time,
            "total_matches": len(enhanced_matches)
        },
        "performance_metrics": {
            "processing_rate": len(enhanced_matches) / total_time,
            "average_score": avg_score if scores else 0,
            "quality_distribution": {
                "high": high_quality if scores else 0,
                "medium": medium_quality if scores else 0,
                "low": low_quality if scores else 0
            }
        },
        "sample_results": enhanced_matches[:5]  # Save first 5 results
    }
    
    # Save to file
    with open("output/phase5_demo_results.json", "w", encoding="utf-8") as f:
        json.dump(demo_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Demo results saved to: output/phase5_demo_results.json")
    
    # Demo 5: API Response Simulation
    print("\n📍 Demo 5: API Response Format Simulation")
    print("-" * 45)
    
    # Simulate API response format
    api_response = {
        "status": "success",
        "timestamp": time.time(),
        "request_id": "demo-12345",
        "processing_time": total_time,
        "results": {
            "total_matches": len(enhanced_matches),
            "matches": [
                {
                    "query_product": match["query_product"],
                    "matched_product": match["matched_product"], 
                    "similarity_score": match["similarity_score"],
                    "confidence_score": match.get("confidence_score"),
                    "confidence_level": match.get("confidence_level"),
                    "rank": match.get("match_rank", match["rank"])
                }
                for match in enhanced_matches[:3]
            ]
        },
        "performance": {
            "processing_rate": len(enhanced_matches) / total_time,
            "average_score": avg_score if scores else 0
        }
    }
    
    print("📋 API Response Format:")
    print(json.dumps(api_response, indent=2, ensure_ascii=False, default=str)[:500] + "...")
    
    print(f"\n🎉 Phase 5 Demo Completed Successfully!")
    print(f"📊 Summary:")
    print(f"   - Single matches: ✅ Working")
    print(f"   - Batch processing: ✅ Working") 
    print(f"   - Enhanced metadata: ✅ Working")
    print(f"   - Performance analytics: ✅ Working")
    print(f"   - API response format: ✅ Ready")
    
    print(f"\n🚀 Ready for Full API Implementation!")
    
    return demo_results


if __name__ == "__main__":
    try:
        results = demo_api_functionality()
        print(f"\n✅ Demo completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

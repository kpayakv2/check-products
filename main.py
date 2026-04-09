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
from pathlib import Path
from typing import Optional, Dict, Any, List

# Import fresh architecture
from fresh_architecture import ProductMatcher, ProductSimilarityPipeline, Config
from fresh_implementations import ComponentFactory


class Phase4Config(Config):
    """Enhanced configuration for Phase 4."""
    
    def __init__(self):
        super().__init__()
        # Performance tracking
        self.enable_performance_tracking = True
        self.enable_detailed_logging = False
        
        # Enhanced output
        self.include_metadata = True
        self.include_confidence_scores = True
        self.export_performance_report = False
        
        # Human feedback integration
        self.include_human_feedback = False
        self.thai_columns = False


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with Phase 4 enhancements."""
    parser = argparse.ArgumentParser(
        description="Product Similarity Checker - Phase 4 Enhanced Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phase 4 Features:
- Enhanced performance tracking and reporting
- Detailed confidence scoring
- Advanced configuration options
- Production-ready optimizations

Examples:
    python main_phase4.py old.csv new.csv --enhanced
    python main_phase4.py old.csv new.csv --track-performance --include-metadata
    python main_phase4.py old.csv new.csv --confidence-scores --export-report
        """
    )
    
    # Required arguments
    parser.add_argument('old_products_file', help='Old/reference products CSV')
    parser.add_argument('new_products_file', help='New/query products CSV')
    
    # Basic options
    parser.add_argument('--output', '-o', default='output/matched_products_phase4.csv')
    parser.add_argument('--threshold', '-t', type=float, default=0.6)
    parser.add_argument('--top-k', '-k', type=int, default=10)
    parser.add_argument('--model', '-m', choices=['mock', 'tfidf', 'optimized-tfidf', 'sentence-bert', 'sentence-transformer'], default='tfidf')
    parser.add_argument('--similarity', '-s', choices=['cosine', 'dot_product'], default='cosine')
    parser.add_argument('--verbose', '-v', action='store_true')
    
    # Phase 4 enhancements
    parser.add_argument('--enhanced', action='store_true', 
                       help='Enable all Phase 4 enhancements')
    parser.add_argument('--track-performance', action='store_true',
                       help='Enable detailed performance tracking')
    parser.add_argument('--include-metadata', action='store_true',
                       help='Include processing metadata in results')
    parser.add_argument('--confidence-scores', action='store_true',
                       help='Include confidence scores for matches')
    parser.add_argument('--export-report', action='store_true',
                       help='Export detailed performance report')
    parser.add_argument('--report-file', default='output/performance_report.json',
                       help='Performance report output file')
    
    return parser.parse_args()


def create_enhanced_pipeline(args: argparse.Namespace) -> ProductSimilarityPipeline:
    """Create pipeline with Phase 4 enhancements."""
    
    # Create configuration
    config = Phase4Config()
    config.similarity_threshold = args.threshold
    config.top_k = args.top_k
    config.model_name = args.model
    config.similarity_method = args.similarity
    
    # Apply enhancements
    if args.enhanced or args.track_performance:
        config.enable_performance_tracking = True
    if args.enhanced or args.include_metadata:
        config.include_metadata = True
    if args.enhanced or args.confidence_scores:
        config.include_confidence_scores = True
    if args.enhanced or args.export_report:
        config.export_performance_report = True
    
    # Create components
    data_source = ComponentFactory.create_data_source("csv")
    data_sink = ComponentFactory.create_data_sink("csv")
    text_processor = ComponentFactory.create_text_processor("thai")
    embedding_model = ComponentFactory.create_embedding_model(args.model)
    similarity_calculator = ComponentFactory.create_similarity_calculator(args.similarity)
    
    # Create matcher
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
    
    return pipeline, config


def enhance_results(matches: List[Dict], config: Phase4Config, feedback_data: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Enhance results with Phase 4 features.
    
    Args:
        matches: List of similarity match results
        config: Phase4Config with enhancement options
        feedback_data: Optional human feedback data for ML training
        
    Returns:
        Enhanced matches with confidence scores and metadata
    """
    enhanced_matches = []
    
    # Calculate confidence scores
    if config.include_confidence_scores and matches:
        similarity_scores = []
        for match in matches:
            score = match.get('similarity_score', match.get('similarity', 0))
            similarity_scores.append(score)
        
        if similarity_scores:
            max_score = max(similarity_scores)
            min_score = min(similarity_scores)
            score_range = max_score - min_score if max_score > min_score else 1.0
        else:
            max_score = min_score = score_range = 0
    
    for i, match in enumerate(matches):
        enhanced_match = match.copy()
        
        # Add metadata
        if config.include_metadata:
            enhanced_match.update({
                'match_rank': i + 1,
                'processing_timestamp': time.time(),
                'processor_version': 'phase4_enhanced'
            })
        
        # Add confidence score
        if config.include_confidence_scores:
            similarity_score = match.get('similarity_score', match.get('similarity', 0))
            if score_range > 0:
                confidence = (similarity_score - min_score) / score_range
            else:
                confidence = 1.0
            enhanced_match['confidence_score'] = round(confidence, 4)
            
            # Add confidence level
            if confidence >= 0.8:
                enhanced_match['confidence_level'] = 'high'
            elif confidence >= 0.5:
                enhanced_match['confidence_level'] = 'medium'
            else:
                enhanced_match['confidence_level'] = 'low'
        
        # Integrate human feedback if available
        if feedback_data and hasattr(config, 'include_human_feedback') and config.include_human_feedback:
            # Find matching feedback for this product pair
            query_product = match.get('query_product', match.get('new_product', ''))
            matched_product = match.get('matched_product', match.get('old_product', ''))
            
            for feedback in feedback_data:
                if (feedback.get('new_product') == query_product and 
                    feedback.get('old_product') == matched_product):
                    enhanced_match.update({
                        'human_feedback': feedback.get('human_feedback'),
                        'human_comments': feedback.get('comments', ''),
                        'reviewer': feedback.get('reviewer', 'anonymous'),
                        'feedback_timestamp': feedback.get('timestamp')
                    })
                    break
        
        enhanced_matches.append(enhanced_match)
    
    return enhanced_matches


def generate_performance_report(start_time: float, 
                              end_time: float,
                              matches: List[Dict],
                              config: Phase4Config,
                              args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """Generate detailed performance report."""
    
    total_time = end_time - start_time

    model_type = getattr(args, 'model', getattr(config, 'model_name', 'unknown')) if args is not None else getattr(config, 'model_name', 'unknown')
    similarity_method = getattr(args, 'similarity', getattr(config, 'similarity_method', 'unknown')) if args is not None else getattr(config, 'similarity_method', 'unknown')

    report = {
        "execution_summary": {
            "total_execution_time": round(total_time, 3),
            "matches_found": len(matches),
            "processing_rate": round(len(matches) / total_time, 2) if total_time > 0 else 0,
            "timestamp": time.time()
        },
        "configuration": {
            "similarity_threshold": config.similarity_threshold,
            "top_k": config.top_k,
            "model_type": model_type,
            "similarity_method": similarity_method,
            "enhancements_enabled": {
                "performance_tracking": getattr(config, 'enable_performance_tracking', False),
                "metadata_inclusion": getattr(config, 'include_metadata', False),
                "confidence_scoring": getattr(config, 'include_confidence_scores', False)
            }
        },
        "results_analysis": {},
        "recommendations": []
    }
    
    # Analyze results
    if matches:
        scores = [m['similarity_score'] for m in matches]
        report["results_analysis"] = {
            "average_similarity": round(sum(scores) / len(scores), 4),
            "max_similarity": round(max(scores), 4),
            "min_similarity": round(min(scores), 4),
            "score_distribution": {
                "high_quality": len([s for s in scores if s >= 0.8]),
                "medium_quality": len([s for s in scores if 0.5 <= s < 0.8]),
                "low_quality": len([s for s in scores if s < 0.5])
            }
        }
        
        # Add recommendations
        avg_score = report["results_analysis"]["average_similarity"]
        if avg_score < 0.5:
            report["recommendations"].append("Consider lowering similarity threshold for more matches")
        elif avg_score > 0.9:
            report["recommendations"].append("Consider raising similarity threshold for more precise matches")
        
        if total_time > 10:
            report["recommendations"].append("Consider using batch processing for large datasets")
    
    return report


def main() -> int:
    """Enhanced main function with Phase 4 features."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Print header
        print("🚀 Product Similarity Checker - Phase 4 Enhanced")
        print("=" * 60)
        
        if args.verbose:
            print("🔧 Configuration:")
            print(f"   📁 Old products: {args.old_products_file}")
            print(f"   📁 New products: {args.new_products_file}")
            print(f"   🎯 Threshold: {args.threshold}")
            print(f"   🔢 Top-k: {args.top_k}")
            print(f"   🧠 Model: {args.model}")
            print(f"   📐 Similarity: {args.similarity}")
            
            enhancements = []
            if args.enhanced: enhancements.append("full enhancement")
            if args.track_performance: enhancements.append("performance tracking")
            if args.include_metadata: enhancements.append("metadata")
            if args.confidence_scores: enhancements.append("confidence scores")
            if args.export_report: enhancements.append("performance report")
            
            if enhancements:
                print(f"   ✨ Enhancements: {', '.join(enhancements)}")
            print()
        
        # Create enhanced pipeline
        print("🔧 Initializing enhanced pipeline...")
        pipeline, config = create_enhanced_pipeline(args)
        
        # Record start time
        start_time = time.time()
        
        # Run pipeline
        print("🎯 Starting enhanced similarity analysis...")
        results = pipeline.run(
            query_data_source=args.new_products_file,
            reference_data_source=args.old_products_file,
            output_destination=args.output,
            query_column="รายการ",
            reference_column="name"
        )
        
        # Record end time
        end_time = time.time()
        
        # Enhance results if requested
        if results and (config.include_metadata or config.include_confidence_scores):
            print("✨ Enhancing results with Phase 4 features...")
            enhanced_results = enhance_results(results, config)
            
            # Save enhanced results
            import pandas as pd
            df = pd.DataFrame(enhanced_results)
            df.to_csv(args.output, index=False)
            results = enhanced_results
        
        # Generate performance report
        if args.export_report or config.export_performance_report:
            print("📊 Generating performance report...")
            report = generate_performance_report(start_time, end_time, results, config, args)
            
            # Save report
            report_file = Path(args.report_file)
            report_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"📄 Report saved to: {report_file}")
        
        # Display results
        execution_time = end_time - start_time
        print("\n✅ Enhanced Analysis Complete!")
        print("=" * 60)
        print(f"📊 Results: {len(results) if results else 0} matches found")
        print(f"⏱️  Execution time: {execution_time:.2f} seconds")
        print(f"📁 Results saved to: {args.output}")
        
        if args.verbose and results:
            print(f"\n🔍 Top matches:")
            for i, match in enumerate(results[:3], 1):
                print(f"   {i}. {match['query_product'][:40]}...")
                print(f"      → {match['matched_product'][:40]}...")
                print(f"      📈 Score: {match['similarity_score']:.4f}")
                if 'confidence_score' in match:
                    print(f"      🎯 Confidence: {match['confidence_score']:.4f} ({match.get('confidence_level', 'unknown')})")
                print()
        
        print("🎉 Phase 4 enhanced processing completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose if 'args' in locals() else False:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

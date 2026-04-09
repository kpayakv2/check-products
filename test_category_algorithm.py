#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test & Evaluation Pipeline: Category Classification Algorithm
==============================================================

ทดสอบอัลกอริทึมการจัดหมวดหมู่สินค้าด้วย Supabase + Embeddings
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import requests

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Supabase Client
try:
    from supabase import create_client, Client
except ImportError:
    print("❌ Please install supabase: pip install supabase")
    sys.exit(1)

# Embedding Model
try:
    from advanced_models import SentenceTransformerModel
except ImportError:
    print("❌ advanced_models.py not found")
    sys.exit(1)

# Configuration
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL", "http://localhost:54321")
# Use service role key for testing (has full access, bypasses RLS)
SUPABASE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU"
)
API_BASE_URL = "http://localhost:8000"
TEST_CSV = "output/approved_products_for_import_20250914_110653.csv"

# Initialize Supabase Client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(f"✅ Supabase connected: {SUPABASE_URL}")
except Exception as e:
    print(f"❌ Failed to connect to Supabase: {e}")
    sys.exit(1)


class CategoryClassifier:
    """Category Classification Algorithm"""
    
    def __init__(self, use_embeddings: bool = True):
        self.taxonomy_tree = {}
        self.taxonomy_flat = []
        self.keyword_rules = []
        self.synonyms = {}
        self.use_embeddings = use_embeddings
        self.embedding_model = None
        self.category_embeddings = {}
        
        if use_embeddings:
            print("🔧 Loading embedding model...")
            self.embedding_model = SentenceTransformerModel(
                model_name="paraphrase-multilingual-MiniLM-L12-v2"
            )
            print(f"✅ Model loaded! Dimension: {self.embedding_model.get_dimension()}")
    
    def load_taxonomy(self):
        """โหลด Taxonomy จาก Supabase"""
        print("\n📚 Loading Taxonomy from Supabase...")
        
        try:
            response = supabase.table("taxonomy_nodes").select("*").execute()
            nodes = response.data
            
            print(f"   Response type: {type(response)}")
            print(f"   Data type: {type(nodes)}")
            print(f"   Data length: {len(nodes) if nodes else 0}")
            
            if not nodes or len(nodes) == 0:
                print("⚠️ No taxonomy nodes found in database")
                print(f"   Full response: {response}")
                return False
            
            # Build tree structure
            self.taxonomy_flat = nodes
            
            # Group by level
            by_level = defaultdict(list)
            for node in nodes:
                by_level[node['level']].append(node)
            
            print(f"✅ Loaded {len(nodes)} taxonomy nodes")
            print(f"   Levels: {sorted(by_level.keys())}")
            for level in sorted(by_level.keys()):
                print(f"   - Level {level}: {len(by_level[level])} categories")
            
            # Generate embeddings for categories
            if self.use_embeddings:
                self._generate_category_embeddings()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load taxonomy: {e}")
            return False
    
    def _generate_category_embeddings(self):
        """สร้าง Embeddings สำหรับหมวดหมู่"""
        print("\n🔄 Generating category embeddings...")
        
        category_texts = []
        category_ids = []
        
        for node in self.taxonomy_flat:
            # Combine name, keywords, and description
            text_parts = [node['name_th']]
            
            if node.get('name_en'):
                text_parts.append(node['name_en'])
            
            if node.get('keywords'):
                text_parts.extend(node['keywords'])
            
            if node.get('description'):
                text_parts.append(node['description'])
            
            combined_text = " ".join(text_parts)
            category_texts.append(combined_text)
            category_ids.append(node['id'])
        
        # Generate embeddings in batch
        embeddings = self.embedding_model.encode(category_texts)
        
        # Store embeddings
        for cat_id, embedding in zip(category_ids, embeddings):
            self.category_embeddings[cat_id] = embedding
        
        print(f"✅ Generated embeddings for {len(category_ids)} categories")
    
    def load_keyword_rules(self):
        """โหลด Keyword Rules จาก Supabase"""
        print("\n📋 Loading keyword rules...")
        
        try:
            response = supabase.table("keyword_rules").select("*").eq("is_active", True).execute()
            self.keyword_rules = response.data
            print(f"✅ Loaded {len(self.keyword_rules)} keyword rules")
            return True
        except Exception as e:
            print(f"⚠️ Failed to load keyword rules: {e}")
            return False
    
    def load_synonyms(self):
        """โหลด Synonyms จาก Supabase"""
        print("\n📖 Loading synonyms...")
        
        try:
            # Load lemmas with terms
            response = supabase.table("synonym_lemmas")\
                .select("*, synonym_terms(*)")\
                .eq("is_verified", True)\
                .execute()
            
            lemmas = response.data
            
            # Build synonym map
            for lemma in lemmas:
                if lemma.get('synonym_terms'):
                    for term in lemma['synonym_terms']:
                        self.synonyms[term['term']] = lemma['lemma']
            
            print(f"✅ Loaded {len(self.synonyms)} synonyms")
            return True
        except Exception as e:
            print(f"⚠️ Failed to load synonyms: {e}")
            return False
    
    def classify_keyword_based(self, product_name: str, top_k: int = 5) -> List[Dict]:
        """จัดหมวดหมู่ด้วย Keyword Matching"""
        product_lower = product_name.lower()
        matches = {}  # Use dict to deduplicate by category_id
        
        # Check keyword rules (FIX: Add category name lookup)
        for rule in self.keyword_rules:
            keywords = rule.get('keywords', [])
            for keyword in keywords:
                if keyword.lower() in product_lower:
                    # Get category name from taxonomy_flat
                    cat_info = next((n for n in self.taxonomy_flat if n['id'] == rule['category_id']), None)
                    
                    cat_id = rule['category_id']
                    confidence = rule.get('priority', 1) * 0.1  # Normalize to 0-1 range
                    
                    # Keep highest confidence for each category
                    if cat_id not in matches or matches[cat_id]['confidence'] < confidence:
                        match_data = {
                            'category_id': cat_id,
                            'method': 'keyword_rule',
                            'matched_keyword': keyword,
                            'confidence': confidence
                        }
                        
                        # Add category name if found
                        if cat_info:
                            match_data['category_name'] = cat_info['name_th']
                            match_data['category_level'] = cat_info.get('level', 0)
                        else:
                            match_data['category_name'] = f"Unknown (ID: {cat_id[:8]}...)"
                            match_data['category_level'] = 999
                        
                        matches[cat_id] = match_data
        
        # Check taxonomy keywords
        for node in self.taxonomy_flat:
            if node.get('keywords'):
                for keyword in node['keywords']:
                    if keyword.lower() in product_lower:
                        cat_id = node['id']
                        confidence = 0.7
                        
                        if cat_id not in matches or matches[cat_id]['confidence'] < confidence:
                            matches[cat_id] = {
                                'category_id': cat_id,
                                'category_name': node['name_th'],
                                'category_level': node.get('level', 0),
                                'method': 'taxonomy_keyword',
                                'matched_keyword': keyword,
                                'confidence': confidence
                            }
        
        # Check category names (exact match gets highest confidence)
        for node in self.taxonomy_flat:
            if node['name_th'].lower() in product_lower:
                cat_id = node['id']
                matches[cat_id] = {
                    'category_id': cat_id,
                    'category_name': node['name_th'],
                    'category_level': node.get('level', 0),
                    'method': 'name_match',
                    'confidence': 0.95
                }
        
        # Convert to list and sort by confidence (prefer lower level = more specific)
        results = list(matches.values())
        results.sort(key=lambda x: (-x['confidence'], x['category_level']))
        
        return results[:top_k]
    
    def classify_embedding_based(self, product_name: str, top_k: int = 5) -> List[Dict]:
        """จัดหมวดหมู่ด้วย Embedding Similarity"""
        if not self.use_embeddings or not self.category_embeddings:
            return []
        
        # Generate product embedding
        product_embedding = self.embedding_model.encode([product_name])[0]
        
        # Calculate similarities
        similarities = []
        for cat_id, cat_embedding in self.category_embeddings.items():
            similarity = np.dot(product_embedding, cat_embedding) / (
                np.linalg.norm(product_embedding) * np.linalg.norm(cat_embedding)
            )
            
            # Find category info
            cat_info = next((n for n in self.taxonomy_flat if n['id'] == cat_id), None)
            if cat_info:
                similarities.append({
                    'category_id': cat_id,
                    'category_name': cat_info['name_th'],
                    'method': 'embedding',
                    'confidence': float(similarity)
                })
        
        # Sort by confidence and return top_k
        similarities.sort(key=lambda x: x['confidence'], reverse=True)
        return similarities[:top_k]
    
    def classify_hybrid(self, product_name: str, top_k: int = 3) -> List[Dict]:
        """จัดหมวดหมู่ด้วย Hybrid Approach (Keyword + Embedding)"""
        
        # Get keyword matches
        keyword_matches = self.classify_keyword_based(product_name)
        
        # Get embedding matches
        embedding_matches = self.classify_embedding_based(product_name, top_k=10)
        
        # Combine and boost keyword matches
        combined = {}
        
        # Add keyword matches with boost
        for match in keyword_matches:
            cat_id = match['category_id']
            if cat_id not in combined:
                combined[cat_id] = match.copy()
                combined[cat_id]['methods'] = [match['method']]
            else:
                combined[cat_id]['confidence'] = max(
                    combined[cat_id]['confidence'],
                    match['confidence']
                )
                if match['method'] not in combined[cat_id]['methods']:
                    combined[cat_id]['methods'].append(match['method'])
        
        # Add embedding matches
        for match in embedding_matches:
            cat_id = match['category_id']
            if cat_id not in combined:
                combined[cat_id] = match.copy()
                combined[cat_id]['methods'] = [match['method']]
            else:
                # Boost confidence if found by both methods
                combined[cat_id]['confidence'] = (
                    combined[cat_id]['confidence'] * 0.6 +
                    match['confidence'] * 0.4
                )
                if match['method'] not in combined[cat_id]['methods']:
                    combined[cat_id]['methods'].append(match['method'])
        
        # Convert to list and sort
        results = list(combined.values())
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results[:top_k]
    
    def classify(self, product_name: str, method: str = 'hybrid') -> List[Dict]:
        """Main classification method"""
        if method == 'keyword':
            return self.classify_keyword_based(product_name)
        elif method == 'embedding':
            return self.classify_embedding_based(product_name)
        else:
            return self.classify_hybrid(product_name)


def run_evaluation(classifier: CategoryClassifier, test_file: str):
    """Run comprehensive evaluation"""
    
    print("\n" + "=" * 70)
    print("🧪 CATEGORY CLASSIFICATION EVALUATION")
    print("=" * 70)
    
    # Load test data
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    df = pd.read_csv(test_file, encoding='utf-8-sig')
    print(f"\n📄 Test Data: {test_file}")
    print(f"   Products: {len(df)}")
    
    # Test configuration
    test_methods = ['keyword', 'embedding', 'hybrid']
    test_sample_size = min(20, len(df))
    
    results = {
        'keyword': [],
        'embedding': [],
        'hybrid': []
    }
    
    print(f"\n🔬 Testing {test_sample_size} products with 3 methods...")
    print(f"   Methods: {', '.join(test_methods)}")
    
    # Sample products
    test_products = df['product_name'].head(test_sample_size).tolist()
    
    # Test each method
    for method in test_methods:
        print(f"\n📊 Testing method: {method.upper()}")
        method_start = time.time()
        
        for i, product in enumerate(test_products, 1):
            try:
                predictions = classifier.classify(product, method=method)
                results[method].append({
                    'product': product,
                    'predictions': predictions,
                    'top_category': predictions[0] if predictions else None
                })
                
                if i % 5 == 0:
                    print(f"   Progress: {i}/{test_sample_size}")
                    
            except Exception as e:
                print(f"   ❌ Error classifying '{product}': {e}")
                results[method].append({
                    'product': product,
                    'predictions': [],
                    'top_category': None,
                    'error': str(e)
                })
        
        method_time = time.time() - method_start
        print(f"   ✅ Completed in {method_time:.2f}s")
        print(f"   Avg time per product: {method_time/test_sample_size:.3f}s")
    
    # Analysis
    print("\n" + "=" * 70)
    print("📈 EVALUATION RESULTS")
    print("=" * 70)
    
    for method in test_methods:
        print(f"\n🔍 Method: {method.upper()}")
        
        method_results = results[method]
        
        # Count products with predictions
        with_predictions = sum(1 for r in method_results if r.get('top_category'))
        without_predictions = len(method_results) - with_predictions
        
        print(f"   Products with predictions: {with_predictions}/{len(method_results)} ({with_predictions/len(method_results)*100:.1f}%)")
        print(f"   Products without predictions: {without_predictions}")
        
        # Confidence statistics
        confidences = [
            r['top_category']['confidence']
            for r in method_results
            if r.get('top_category')
        ]
        
        if confidences:
            print(f"\n   Confidence Statistics:")
            print(f"   - Mean: {np.mean(confidences):.4f}")
            print(f"   - Median: {np.median(confidences):.4f}")
            print(f"   - Min: {np.min(confidences):.4f}")
            print(f"   - Max: {np.max(confidences):.4f}")
            print(f"   - Std: {np.std(confidences):.4f}")
        
        # Category distribution
        category_counts = defaultdict(int)
        for r in method_results:
            if r.get('top_category'):
                cat_name = r['top_category'].get('category_name', 'Unknown')
                category_counts[cat_name] += 1
        
        if category_counts:
            print(f"\n   Top 5 Predicted Categories:")
            for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   - {cat}: {count} products ({count/len(method_results)*100:.1f}%)")
        
        # Show sample predictions
        print(f"\n   Sample Predictions (first 5):")
        for i, r in enumerate(method_results[:5], 1):
            product = r['product'][:40]
            if r.get('top_category'):
                cat = r['top_category'].get('category_name', 'Unknown')
                conf = r['top_category'].get('confidence', 0.0)
                print(f"   {i}. {product:40} → {cat:20} ({conf:.3f})")
            else:
                print(f"   {i}. {product:40} → NO PREDICTION")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    for method in test_methods:
        output_file = output_dir / f"category_eval_{method}_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results[method], f, ensure_ascii=False, indent=2)
        print(f"\n💾 Saved {method} results: {output_file}")
    
    # Generate comparison report
    comparison_file = output_dir / f"category_eval_comparison_{timestamp}.md"
    generate_comparison_report(results, test_products, comparison_file)
    print(f"💾 Saved comparison report: {comparison_file}")
    
    return results


def generate_comparison_report(results: Dict, test_products: List[str], output_file: Path):
    """Generate markdown comparison report"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Category Classification Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Test Products:** {len(test_products)}\n\n")
        
        f.write("## Methods Compared\n\n")
        f.write("1. **Keyword-based**: Matching keywords and category names\n")
        f.write("2. **Embedding-based**: Semantic similarity using sentence transformers\n")
        f.write("3. **Hybrid**: Combination of keyword and embedding methods\n\n")
        
        f.write("## Summary Statistics\n\n")
        f.write("| Method | Coverage | Avg Confidence | Min Conf | Max Conf |\n")
        f.write("|--------|----------|----------------|----------|----------|\n")
        
        for method in ['keyword', 'embedding', 'hybrid']:
            method_results = results[method]
            with_pred = sum(1 for r in method_results if r.get('top_category'))
            coverage = with_pred / len(method_results) * 100
            
            confidences = [
                r['top_category']['confidence']
                for r in method_results
                if r.get('top_category')
            ]
            
            if confidences:
                f.write(f"| {method.capitalize():8} | {coverage:6.1f}% | {np.mean(confidences):14.4f} | {np.min(confidences):8.4f} | {np.max(confidences):8.4f} |\n")
            else:
                f.write(f"| {method.capitalize():8} | {coverage:6.1f}% | N/A | N/A | N/A |\n")
        
        f.write("\n## Sample Predictions\n\n")
        
        for i, product in enumerate(test_products[:10], 1):
            f.write(f"### {i}. {product}\n\n")
            
            for method in ['keyword', 'embedding', 'hybrid']:
                result = results[method][i-1]
                f.write(f"**{method.capitalize()}:**\n")
                
                if result.get('top_category'):
                    cat = result['top_category']
                    f.write(f"- Category: {cat.get('category_name', 'N/A')}\n")
                    f.write(f"- Confidence: {cat['confidence']:.4f}\n")
                    f.write(f"- Method: {cat.get('method', 'N/A')}\n")
                else:
                    f.write("- No prediction\n")
                
                f.write("\n")
            
            f.write("---\n\n")


def main():
    """Main test function"""
    
    print("=" * 70)
    print("🧪 CATEGORY CLASSIFICATION ALGORITHM TEST")
    print("=" * 70)
    
    # Initialize classifier
    print("\n🔧 Initializing classifier...")
    classifier = CategoryClassifier(use_embeddings=True)
    
    # Load taxonomy
    if not classifier.load_taxonomy():
        print("❌ Failed to load taxonomy")
        return
    
    # Load keyword rules
    classifier.load_keyword_rules()
    
    # Load synonyms
    classifier.load_synonyms()
    
    # Test individual product
    print("\n" + "=" * 70)
    print("🧪 QUICK TEST: Single Product Classification")
    print("=" * 70)
    
    test_product = "กล่องล็อค 560 มล"
    print(f"\nProduct: {test_product}")
    
    for method in ['keyword', 'embedding', 'hybrid']:
        print(f"\n📊 Method: {method.upper()}")
        results = classifier.classify(test_product, method=method)
        
        if results:
            for i, result in enumerate(results[:3], 1):
                cat_name = result.get('category_name', 'N/A')
                confidence = result['confidence']
                method_used = result.get('method', 'N/A')
                print(f"   {i}. {cat_name:30} (conf: {confidence:.4f}, method: {method_used})")
        else:
            print("   No predictions")
    
    # Run full evaluation
    if Path(TEST_CSV).exists():
        run_evaluation(classifier, TEST_CSV)
    else:
        print(f"\n⚠️ Test CSV not found: {TEST_CSV}")
        print("   Skipping full evaluation")
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    main()

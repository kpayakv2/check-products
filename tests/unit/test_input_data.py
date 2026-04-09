#!/usr/bin/env python3
"""
Test SentenceTransformer Model with Real Product Data from Input Folder
=======================================================================

ทดสอบโมเดล SentenceTransformer กับข้อมูลสินค้าจริงจากโฟลเดอร์ input
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# Add project path
sys.path.append(str(Path(__file__).parent))

def load_input_data():
    """โหลดข้อมูลจากโฟลเดอร์ input"""
    print("📂 Loading data from input folder...")
    
    input_dir = Path("input")
    new_product_dir = input_dir / "new_product" 
    old_product_dir = input_dir / "old_product"
    
    data = {}
    
    # Load new products
    print(f"   📁 New products directory: {new_product_dir}")
    for file_path in new_product_dir.glob("*.csv"):
        print(f"      📄 Found: {file_path.name}")
        try:
            df = pd.read_csv(file_path)
            print(f"         Shape: {df.shape}")
            print(f"         Columns: {list(df.columns)}")
            data['new_products_csv'] = df
            break  # Use first CSV found
        except Exception as e:
            print(f"         ❌ Error reading {file_path}: {e}")
    
    # Load old products  
    print(f"   📁 Old products directory: {old_product_dir}")
    for file_path in old_product_dir.glob("*.csv"):
        print(f"      📄 Found: {file_path.name}")
        try:
            df = pd.read_csv(file_path)
            print(f"         Shape: {df.shape}")
            print(f"         Columns: {list(df.columns)}")
            data['old_products_csv'] = df
            break  # Use first CSV found
        except Exception as e:
            print(f"         ❌ Error reading {file_path}: {e}")
    
    return data

def extract_product_names(data):
    """สกัดชื่อสินค้าจากข้อมูล"""
    print("\n🔍 Extracting product names...")
    
    products = {"new": [], "old": []}
    
    # Extract from new products
    if 'new_products_csv' in data:
        df = data['new_products_csv']
        print(f"   🆕 New products CSV columns: {list(df.columns)}")
        
        # Try different possible column names for product names
        product_columns = ['รายการ', 'product_name', 'name', 'Product', 'สินค้า', 'ชื่อสินค้า']
        product_col = None
        
        for col in product_columns:
            if col in df.columns:
                product_col = col
                break
        
        if product_col:
            products['new'] = df[product_col].dropna().astype(str).tolist()[:20]  # First 20
            print(f"      ✅ Extracted {len(products['new'])} new products from '{product_col}' column")
        else:
            print(f"      ❌ No product name column found. Available: {list(df.columns)}")
    
    # Extract from old products
    if 'old_products_csv' in data:
        df = data['old_products_csv'] 
        print(f"   📝 Old products CSV columns: {list(df.columns)}")
        
        product_columns = ['product_name', 'รายการ', 'name', 'Product', 'สินค้า', 'ชื่อสินค้า']
        product_col = None
        
        for col in product_columns:
            if col in df.columns:
                product_col = col
                break
        
        if product_col:
            products['old'] = df[product_col].dropna().astype(str).tolist()[:20]  # First 20
            print(f"      ✅ Extracted {len(products['old'])} old products from '{product_col}' column")
        else:
            print(f"      ❌ No product name column found. Available: {list(df.columns)}")
    
    return products

def test_sentence_transformer_with_cache():
    """ทดสอบ SentenceTransformer จาก cache"""
    print(f"\n🤖 Testing SentenceTransformer from local cache...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        cache_dir = Path("model_cache")
        if not cache_dir.exists():
            print("❌ Model cache not found. Please download model first.")
            return None
        
        print(f"   📁 Cache directory: {cache_dir.absolute()}")
        
        # Load model from cache
        start_time = time.time()
        model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
            cache_folder=str(cache_dir)
        )
        load_time = time.time() - start_time
        
        print(f"   ✅ Model loaded in {load_time:.2f} seconds")
        print(f"   📊 Dimension: {model.get_sentence_embedding_dimension()}")
        
        return model
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return None

def calculate_similarity_matrix(model, new_products, old_products):
    """คำนวณ similarity matrix"""
    print(f"\n🧮 Calculating similarity matrix...")
    
    # Generate embeddings
    print(f"   🔄 Encoding {len(new_products)} new products...")
    start_time = time.time()
    new_embeddings = model.encode(new_products, convert_to_numpy=True, show_progress_bar=True)
    new_time = time.time() - start_time
    print(f"   ✅ New products encoded in {new_time:.2f}s. Shape: {new_embeddings.shape}")
    
    print(f"   🔄 Encoding {len(old_products)} old products...")
    start_time = time.time()
    old_embeddings = model.encode(old_products, convert_to_numpy=True, show_progress_bar=True)
    old_time = time.time() - start_time
    print(f"   ✅ Old products encoded in {old_time:.2f}s. Shape: {old_embeddings.shape}")
    
    # Calculate similarity matrix
    print(f"   🔄 Calculating cosine similarity...")
    start_time = time.time()
    similarity_matrix = np.dot(new_embeddings, old_embeddings.T)  # Cosine similarity (already normalized)
    sim_time = time.time() - start_time
    print(f"   ✅ Similarity calculated in {sim_time:.2f}s. Shape: {similarity_matrix.shape}")
    
    return similarity_matrix, new_embeddings, old_embeddings

def analyze_similarity_results(similarity_matrix, new_products, old_products, threshold=0.6):
    """วิเคราะห์ผลลัพธ์ความคล้าย"""
    print(f"\n📊 Analyzing similarity results (threshold: {threshold})...")
    
    # Find matches above threshold
    matches = []
    for i in range(len(new_products)):
        for j in range(len(old_products)):
            score = similarity_matrix[i][j]
            if score >= threshold:
                matches.append({
                    'new_idx': i,
                    'old_idx': j,
                    'new_product': new_products[i],
                    'old_product': old_products[j],
                    'similarity': score
                })
    
    # Sort by similarity
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    
    print(f"   🎯 Found {len(matches)} matches above threshold {threshold}")
    
    if matches:
        print(f"\n   🏆 Top 10 matches:")
        print(f"   {'New Product':<40} {'Old Product':<40} {'Score':<8}")
        print(f"   {'-'*40} {'-'*40} {'-'*8}")
        
        for match in matches[:10]:
            new_name = match['new_product'][:37] + "..." if len(match['new_product']) > 40 else match['new_product']
            old_name = match['old_product'][:37] + "..." if len(match['old_product']) > 40 else match['old_product']
            print(f"   {new_name:<40} {old_name:<40} {match['similarity']:.4f}")
    else:
        print(f"   ❌ No matches found above threshold {threshold}")
        
        # Show top matches regardless of threshold
        print(f"\n   🔍 Top 5 matches (any score):")
        all_pairs = []
        for i in range(len(new_products)):
            for j in range(len(old_products)):
                all_pairs.append({
                    'new_product': new_products[i],
                    'old_product': old_products[j],
                    'similarity': similarity_matrix[i][j]
                })
        
        all_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        print(f"   {'New Product':<40} {'Old Product':<40} {'Score':<8}")
        print(f"   {'-'*40} {'-'*40} {'-'*8}")
        
        for pair in all_pairs[:5]:
            new_name = pair['new_product'][:37] + "..." if len(pair['new_product']) > 40 else pair['new_product']
            old_name = pair['old_product'][:37] + "..." if len(pair['old_product']) > 40 else pair['old_product']
            print(f"   {new_name:<40} {old_name:<40} {pair['similarity']:.4f}")
    
    return matches

def test_different_thresholds(similarity_matrix, new_products, old_products):
    """ทดสอบ threshold ต่างๆ"""
    print(f"\n🎚️ Testing different thresholds:")
    
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    
    for threshold in thresholds:
        count = np.sum(similarity_matrix >= threshold)
        percentage = (count / similarity_matrix.size) * 100
        print(f"   >= {threshold:.1f}: {count:4d} pairs ({percentage:5.1f}%)")

def save_results(matches, similarity_matrix, new_products, old_products):
    """บันทึกผลลัพธ์"""
    print(f"\n💾 Saving results...")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Save matches
    if matches:
        matches_df = pd.DataFrame(matches)
        matches_file = output_dir / f"sentence_transformer_matches_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        matches_df.to_csv(matches_file, index=False, encoding='utf-8-sig')
        print(f"   ✅ Matches saved to: {matches_file}")
    
    # Save similarity matrix
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=[f"NEW_{i}_{prod[:30]}" for i, prod in enumerate(new_products)],
        columns=[f"OLD_{i}_{prod[:30]}" for i, prod in enumerate(old_products)]
    )
    matrix_file = output_dir / f"similarity_matrix_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    similarity_df.to_csv(matrix_file, encoding='utf-8-sig')
    print(f"   ✅ Similarity matrix saved to: {matrix_file}")

def main():
    """Main function"""
    print("🚀 Testing SentenceTransformer with Real Product Data")
    print("=" * 65)
    
    # Load data
    data = load_input_data()
    
    if not data:
        print("❌ No data loaded. Please check input folder.")
        return
    
    # Extract product names
    products = extract_product_names(data)
    
    if not products['new'] or not products['old']:
        print("❌ Could not extract product names from both datasets.")
        return
    
    print(f"\n📊 Dataset Summary:")
    print(f"   New products: {len(products['new'])}")
    print(f"   Old products: {len(products['old'])}")
    
    # Show samples
    print(f"\n🔍 Sample new products:")
    for i, product in enumerate(products['new'][:3], 1):
        print(f"   {i}. {product}")
    
    print(f"\n🔍 Sample old products:")
    for i, product in enumerate(products['old'][:3], 1):
        print(f"   {i}. {product}")
    
    # Test model
    model = test_sentence_transformer_with_cache()
    if not model:
        return
    
    # Calculate similarities
    similarity_matrix, new_emb, old_emb = calculate_similarity_matrix(
        model, products['new'], products['old']
    )
    
    # Analyze results
    matches = analyze_similarity_results(
        similarity_matrix, products['new'], products['old'], threshold=0.6
    )
    
    # Test different thresholds
    test_different_thresholds(similarity_matrix, products['new'], products['old'])
    
    # Save results
    save_results(matches, similarity_matrix, products['new'], products['old'])
    
    print(f"\n🎉 Testing completed!")
    print(f"📁 Results saved in output/ folder")

if __name__ == "__main__":
    main()
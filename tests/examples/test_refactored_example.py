"""
Refactored API Integration Test - Using New Organized System
==========================================================

Example of how to refactor existing tests to use new utilities and fixtures
"""

import pytest
from tests.config import TestConfig
from tests.utils import APITestHelper, DataTestHelper


def test_api_integration_refactored(server_running, sample_csv_bytes, api_timeout, long_timeout):
    """
    Refactored API integration test using new organized system
    
    This replaces the duplicate code in test_api_integration.py
    """
    print("🧪 Testing API endpoints with organized system...")
    
    # Skip if server not running (using fixture)
    if not server_running:
        pytest.skip("API server not running")
    
    # 1. Test status endpoint (using helper)
    print("\\n1️⃣ Testing API Status...")
    success, data = APITestHelper.test_api_status('main', api_timeout)
    
    if success:
        print("✅ Status endpoint works")
        print(f"   Response: {data}")
    else:
        pytest.fail(f"❌ Status endpoint failed: {data}")
    
    # 2. Test file upload (using helper and fixtures)
    print("\\n2️⃣ Testing File Upload...")
    old_csv = sample_csv_bytes['old']
    new_csv = sample_csv_bytes['new']
    
    success, response = APITestHelper.test_file_upload(old_csv, new_csv, 'main', long_timeout)
    
    if success:
        print("✅ File upload successful")
        print(f"   Response: {response}")
    else:
        print(f"⚠️ File upload failed: {response}")
        # Continue with other tests
    
    # 3. Test feedback endpoint (using helper)
    print("\\n3️⃣ Testing Save Feedback...")
    feedback_data = APITestHelper.create_sample_feedback_data(
        old_product='ข้าวขาว หอมมะลิ',
        new_product='ข้าวขาว หอมมะลิ 100%',
        similarity=0.9,
        human_feedback='similar'
    )
    
    success, response = APITestHelper.test_save_feedback(feedback_data, 'main', api_timeout)
    
    if success:
        print("✅ Save feedback successful")
        print(f"   Response: {response}")
    else:
        print(f"⚠️ Save feedback failed: {response}")


def test_api_endpoints_comprehensive(api_base_urls, sample_feedback_data, api_timeout):
    """
    Comprehensive API endpoints test using configuration and fixtures
    
    This demonstrates testing multiple servers with same logic
    """
    print("🔍 Testing API endpoints across all servers...")
    
    # Test each configured server
    for server_name, base_url in api_base_urls.items():
        print(f"\\n🔧 Testing {server_name} server ({base_url})...")
        
        # Check if server is running
        if not APITestHelper.check_server_running(server_name):
            print(f"⚠️ {server_name} server not running, skipping...")
            continue
        
        # Test status endpoint
        success, data = APITestHelper.test_api_status(server_name, api_timeout)
        if success:
            print(f"✅ {server_name} status endpoint works")
        else:
            print(f"❌ {server_name} status endpoint failed: {data}")
        
        # Test feedback endpoint (if supported)
        if server_name in ['main', 'alt']:  # Only certain servers support feedback
            success, response = APITestHelper.test_save_feedback(
                sample_feedback_data, server_name, api_timeout
            )
            if success:
                print(f"✅ {server_name} feedback endpoint works")
            else:
                print(f"⚠️ {server_name} feedback endpoint failed: {response}")


def test_model_integration_refactored(sentence_transformer_model, sample_old_products, sample_new_products):
    """
    Refactored model integration test using fixtures
    
    This replaces duplicate model loading code
    """
    print("🤖 Testing model integration with organized system...")
    
    # Model already loaded via session-scoped fixture
    print("✅ Model loaded from fixture")
    
    # Test encoding with sample data
    print("\\n🔤 Testing text encoding...")
    old_embeddings = sentence_transformer_model.encode(sample_old_products)
    new_embeddings = sentence_transformer_model.encode(sample_new_products)
    
    # Verify embeddings
    assert old_embeddings is not None
    assert new_embeddings is not None
    assert len(old_embeddings) == len(sample_old_products)
    assert len(new_embeddings) == len(sample_new_products)
    
    print(f"✅ Encoded {len(sample_old_products)} old products")
    print(f"✅ Encoded {len(sample_new_products)} new products")
    
    # Test similarity calculation
    print("\\n📊 Testing similarity calculation...")
    from sentence_transformers.util import cos_sim
    
    similarities = cos_sim(new_embeddings, old_embeddings)
    assert similarities is not None
    assert similarities.shape == (len(sample_new_products), len(sample_old_products))
    
    print(f"✅ Similarity matrix: {similarities.shape}")
    
    # Find best matches
    for i, new_product in enumerate(sample_new_products):
        best_match_idx = similarities[i].argmax().item()
        best_similarity = similarities[i][best_match_idx].item()
        best_match = sample_old_products[best_match_idx]
        
        print(f"📝 '{new_product}' -> '{best_match}' (similarity: {best_similarity:.3f})")


def test_data_processing_refactored(sample_old_dataframe, sample_new_dataframe, test_output_dir):
    """
    Refactored data processing test using data helpers and fixtures
    
    This replaces duplicate data creation and CSV handling code
    """
    print("📊 Testing data processing with organized system...")
    
    # Data already created via fixtures
    print("✅ Sample data loaded from fixtures")
    print(f"   Old products: {len(sample_old_dataframe)} items")
    print(f"   New products: {len(sample_new_dataframe)} items")
    
    # Validate data structure using helper
    required_columns = ['รายการ']
    old_valid, old_missing = DataTestHelper.validate_csv_structure(sample_old_dataframe, required_columns)
    new_valid, new_missing = DataTestHelper.validate_csv_structure(sample_new_dataframe, required_columns)
    
    assert old_valid, f"Old data missing columns: {old_missing}"
    assert new_valid, f"New data missing columns: {new_missing}"
    
    print("✅ Data structure validation passed")
    
    # Create mock similarity results
    import numpy as np
    old_products = sample_old_dataframe['รายการ'].tolist()
    new_products = sample_new_dataframe['รายการ'].tolist()
    
    # Generate random similarity scores for demonstration
    np.random.seed(42)  # For reproducible results
    similarity_scores = np.random.rand(len(new_products), len(old_products))
    
    # Create similarity matrix using helper
    similarity_df = DataTestHelper.create_similarity_matrix(
        new_products, old_products, similarity_scores
    )
    
    assert similarity_df.shape == (len(new_products), len(old_products))
    print(f"✅ Similarity matrix created: {similarity_df.shape}")
    
    # Save results using helper
    output_file = DataTestHelper.save_test_results(
        similarity_df, 
        "test_similarity_matrix.csv",
        str(test_output_dir)
    )
    
    assert output_file.exists()
    print(f"✅ Results saved to: {output_file}")


if __name__ == "__main__":
    # This file demonstrates the new organized testing approach
    print("🎯 This is an example of refactored tests using the new organized system")
    print("🔧 Run with: pytest tests/examples/test_refactored_example.py -v")
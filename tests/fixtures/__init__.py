"""
Test Fixtures - Shared pytest fixtures for all tests
====================================================

Common fixtures to reduce duplication across test files
"""

import pytest
import pandas as pd
from pathlib import Path
from ..config import TestConfig
from ..utils import APITestHelper, ModelTestHelper, DataTestHelper


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def sentence_transformer_model():
    """
    Session-scoped SentenceTransformer model fixture
    Loads model once per test session to save time
    """
    success, model, error = ModelTestHelper.load_sentence_transformer()
    if success:
        return model
    else:
        pytest.skip(f"Could not load SentenceTransformer: {error}")


@pytest.fixture(scope="session") 
def model_available():
    """Check if SentenceTransformer model is available"""
    success, _, _ = ModelTestHelper.load_sentence_transformer()
    return success


@pytest.fixture
def model_cache_dir():
    """Get model cache directory"""
    return TestConfig.MODELS['cache_dir']


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def sample_old_products():
    """Sample old products for testing"""
    return TestConfig.SAMPLE_PRODUCTS['old'].copy()


@pytest.fixture
def sample_new_products():
    """Sample new products for testing"""
    return TestConfig.SAMPLE_PRODUCTS['new'].copy()


@pytest.fixture
def sample_old_dataframe():
    """Sample old products as DataFrame"""
    return DataTestHelper.create_sample_dataframe('old')


@pytest.fixture
def sample_new_dataframe():
    """Sample new products as DataFrame"""
    return DataTestHelper.create_sample_dataframe('new')


@pytest.fixture
def sample_csv_bytes():
    """Sample CSV data as bytes for API testing"""
    return {
        'old': DataTestHelper.create_test_csv_bytes('old'),
        'new': DataTestHelper.create_test_csv_bytes('new')
    }


@pytest.fixture
def real_input_data():
    """Real input data from files (if available)"""
    return DataTestHelper.load_input_data()


@pytest.fixture
def test_output_dir(tmp_path):
    """Temporary output directory for tests"""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# =============================================================================
# API Fixtures
# =============================================================================

@pytest.fixture
def api_base_urls():
    """API base URLs for testing"""
    return TestConfig.API_URLS.copy()


@pytest.fixture
def sample_feedback_data():
    """Sample feedback data for API testing"""
    return APITestHelper.create_sample_feedback_data()


@pytest.fixture
def api_timeout():
    """Standard API timeout"""
    return TestConfig.REQUEST_SETTINGS['timeout']


@pytest.fixture
def long_timeout():
    """Long timeout for file operations"""
    return TestConfig.REQUEST_SETTINGS['long_timeout']


@pytest.fixture(scope="session")
def server_running():
    """Check if main server is running"""
    return APITestHelper.check_server_running('main')


# =============================================================================
# Test Environment Fixtures  
# =============================================================================

@pytest.fixture
def offline_mode():
    """Enable offline mode for testing"""
    ModelTestHelper.force_offline_mode()
    yield
    ModelTestHelper.clear_offline_mode()


@pytest.fixture
def test_database_path(tmp_path):
    """Temporary database path for testing"""
    return tmp_path / TestConfig.DATABASE['test_db']


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def thai_product_samples():
    """Get sample Thai products for testing"""
    return DataTestHelper.get_sample_thai_products(5)


@pytest.fixture
def similarity_threshold():
    """Default similarity threshold for testing"""
    return 0.7


@pytest.fixture
def confidence_threshold():
    """Default confidence threshold for testing"""
    return 0.8
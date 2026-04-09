# conftest.py - pytest configuration and fixtures
import pytest
import sys
import os
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import shared fixtures from fixtures module
pytest_plugins = ["tests.fixtures"]

# Import configurations (if they exist)
try:
    from main import *  # Import main application components
except ImportError:
    pass  # Continue if main module is not available

@pytest.fixture
def sample_products():
    """Sample product data for testing"""
    return [
        {
            'name': 'iPhone 13 Pro',
            'description': 'Apple iPhone 13 Pro 256GB Blue',
            'price': 999.99,
            'category': 'Electronics'
        },
        {
            'name': 'Samsung Galaxy S21',
            'description': 'Samsung Galaxy S21 5G 128GB Black',
            'price': 799.99,
            'category': 'Electronics'
        }
    ]

@pytest.fixture
def temp_directory(tmp_path):
    """Temporary directory for test files"""
    return tmp_path

@pytest.fixture
def mock_model_cache():
    """Mock model cache for testing"""
    class MockCache:
        def __init__(self):
            self.cache = {}
        
        def get(self, key):
            return self.cache.get(key)
        
        def set(self, key, value):
            self.cache[key] = value
        
        def clear(self):
            self.cache.clear()
    
    return MockCache()

# Add markers for test categorization
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "ui: UI tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
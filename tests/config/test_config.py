"""
Test Configuration - Central configuration for all tests
========================================================

Centralized configuration to reduce duplication across test files
"""

class TestConfig:
    """Central configuration for all tests"""
    
    # API Base URLs
    API_URLS = {
        'main': 'http://localhost:5000',
        'api': 'http://localhost:8000', 
        'alt': 'http://localhost:5001'
    }
    
    # API Endpoints
    ENDPOINTS = {
        'status': '/api/status',
        'feedback': '/save-feedback',
        'upload': '/upload',
        'analyze': '/analyze',
        'cache_stats': '/api/cache/stats',
        'export_ml': '/export-ml-data',
        'health': '/api/v1/health'
    }
    
    # Model Names
    MODELS = {
        'multilingual': 'paraphrase-multilingual-MiniLM-L12-v2',
        'cache_dir': 'model_cache'
    }
    
    # Test Data Paths
    DATA_PATHS = {
        'input_dir': 'input',
        'output_dir': 'output', 
        'new_products': 'input/new_product/POS_เพิ่มสินค้า_20250727_063658_จากไฟล์สินค้าใหม่.csv',
        'old_products': 'input/old_product/cleaned_products.csv'
    }
    
    # Test Database
    DATABASE = {
        'test_db': 'test_human_feedback.db',
        'main_db': 'human_feedback.db'
    }
    
    # Request Settings
    REQUEST_SETTINGS = {
        'timeout': 5,
        'long_timeout': 10
    }
    
    # Sample Thai Product Names for Testing
    SAMPLE_PRODUCTS = {
        'old': [
            'แปลงเก่า ชาไทย',
            'ข้าวขาว หอมมะลิ',
            'น้ำปลา ตราอีก้อน',
            'พริกแกงแดง เก่า',
            'มะม่วงเก่า หวาน',
            'ไม่แขวนเสื้อ - ลวด 111 หนุมาน STCP'
        ],
        'new': [
            'ชาไทย แปลงใหม่',
            'ข้าวขาว หอมมะลิ 100%', 
            'น้ำจิ้มซีฟู๊ด ใหม่',
            'ผงขมิ้น ออร์แกนิค',
            'มะม่วงอร่อย สุกหวาน',
            'ไม่แขวนเสื้อ - ลวด 111 หนุมาน STCP'
        ]
    }
    
    @classmethod
    def get_api_url(cls, server='main'):
        """Get API base URL"""
        return cls.API_URLS.get(server, cls.API_URLS['main'])
    
    @classmethod
    def get_endpoint_url(cls, endpoint, server='main'):
        """Get full endpoint URL"""
        base_url = cls.get_api_url(server)
        endpoint_path = cls.ENDPOINTS.get(endpoint, endpoint)
        return f"{base_url}{endpoint_path}"
    
    @classmethod
    def get_model_name(cls, model_type='multilingual'):
        """Get model name"""
        return cls.MODELS.get(model_type)
"""
API Testing Utilities - Common API testing functions
===================================================

Centralized API testing utilities to reduce duplication
"""

import requests
import json
from typing import Dict, Any, Optional, Tuple
from ..config import TestConfig


class APITestHelper:
    """Helper class for API testing"""
    
    @staticmethod
    def test_api_status(server='main', timeout=None) -> Tuple[bool, Dict[str, Any]]:
        """
        Test API status endpoint
        
        Args:
            server: Server type ('main', 'api', 'alt')
            timeout: Request timeout
            
        Returns:
            Tuple of (success, response_data)
        """
        if timeout is None:
            timeout = TestConfig.REQUEST_SETTINGS['timeout']
            
        try:
            url = TestConfig.get_endpoint_url('status', server)
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {'error': f'Status code: {response.status_code}'}
                
        except Exception as e:
            return False, {'error': str(e)}
    
    @staticmethod
    def test_save_feedback(feedback_data: Dict[str, Any], server='main', timeout=None) -> Tuple[bool, Dict[str, Any]]:
        """
        Test save feedback endpoint
        
        Args:
            feedback_data: Feedback data to send
            server: Server type
            timeout: Request timeout
            
        Returns:
            Tuple of (success, response_data)
        """
        if timeout is None:
            timeout = TestConfig.REQUEST_SETTINGS['timeout']
            
        try:
            url = TestConfig.get_endpoint_url('feedback', server)
            response = requests.post(url, json=feedback_data, timeout=timeout)
            
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {'error': f'Status code: {response.status_code}'}
                
        except Exception as e:
            return False, {'error': str(e)}
    
    @staticmethod
    def test_file_upload(old_csv_data: bytes, new_csv_data: bytes, server='main', timeout=None) -> Tuple[bool, Dict[str, Any]]:
        """
        Test file upload endpoint
        
        Args:
            old_csv_data: Old products CSV data
            new_csv_data: New products CSV data  
            server: Server type
            timeout: Request timeout
            
        Returns:
            Tuple of (success, response_data)
        """
        if timeout is None:
            timeout = TestConfig.REQUEST_SETTINGS['long_timeout']
            
        try:
            url = TestConfig.get_endpoint_url('upload', server)
            files = {
                'old_products': ('old_products.csv', old_csv_data, 'text/csv'),
                'new_products': ('new_products.csv', new_csv_data, 'text/csv')
            }
            
            response = requests.post(url, files=files, timeout=timeout)
            
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {'error': f'Status code: {response.status_code}'}
                
        except Exception as e:
            return False, {'error': str(e)}
    
    @staticmethod
    def create_sample_feedback_data(**overrides) -> Dict[str, Any]:
        """
        Create sample feedback data for testing
        
        Args:
            **overrides: Override default values
            
        Returns:
            Feedback data dictionary
        """
        default_data = {
            'old_product': 'ไม่แขวนเสื้อ - ลวด 111 หนุมาน STCP',
            'new_product': 'ไม่แขวนเสื้อ - ลวด 111 หนุมาน STCP',
            'human_feedback': 'similar',
            'similarity': 1.0,
            'confidence': 0.95,
            'reviewer': 'test_user',
            'comments': 'ทดสอบระบบ API'
        }
        
        default_data.update(overrides)
        return default_data
    
    @staticmethod
    def check_server_running(server='main') -> bool:
        """
        Check if server is running
        
        Args:
            server: Server type to check
            
        Returns:
            True if server is running
        """
        success, _ = APITestHelper.test_api_status(server, timeout=2)
        return success
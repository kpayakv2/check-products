# 🔌 API Reference & Testing Guide

> 📖 **ดูข้อมูลเพิ่มเติม**: [Development Documentation](../development/) สำหรับรายละเอียดเกี่ยวกับสูตรการคำนวณคะแนน, threshold definitions และ confidence levels

## 🚀 **API Server Setup**

### **Starting the Server**
```bash
# Start API server
python api_server.py

# Server will run on:
# - API: http://localhost:8000
# - Docs: http://localhost:8000/docs  
# - Web UI: http://localhost:8000/web
```

### **Health Check**
```bash
curl http://localhost:8000/api/v1/health
```

---

## 📋 **API Endpoints**

### **1. Health Check**
```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-05T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "ml_model": "ready",
    "preprocessor": "ready"
  }
}
```

### **2. Single Product Matching**
```http
POST /api/v1/match/single
Content-Type: application/json
```

**Request Body:**
```json
{
  "query_product": "iPhone 14 Pro Max",
  "reference_products": ["Samsung Galaxy S23", "Huawei P50 Pro"],
  "threshold": 0.6,
  "top_k": 5,
  "include_metadata": true,
  "include_confidence": true
}
```

**Response:**
```json
{
  "matches": [
    {
      "query_product": "iPhone 14 Pro Max",
      "matched_product": "Samsung Galaxy S23",
      "similarity_score": 0.73,
      "confidence_score": 0.6250,
      "confidence_level": "medium",
      "rank": 1
    }
  ],
  "metadata": {
    "processing_time": 0.045,
    "algorithm": "hybrid",
    "similarity_weights": {
      "cosine": 0.7,
      "euclidean": 0.3
    },
    "threshold_used": 0.6,
    "total_comparisons": 2,
    "score_range": {
      "min": 0.31,
      "max": 0.95,
      "average": 0.764
    }
  }
}
```

### **3. Batch Processing**
```http
POST /api/v1/match/batch
Content-Type: application/json
```

**Request Body:**
```json
{
  "query_products": [
    "iPhone 14 Pro Max",
    "Samsung Galaxy S23"
  ],
  "reference_products": [
    "iPhone 14 Pro", 
    "Galaxy S23+",
    "Pixel 7 Pro"
  ],
  "threshold": 0.6,
  "max_matches_per_query": 3
}
```

**Response:**
```json
{
  "job_id": "job_123456789",
  "status": "processing",
  "estimated_time": 2.5,
  "total_queries": 2,
  "message": "Job started successfully"
}
```

### **4. File Upload Processing**
```http
POST /api/v1/match/upload
Content-Type: multipart/form-data
```

**Form Data:**
```
query_file: products_new.csv
reference_file: products_old.csv
threshold: 0.7 (optional)
top_k: 5 (optional)
```

**Response:**
```json
{
  "job_id": "upload_123456789",
  "status": "uploaded",
  "query_count": 150,
  "reference_count": 500,
  "estimated_processing_time": 12.5
}
```

### **5. Job Status Tracking**
```http
GET /api/v1/jobs/{job_id}
```

**Response:**
```json
{
  "job_id": "job_123456789",
  "status": "completed",
  "progress": 100,
  "started_at": "2025-09-05T10:30:00Z",
  "completed_at": "2025-09-05T10:32:30Z",
  "processing_time": 150.5,
  "total_matches": 1248,
  "result_available": true
}
```

### **6. Results Download**
```http
GET /api/v1/results/{job_id}
Accept: application/json  # or text/csv
```

**JSON Response:**
```json
{
  "job_id": "job_123456789",
  "results": [
    {
      "query_product": "iPhone 14",
      "matches": [
        {
          "product": "iPhone 14 Pro",
          "similarity": 0.87,
          "confidence": "high"
        }
      ]
    }
  ],
  "summary": {
    "total_queries": 150,
    "total_matches": 1248,
    "avg_similarity": 0.764,
    "processing_time": 150.5
  }
}
```

---

## 🔌 **WebSocket Real-time Updates**

### **Connection**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = function(event) {
    console.log('WebSocket connected');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    handleUpdate(data);
};
```

### **Message Types**
```javascript
// Connection established
{
    "type": "connection_established",
    "client_id": "client_123",
    "timestamp": "2025-09-05T10:30:00Z"
}

// Job progress update
{
    "type": "progress",
    "job_id": "job_123456789",
    "progress": 75,
    "current": 112,
    "total": 150,
    "estimated_remaining": 45.2
}

// Job completion
{
    "type": "job_completed",
    "job_id": "job_123456789",
    "status": "completed",
    "total_matches": 1248,
    "processing_time": 150.5,
    "download_url": "/api/v1/results/job_123456789"
}

// Error notification
{
    "type": "error",
    "job_id": "job_123456789",
    "error": "Processing failed",
    "details": "Memory limit exceeded"
}
```

---

## 🧪 **Testing Guide**

### **Manual API Testing**

#### **Using cURL**
```bash
# Test health endpoint
curl -X GET http://localhost:8000/api/v1/health

# Test single match
curl -X POST http://localhost:8000/api/v1/match/single \
  -H "Content-Type: application/json" \
  -d '{
    "query_product": "iPhone 14",
    "reference_products": ["Samsung Galaxy S23"],
    "threshold": 0.6
  }'

# Test batch processing
curl -X POST http://localhost:8000/api/v1/match/batch \
  -H "Content-Type: application/json" \
  -d '{
    "query_products": ["iPhone 14", "Galaxy S23"],
    "reference_products": ["iPhone Pro", "Samsung S23+"],
    "threshold": 0.7
  }'
```

#### **Using Python Requests**
```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/api/v1/health")
print(f"Health: {response.json()}")

# Single match
match_request = {
    "query_product": "iPhone 14 Pro Max",
    "reference_products": ["Samsung Galaxy S23", "Pixel 7"],
    "threshold": 0.6,
    "top_k": 3
}

response = requests.post(
    f"{BASE_URL}/api/v1/match/single",
    json=match_request
)
print(f"Matches: {response.json()}")

# File upload
files = {
    'query_file': open('new_products.csv', 'rb'),
    'reference_file': open('old_products.csv', 'rb')
}
data = {'threshold': 0.7}

response = requests.post(
    f"{BASE_URL}/api/v1/match/upload",
    files=files,
    data=data
)
print(f"Upload: {response.json()}")
```

### **Automated Testing**

#### **Running Test Suite**
```bash
# Run all tests
pytest tests/

# Run API-specific tests
python test_api_client.py

# Run with detailed output
pytest tests/ -v --tb=short

# Run performance tests
pytest tests/test_performance.py
```

#### **API Test Client**
```python
# test_api_client.py example usage
from test_api_client import APITestClient

client = APITestClient(base_url="http://localhost:8000")

# Run comprehensive tests
client.test_health_endpoint()
client.test_single_match()
client.test_batch_processing()
client.test_file_upload()
client.test_websocket_connection()

print("All API tests passed!")
```

### **Load Testing**
```python
import asyncio
import aiohttp
import time

async def load_test_single_match(session, product_id):
    """Test single match endpoint under load"""
    payload = {
        "query_product": f"Test Product {product_id}",
        "reference_products": ["Reference 1", "Reference 2"],
        "threshold": 0.6
    }
    
    async with session.post('/api/v1/match/single', json=payload) as response:
        return await response.json()

async def run_load_test(concurrent_requests=50):
    """Run load test with multiple concurrent requests"""
    connector = aiohttp.TCPConnector(limit=100)
    async with aiohttp.ClientSession(
        "http://localhost:8000",
        connector=connector
    ) as session:
        
        start_time = time.time()
        
        tasks = [
            load_test_single_match(session, i) 
            for i in range(concurrent_requests)
        ]
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Completed {concurrent_requests} requests in {duration:.2f}s")
        print(f"Average response time: {duration/concurrent_requests:.3f}s")
        print(f"Requests per second: {concurrent_requests/duration:.2f}")

# Run load test
asyncio.run(run_load_test(100))
```

---

## 📊 **Performance Monitoring**

### **Built-in Metrics**
```python
# Available in API responses
{
  "metadata": {
    "processing_time": 0.045,        # Total processing time (seconds)
    "preprocessing_time": 0.012,     # Text preprocessing time
    "embedding_time": 0.025,         # Embedding generation time
    "similarity_time": 0.008,        # Similarity calculation time
    "memory_usage": "45.2MB",        # Peak memory usage
    "algorithm": "tfidf",            # Algorithm used
    "model_version": "1.0.0"         # Model version
  }
}
```

### **Health Monitoring**
```bash
# Monitor API health
watch -n 5 'curl -s http://localhost:8000/api/v1/health | jq'

# Monitor system resources
htop

# Monitor API logs (if running with logging)
tail -f api_server.log
```

---

## 🔧 **Configuration**

### **Environment Variables**
```bash
# API Configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export API_WORKERS=4

# Model Configuration  
export MODEL_TYPE=tfidf
export SIMILARITY_THRESHOLD=0.6
export MAX_BATCH_SIZE=1000

# Performance
export ENABLE_CACHING=true
export CACHE_SIZE=1000
export EMBEDDING_BATCH_SIZE=32
```

### **API Server Configuration**
```python
# Configuration in api_server.py
class APIConfig:
    host: str = "localhost"
    port: int = 8000
    workers: int = 1
    
    # Model settings
    model_type: str = "tfidf"
    similarity_threshold: float = 0.6
    
    # Performance
    enable_background_jobs: bool = True
    max_concurrent_jobs: int = 10
    job_timeout: int = 3600  # 1 hour
```

---

## 🛠️ **Troubleshooting**

### **Common Issues**

#### **Server Won't Start**
```bash
# Check if port is in use
netstat -an | grep 8000

# Kill existing process
pkill -f api_server.py

# Check Python environment
python --version
pip list | grep fastapi
```

#### **Slow API Response**
```python
# Check model loading
import time
start = time.time()
from fresh_implementations import ComponentFactory
model = ComponentFactory.create_embedding_model("tfidf")
print(f"Model loading time: {time.time() - start:.2f}s")

# Monitor memory usage
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

#### **WebSocket Connection Issues**
```javascript
// Check WebSocket connection
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onerror = function(error) {
    console.error('WebSocket error:', error);
};

ws.onclose = function(event) {
    console.log('WebSocket closed:', event.code, event.reason);
};
```

### **Error Codes**
```http
400 Bad Request     - Invalid request format
422 Validation Error - Invalid input parameters  
500 Internal Error  - Server processing error
503 Service Unavailable - Server overloaded
```

---

## 📝 **API Response Schemas**

### **Standard Response Format**
```json
{
  "success": true,
  "data": { ... },
  "metadata": { ... },
  "timestamp": "2025-09-05T10:30:00Z"
}
```

### **Error Response Format**
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid threshold value",
    "details": {
      "field": "threshold",
      "value": -0.5,
      "expected": "between 0.0 and 1.0"
    }
  },
  "timestamp": "2025-09-05T10:30:00Z"
}
```

---

**🔌 API นี้พร้อมใช้งานจริงใน production environment และรองรับการใช้งานที่หลากหลาย!**

# 📚 Product Similarity Checker - Complete Documentation

## 🎯 **Project Overview**

**Product Similarity Checker** เป็นระบบ AI สำหรับจับคู่และค้นหาสินค้าที่คล้ายกันโดยใช้เทคโนโลยี Machine Learning และ Natural Language Processing

---

## 📋 **Table of Contents**

1. [🚀 Quick Start](#-quick-start)
2. [🏗️ Architecture Overview](#️-architecture-overview)  
3. [🧹 Text Preprocessing](#-text-preprocessing)
4. [📊 Project Capabilities](#-project-capabilities)
5. [🔧 API Documentation](#-api-documentation)
6. [🧪 Testing Guide](#-testing-guide)
7. [🤝 Contributing](#-contributing)

---

## 🚀 **Quick Start**

### Installation
```bash
# Clone repository
git clone https://github.com/kpayakv2/check-products.git
cd check-products

# Install dependencies
pip install -r requirements.txt

# Run basic matching
python main.py old_products.csv new_products.csv

# Start API server (Phase 5)
python api_server.py

# Access web interface
open http://localhost:8000/web
```

### Basic Usage
```python
# Phase 3: Basic CLI
python main.py

# Phase 4: Enhanced processing
python main_phase4.py old.csv new.csv --enhanced

# Phase 5: API & Web interface
curl -X POST http://localhost:8000/api/v1/match/single \
  -H "Content-Type: application/json" \
  -d '{"query_product": "iPhone 14", "reference_products": ["Samsung Galaxy"]}'
```

---

## 🏗️ **Architecture Overview**

### **System Architecture**
```
┌─────────────────────────────────────┐
│          🌐 Web Interface           │  ← Phase 5: Modern Dashboard
├─────────────────────────────────────┤
│          🔌 REST API Layer          │  ← Phase 5: FastAPI Server
├─────────────────────────────────────┤
│       🧠 Business Logic Layer       │  ← Phase 4: Enhanced Processing
├─────────────────────────────────────┤
│        🏛️ Architecture Layer        │  ← Phase 2: Clean Architecture
├─────────────────────────────────────┤
│        📊 Data Processing Layer     │  ← Phase 3: Core Matching
└─────────────────────────────────────┘
```

### **Phase Evolution**

#### **Phase 3: Core Engine** (`main.py`)
- Basic product similarity matching
- SentenceTransformer + Cosine Similarity
- CSV input/output processing

#### **Phase 4: Advanced Processing** (`main_phase4.py`)
- Performance optimization (2,900+ matches/second)
- Enhanced configuration and reporting
- Multi-algorithm support (TF-IDF + SentenceTransformer)

#### **Phase 5: Production API** (`api_server.py`)
- FastAPI REST server with real-time capabilities
- Modern web dashboard with WebSocket updates
- Background job processing and progress tracking

### **Core Components**

#### **Architecture Foundation**
- **`fresh_architecture.py`** - Abstract interfaces and design patterns
- **`fresh_implementations.py`** - Concrete implementations and factory patterns

#### **Data Processing**
- **`clean_csv_products.py`** - CSV data cleaning and preparation
- **`filter_matched_products.py`** - Result filtering and refinement
- **`src/core/preprocessing.py`** - Advanced text preprocessing

#### **Web & API**
- **`api_server.py`** - Production REST API server
- **`web/index.html`** - Modern interactive dashboard
- **`test_api_client.py`** - API testing and validation

---

## 🧹 **Text Preprocessing**

### **Overview**
Text preprocessing คือหัวใจสำคัญของการเตรียมข้อมูลก่อนเข้า ML model เพื่อเพิ่มประสิทธิภาพการจับคู่สินค้า

### **Preprocessing Classes**

#### **1. BasicTextPreprocessor** - การทำความสะอาดพื้นฐาน
```python
# Features:
- Unicode normalization
- Lowercase conversion  
- Extra whitespace removal
- Special character handling

# Example:
"iPhone   14   PRO  MAX!!!" → "iphone 14 pro max!!!"
```

#### **2. ThaiTextPreprocessor** - การประมวลผลภาษาไทยเฉพาะ
```python
# Features:
- Thai character normalization (เ็ → เ)
- Thai number conversion (๑๒ใ → 123)
- Tone mark handling
- Thai-specific cleaning

# Example:
"ไอโฟน ๑๔ โปร แม็กซ์" → "ไอโฟน 14 โปร แม็กซ์"
```

#### **3. ProductTextPreprocessor** - การประมวลผลเฉพาะสินค้า
```python
# Features:
- Brand prefix removal (แบรนด์, ยี่ห้อ)
- Unit normalization (กก. → กิโลกรัม)
- Promotional text removal (ราคาพิเศษ, โปรโมชั่น)
- Color standardization (แดง → สีแดง)

# Example:
"แบรนด์ iPhone 14 Pro Max สีแดง 256GB ราคาพิเศษ"
→ "iphone 14 pro max สีแดง 256gb"
```

#### **4. ChainedTextPreprocessor** - การรวมการประมวลผลหลายขั้นตอน
```python
# Sequential processing pipeline
preprocessor = ChainedTextPreprocessor([
    BasicTextPreprocessor(),
    ThaiTextPreprocessor(), 
    ProductTextPreprocessor()
])

# Example:
"แบรนด์ ไอโฟน ๑๔ โปร แม็กซ์ สีแดง ๒๕๖GB ราคาพิเศษ!!!"
→ "ไอโฟน 14 โปร แม็กซ์ สีแดง 256gb"
```

### **Handling Typos & Whitespace**

#### **Whitespace Management**
- **Regex normalization**: `re.sub(r'\s+', ' ', text).strip()`
- **Multi-level processing**: Basic → Thai → Product
- **Unicode normalization**: `unicodedata.normalize('NFKC', text)`

#### **Typo Handling Strategy**
```python
# Instead of traditional spell checking, uses semantic similarity
text1 = "ไอโฟนน์ 14 โปรว แม็กซ์"  # with typos
text2 = "iPhone 14 Pro Max"         # correct spelling

# ML model understands they mean the same thing
similarity = cosine_similarity(embeddings1, embeddings2)
# Result: 0.85+ (very similar)
```

**Key Benefits:**
- ✅ No dictionary required: ML model learns from data
- ✅ Better Thai support than traditional spell checkers
- ✅ Context-aware understanding
- ✅ Flexible threshold adjustment

---

## 📊 **Project Capabilities**

### **🎯 Core Functionality**

#### **Product Similarity Matching**
- ค้นหาสินค้าที่คล้ายกันจากชื่อสินค้า (ไทย/อังกฤษ)
- คำนวณคะแนนความคล้ายด้วย AI algorithms
- ประสิทธิภาพ: **2,900+ matches/second**

#### **Real-world Performance**
```
✅ Input: 406 product entries
✅ Output: 1,248 matches found  
⚡ Processing Time: 0.429 seconds
🎯 Average Similarity: 76.84%
```

### **🌐 Multi-Interface Support**

#### **Web Interface** (`http://localhost:8000/web`)
- Modern responsive dashboard
- Real-time processing with WebSocket updates
- Interactive charts and visualizations
- Mobile-friendly design

#### **REST API** (`http://localhost:8000/docs`)
```http
POST /api/v1/match/single      # Single product matching
POST /api/v1/match/batch       # Batch processing
POST /api/v1/match/upload      # File upload processing
GET  /api/v1/jobs/{job_id}     # Job status tracking
GET  /api/v1/results/{job_id}  # Download results
WebSocket /ws                  # Real-time updates
```

#### **Command Line Interface**
```bash
# Phase 3: Basic
python main.py old.csv new.csv

# Phase 4: Enhanced  
python main_phase4.py old.csv new.csv --enhanced --threshold 0.7

# Utilities
python clean_csv_products.py --input raw.csv --output clean.csv
python filter_matched_products.py --input results.csv --threshold 0.6
```

### **🎯 Use Cases**

#### **E-commerce Applications**
- **Product Deduplication**: หาสินค้าซ้ำในระบบ
- **Recommendation Systems**: แนะนำสินค้าที่คล้ายกัน
- **Inventory Management**: จับคู่สินค้าในคลังสินค้า
- **Price Comparison**: เปรียบเทียบราคาสินค้าคล้าย

#### **Data Management**
- **Database Cleanup**: ทำความสะอาดฐานข้อมูล
- **Data Integration**: รวมข้อมูลจากหลายแหล่ง
- **Quality Assessment**: ประเมินคุณภาพข้อมูล

---

## 🔧 **API Documentation**

### **Authentication**
Currently no authentication required (development mode)

### **Endpoints**

#### **Health Check**
```http
GET /api/v1/health
```
Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-09-05T10:30:00Z",
  "version": "1.0.0"
}
```

#### **Single Product Matching**
```http
POST /api/v1/match/single
Content-Type: application/json

{
  "query_product": "iPhone 14 Pro Max",
  "reference_products": ["Samsung Galaxy S23", "Huawei P50"],
  "threshold": 0.6,
  "top_k": 5
}
```

#### **Batch Processing**
```http
POST /api/v1/match/batch
Content-Type: application/json

{
  "query_products": ["iPhone 14", "Samsung S23"],
  "reference_products": ["iPhone 14 Pro", "Galaxy S23+"],
  "threshold": 0.6
}
```

#### **File Upload**
```http
POST /api/v1/match/upload
Content-Type: multipart/form-data

Form Data:
- query_file: CSV file with query products
- reference_file: CSV file with reference products  
- threshold: 0.6 (optional)
```

#### **WebSocket Connection**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'progress') {
        console.log(`Progress: ${data.progress}%`);
    }
};
```

### **Response Format**
```json
{
  "matches": [
    {
      "query_product": "iPhone 14",
      "matched_product": "iPhone 14 Pro",
      "similarity_score": 0.87,
      "confidence": "high"
    }
  ],
  "metadata": {
    "processing_time": 0.045,
    "total_matches": 1248,
    "algorithm": "tfidf",
    "threshold": 0.6
  }
}
```

---

## 🧪 **Testing Guide**

### **Running Tests**
```bash
# All tests
pytest

# With coverage
pytest --cov=main --cov-report=html

# Specific test file
pytest tests/test_functions.py

# API testing
python test_api_client.py
```

### **Test Structure**
```
tests/
├── test_functions.py     # Core function testing
├── test_run_output.py    # Output validation
├── test_smoke.py         # Basic functionality
└── test_util.py          # Utility function tests
```

### **Testing Infrastructure**
- **Unit Tests**: Individual function testing
- **Integration Tests**: Module interaction testing
- **Performance Tests**: Speed and memory benchmarks
- **API Tests**: REST endpoint validation
- **Smoke Tests**: Basic functionality verification

---

## 🤝 **Contributing**

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Code formatting
black .
isort .

# Type checking
mypy src/

# Linting
flake8 src/
```

### **Project Structure**
```
check-products/
├── 📁 Core Processing
│   ├── main.py                    # Phase 3: Basic engine
│   ├── main_phase4.py             # Phase 4: Enhanced processing  
│   ├── fresh_architecture.py      # Abstract interfaces
│   └── fresh_implementations.py   # Concrete implementations
├── 📁 API & Web
│   ├── api_server.py              # FastAPI server
│   ├── web/index.html             # Web dashboard
│   └── test_api_client.py         # API testing
├── 📁 Data Processing
│   ├── clean_csv_products.py      # CSV cleaning
│   ├── filter_matched_products.py # Result filtering
│   └── src/core/preprocessing.py  # Text preprocessing
├── 📁 Testing
│   └── tests/                     # Test suites
└── 📁 Documentation
    └── docs/                      # This documentation
```

### **Contribution Guidelines**
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass
5. Format code with black and isort
6. Submit pull request

### **Code Standards**
- Follow PEP 8 style guidelines
- Write comprehensive docstrings
- Add type hints for all functions
- Maintain test coverage above 80%
- Document new features and changes

---

## 📞 **Support & Contact**

- **Issues**: [GitHub Issues](https://github.com/kpayakv2/check-products/issues)
- **Documentation**: This comprehensive guide
- **API Reference**: http://localhost:8000/docs (when server is running)

---

## 📝 **License**

This project is open source and available under the MIT License.

---

**🎉 Ready to start matching products with AI? Follow the Quick Start guide above!**

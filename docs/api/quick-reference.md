# 🚀 **Quick API Reference - /analyze Endpoint**

## 💡 **TL;DR - Quick Start**

```bash
# 1. Start server
python web_server.py

# 2. Upload files
POST http://localhost:5000/upload (old products)
POST http://localhost:5000/upload (new products)  

# 3. Analyze
POST http://localhost:5000/analyze

# 4. Get results
GET http://localhost:5000/get-results
```

---

## 🎯 **Key Capabilities**

| Feature | Status | Description |
|---------|--------|-------------|
| 🤖 **AI Analysis** | ✅ | Multi-model product similarity detection |
| 🌐 **Offline Mode** | ✅ | Works without internet (models cached) |
| 🇹🇭 **Multilingual** | ✅ | Thai-English product name support |
| 👥 **Human Review** | ✅ | Interactive verification workflow |
| 📊 **Smart Classification** | ✅ | Auto-categorize unique vs duplicate |
| ⚡ **Performance** | ✅ | Optimized for speed and accuracy |

---

## 📈 **Performance Comparison**

| Model | Speed | Accuracy | Memory | Best For |
|-------|-------|----------|--------|----------|
| **TF-IDF** | ⚡⚡⚡ | ⭐⭐⭐ | 💾 | Large datasets |
| **SentenceTransformer** | ⚡ | ⭐⭐⭐⭐⭐ | 💾💾💾 | High accuracy |
| **Optimized TF-IDF** | ⚡⚡ | ⭐⭐⭐⭐ | 💾💾 | Balanced |

---

## 🔧 **Configuration Options**

```json
{
  "model_type": "sentence-bert",  // ⭐ Recommended for accuracy
  "similarity_method": "cosine",
  "threshold": 0.6,              // Higher = stricter
  "top_k": 10
}
```

**Change via**: `POST /api/config`

---

## 📊 **Response Data**

```json
{
  "success": true,
  "unique_count": 750,           // ✅ Definitely unique
  "duplicate_check_count": 250,  // 🔍 Need human review  
  "summary": "จาก 1000 สินค้าใหม่: 750 ไม่ซ้ำ, 250 ต้องตรวจสอบ"
}
```

---

## 🎭 **Use Cases**

### **E-commerce** 📱
- Prevent duplicate product listings
- Clean product catalogs  
- Merge supplier databases

### **Inventory** 📦
- Deduplicate stock items
- Consolidate product data
- Quality assurance checks

### **Data Management** 📊
- Clean messy datasets
- Find data inconsistencies
- Merge multiple sources

---

## ⚡ **Performance Tips**

### **For Speed** 🏃‍♂️:
- Use `"model_type": "tfidf"`
- Lower `threshold` (0.4-0.6)  
- Process in smaller batches

### **For Accuracy** 🎯:
- Use `"model_type": "sentence-bert"`
- Higher `threshold` (0.7-0.8)
- Enable human review workflow

### **For Balance** ⚖️:
- Use `"model_type": "optimized-tfidf"`
- Medium `threshold` (0.6-0.7)
- Batch size 100-500 items

---

## 🔍 **Troubleshooting**

| Issue | Solution |
|-------|----------|
| **"Upload both files"** | Upload old & new product files first |
| **"Pipeline error"** | Check model availability, restart server |
| **Slow performance** | Switch to TF-IDF model, reduce batch size |
| **Low accuracy** | Use SentenceTransformer, increase threshold |
| **Memory error** | Process smaller batches, increase RAM |

---

## 📱 **Mobile-Friendly**

✅ Responsive web interface  
✅ Touch-optimized human review  
✅ Progress indicators  
✅ Mobile file upload  

---

## 🔐 **Security & Privacy**

✅ Local processing (no data sent to external APIs)  
✅ Offline capable (no internet required)  
✅ File upload validation  
✅ Error handling and logging  

---

## 📞 **Support**

- **API Status**: `GET /api/status`
- **Live Demo**: http://localhost:5000  
- **Documentation**: `API_ANALYZE_CAPABILITIES.md`
- **Test Suite**: `python test_offline_capability.py`

---

**⚡ Ready to use!** Start with TF-IDF for speed, upgrade to SentenceTransformer for accuracy!

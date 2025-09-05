# 🧹 Advanced Text Preprocessing Guide

## 🎯 **Overview**

Text preprocessing เป็นขั้นตอนสำคัญที่เตรียมข้อมูลก่อนป้อนเข้า ML model เพื่อเพิ่มประสิทธิภาพการจับคู่สินค้า โดยเฉพาะสำหรับข้อมูลภาษาไทยและสินค้าที่มีลักษณะเฉพาะ

---

## 🔧 **Preprocessing Classes**

### **1. BasicTextPreprocessor** - การทำความสะอาดพื้นฐาน

```python
class BasicTextPreprocessor(TextPreprocessor):
    def __init__(self, 
                 lowercase: bool = True,           # แปลงเป็นตัวเล็ก
                 remove_extra_spaces: bool = True, # ลบช่องว่างเกิน
                 remove_special_chars: bool = False, # ลบอักขระพิเศษ
                 normalize_unicode: bool = True):   # Normalize Unicode
```

**🔧 การทำงาน:**
```python
# Unicode Normalization
text = unicodedata.normalize('NFKC', text)

# Lowercase Conversion
if self.lowercase:
    text = text.lower()

# Special Character Removal
if self.remove_special_chars:
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s\.,!?()-]', '', text)

# Extra Space Removal
if self.remove_extra_spaces:
    text = re.sub(r'\s+', ' ', text).strip()
```

**📊 ตัวอย่าง:**
```python
Input:  "iPhone   14   PRO  MAX!!!"
Output: "iphone 14 pro max!!!"

Input:  "SAMSUNG    Galaxy   S23"
Output: "samsung galaxy s23"
```

---

### **2. ThaiTextPreprocessor** - การประมวลผลภาษาไทยเฉพาะ

```python
class ThaiTextPreprocessor(TextPreprocessor):
    def __init__(self,
                 normalize_thai_chars: bool = True,  # แก้ไขการเรียงอักษร
                 remove_tone_marks: bool = False,    # ลบวรรณยุกต์
                 standardize_spaces: bool = True,    # จัดช่องว่าง
                 normalize_numbers: bool = True):    # แปลงเลขไทยเป็นอารบิก
```

**🇹🇭 Thai-Specific Features:**
```python
# Thai Character Normalization
if self.normalize_thai_chars:
    text = text.replace('เ็', 'เ')  # Fix character ordering
    text = text.replace('แ็', 'แ')

# Thai Number Conversion
if self.normalize_numbers:
    thai_to_arabic = str.maketrans('๐๑๒๓๔๕๖๗๘๙', '0123456789')
    text = text.translate(thai_to_arabic)

# Tone Mark Removal (optional)
if self.remove_tone_marks:
    tone_marks = {'\u0E48', '\u0E49', '\u0E4A', '\u0E4B'}
    text = ''.join(char for char in text if char not in tone_marks)
```

**📊 ตัวอย่าง:**
```python
Input:  "ไอโฟน ๑๔ โปร แม็กซ์"
Output: "ไอโฟน 14 โปร แม็กซ์"

Input:  "แซมซุง แกแลกซี่ เอส๒๓"
Output: "แซมซุง แกแลกซี่ เอส23"
```

---

### **3. ProductTextPreprocessor** - การประมวลผลเฉพาะสินค้า

```python
class ProductTextPreprocessor(TextPreprocessor):
    def __init__(self,
                 remove_brand_prefixes: bool = True,    # ลบคำนำหน้ายี่ห้อ
                 normalize_units: bool = True,          # แปลงหน่วยวัด
                 remove_promotional_text: bool = True,  # ลบข้อความโปรโมชั่น
                 standardize_colors: bool = True):      # มาตรฐานสี
```

**🏷️ Product-Specific Cleaning:**
```python
# Brand Prefix Removal
brand_prefixes = {'แบรนด์', 'ยี่ห้อ', 'brand', 'original', 'authentic'}
for prefix in brand_prefixes:
    pattern = rf'\b{re.escape(prefix)}\s*'
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)

# Unit Normalization
unit_mappings = {
    'กก.': 'กิโลกรัม', 'ก.': 'กรัม', 'มล.': 'มิลลิลิตร',
    'kg': 'กิโลกรัม', 'ml': 'มิลลิลิตร'
}
for old_unit, new_unit in unit_mappings.items():
    pattern = rf'\b{re.escape(old_unit)}\b'
    text = re.sub(pattern, new_unit, text, flags=re.IGNORECASE)

# Promotional Text Removal
promotional_phrases = {'ราคาพิเศษ', 'โปรโมชั่น', 'ลดราคา', 'special', 'sale'}
for phrase in promotional_phrases:
    pattern = rf'\b{re.escape(phrase)}\b'
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)

# Color Standardization
color_mappings = {
    'แดง': 'สีแดง', 'red': 'สีแดง',
    'เขียว': 'สีเขียว', 'green': 'สีเขียว'
}
for old_color, new_color in color_mappings.items():
    pattern = rf'\b{re.escape(old_color)}\b'
    text = re.sub(pattern, new_color, text, flags=re.IGNORECASE)
```

**📊 ตัวอย่าง:**
```python
Input:  "แบรนด์ iPhone 14 Pro Max สีแดง 256GB ราคาพิเศษ"
Output: "iphone 14 pro max สีแดง 256gb"

Input:  "ยี่ห้อ Samsung Galaxy S23 สีดำ โปรโมชั่น"
Output: "samsung galaxy s23 สีดำ"
```

---

### **4. ChainedTextPreprocessor** - การรวมการประมวลผลหลายขั้นตอน

```python
class ChainedTextPreprocessor(TextPreprocessor):
    def __init__(self, preprocessors: List[TextPreprocessor]):
        self.preprocessors = preprocessors
    
    def preprocess(self, text: str) -> str:
        for preprocessor in self.preprocessors:
            text = preprocessor.preprocess(text)
        return text
```

**🔗 Default Thai Product Pipeline:**
```python
def create_default_thai_product_preprocessor() -> TextPreprocessor:
    return ChainedTextPreprocessor([
        BasicTextPreprocessor(
            lowercase=True,
            remove_extra_spaces=True,
            remove_special_chars=False,  # Keep for Thai compatibility
            normalize_unicode=True
        ),
        ThaiTextPreprocessor(
            normalize_thai_chars=True,
            remove_tone_marks=False,     # Keep for better semantics
            standardize_spaces=True,
            normalize_numbers=True
        ),
        ProductTextPreprocessor(
            remove_brand_prefixes=True,
            normalize_units=True,
            remove_promotional_text=True,
            standardize_colors=True
        )
    ])
```

**📊 ตัวอย่างการทำงานแบบ Chained:**
```python
Input:  "แบรนด์ ไอโฟน ๑๔ โปร แม็กซ์ สีแดง ๒๕๖GB ราคาพิเศษ!!!"

# Step 1: BasicTextPreprocessor
→ "แบรนด์ ไอโฟน ๑๔ โปร แม็กซ์ สีแดง ๒๕๖gb ราคาพิเศษ"

# Step 2: ThaiTextPreprocessor  
→ "แบรนด์ ไอโฟน 14 โปร แม็กซ์ สีแดง 256gb ราคาพิเศษ"

# Step 3: ProductTextPreprocessor
→ "ไอโฟน 14 โปร แม็กซ์ สีแดง 256gb"

Final Output: "ไอโฟน 14 โปร แม็กซ์ สีแดง 256gb"
```

---

## 🛠️ **Factory Functions**

### **create_text_preprocessor()** - Flexible Preprocessor Creation
```python
def create_text_preprocessor(preprocessor_type: str, **kwargs) -> TextPreprocessor:
    if preprocessor_type.lower() == "basic":
        return BasicTextPreprocessor(**kwargs)
    elif preprocessor_type.lower() == "thai":
        return ThaiTextPreprocessor(**kwargs)
    elif preprocessor_type.lower() == "product":
        return ProductTextPreprocessor(**kwargs)
    elif preprocessor_type.lower() == "chained":
        return ChainedTextPreprocessor(**kwargs)
    else:
        raise ValueError(f"Unknown preprocessor type: {preprocessor_type}")

# Usage examples:
basic_preprocessor = create_text_preprocessor("basic", lowercase=True)
thai_preprocessor = create_text_preprocessor("thai", normalize_numbers=True)
product_preprocessor = create_text_preprocessor("product", remove_brand_prefixes=True)
```

---

## 🔧 **Handling Whitespace & Typos**

### **Whitespace Management Strategy**
```python
# Multi-level whitespace normalization
def normalize_whitespace(text: str) -> str:
    # 1. Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # 2. Remove leading/trailing spaces
    text = text.strip()
    
    # 3. Handle special Unicode spaces
    text = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000]', ' ', text)
    
    return text

# Examples:
"iPhone    14    Pro   Max" → "iPhone 14 Pro Max"
"ไอโฟน   ๑๔   โปร    แม็กซ์" → "ไอโฟน ๑๔ โปร แม็กซ์"
```

### **Typo Handling Approach**
**แนวทาง: Semantic Similarity แทน Traditional Spell Checking**

```python
# Instead of correcting typos directly, use ML model semantic understanding
def handle_typos_via_similarity():
    # Examples that ML model can understand as similar:
    typo_examples = [
        ("ไอโฟนน์", "ไอโฟน"),      # Extra character
        ("แซมซุง", "ซัมซุง"),       # Character variation
        ("แมคบุ๊ค", "แม็คบุค"),     # Different spelling
    ]
    
    # ML model understanding:
    for typo, correct in typo_examples:
        embeddings_typo = model.encode([typo])
        embeddings_correct = model.encode([correct])
        similarity = cosine_similarity(embeddings_typo, embeddings_correct)
        print(f"{typo} ≈ {correct}: {similarity:.2f}")
        # Results typically > 0.8 (high similarity)
```

**Benefits of ML Approach:**
- ✅ **No Dictionary Required**: Model learns from data
- ✅ **Context Aware**: Understands meaning, not just spelling
- ✅ **Thai Language Support**: Better than traditional spell checkers
- ✅ **Flexible Thresholds**: Adjustable tolerance for typos

---

## 📈 **Performance in ML Pipeline**

### **Before vs After Preprocessing**
```python
# Example comparison
def demonstrate_preprocessing_impact():
    # Raw data (with issues)
    raw_products = [
        "แบรนด์   ไอโฟนน์  ๑๔   โปรว   แม็กซ์  สีแดง   ราคาพิเศษ!!!",
        "iPhone 14 Pro Max Red 256GB"
    ]
    
    # Without preprocessing
    raw_similarity = calculate_similarity(raw_products[0], raw_products[1])
    print(f"Raw similarity: {raw_similarity:.2f}")  # Low score ~0.3
    
    # With preprocessing
    preprocessor = create_default_thai_product_preprocessor()
    clean_products = [preprocessor.preprocess(p) for p in raw_products]
    clean_similarity = calculate_similarity(clean_products[0], clean_products[1])
    print(f"Clean similarity: {clean_similarity:.2f}")  # High score ~0.85
    
    # Cleaned versions:
    # "ไอโฟน 14 โปรว แม็กซ์ สีแดง"
    # "iphone 14 pro max สีแดง 256gb"
```

### **Integration with ML Pipeline**
```python
def complete_processing_pipeline():
    # 1. Load raw data
    raw_products = load_csv_data("products.csv")
    
    # 2. Text preprocessing
    preprocessor = create_default_thai_product_preprocessor()
    cleaned_products = preprocessor.preprocess_batch(raw_products)
    
    # 3. Generate embeddings
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = embedding_model.encode(cleaned_products)
    
    # 4. Calculate similarities
    similarities = cosine_similarity(embeddings)
    
    # 5. Filter and rank results
    filtered_results = filter_by_threshold(similarities, threshold=0.6)
    
    return filtered_results
```

---

## 🔬 **Advanced Configuration**

### **Custom Preprocessing Pipelines**
```python
# For specific domains
def create_electronics_preprocessor():
    return ChainedTextPreprocessor([
        BasicTextPreprocessor(lowercase=True, normalize_unicode=True),
        ThaiTextPreprocessor(normalize_numbers=True, normalize_thai_chars=True),
        ProductTextPreprocessor(
            remove_brand_prefixes=True,
            normalize_units=True,
            custom_stopwords={'รุ่น', 'โมเดล', 'model', 'version'}
        )
    ])

# For fashion products
def create_fashion_preprocessor():
    return ChainedTextPreprocessor([
        BasicTextPreprocessor(lowercase=True),
        ThaiTextPreprocessor(normalize_thai_chars=True),
        ProductTextPreprocessor(
            standardize_colors=True,
            remove_promotional_text=True,
            custom_stopwords={'คอลเลกชั่น', 'collection', 'limited'}
        )
    ])
```

### **Dynamic Configuration**
```python
def create_configurable_preprocessor(config: Dict[str, Any]):
    preprocessors = []
    
    if config.get('enable_basic', True):
        preprocessors.append(BasicTextPreprocessor(**config.get('basic', {})))
    
    if config.get('enable_thai', True):
        preprocessors.append(ThaiTextPreprocessor(**config.get('thai', {})))
    
    if config.get('enable_product', True):
        preprocessors.append(ProductTextPreprocessor(**config.get('product', {})))
    
    return ChainedTextPreprocessor(preprocessors)

# Usage with config file
config = {
    'enable_basic': True,
    'basic': {'lowercase': True, 'remove_extra_spaces': True},
    'enable_thai': True, 
    'thai': {'normalize_numbers': True},
    'enable_product': True,
    'product': {'remove_brand_prefixes': True}
}

preprocessor = create_configurable_preprocessor(config)
```

---

## 🧪 **Testing Preprocessing Functions**

### **Unit Testing Examples**
```python
def test_basic_preprocessor():
    preprocessor = BasicTextPreprocessor()
    
    # Test lowercase
    assert preprocessor.preprocess("IPHONE 14") == "iphone 14"
    
    # Test space normalization
    assert preprocessor.preprocess("iPhone   14   Pro") == "iphone 14 pro"
    
    # Test unicode normalization
    assert preprocessor.preprocess("café") == "café"

def test_thai_preprocessor():
    preprocessor = ThaiTextPreprocessor()
    
    # Test number conversion
    assert preprocessor.preprocess("ไอโฟน ๑๔") == "ไอโฟน 14"
    
    # Test character normalization
    assert preprocessor.preprocess("เ็ก") == "เก"

def test_product_preprocessor():
    preprocessor = ProductTextPreprocessor()
    
    # Test brand removal
    assert "แบรนด์" not in preprocessor.preprocess("แบรนด์ iPhone")
    
    # Test color standardization
    assert preprocessor.preprocess("สีแดง") == "สีแดง"
    assert preprocessor.preprocess("red") == "สีแดง"
```

### **Integration Testing**
```python
def test_chained_preprocessing():
    preprocessor = create_default_thai_product_preprocessor()
    
    input_text = "แบรนด์ ไอโฟน ๑๔ โปร แม็กซ์ สีแดง ราคาพิเศษ"
    expected = "ไอโฟน 14 โปร แม็กซ์ สีแดง"
    
    result = preprocessor.preprocess(input_text)
    assert result == expected
```

---

## 🎯 **Best Practices**

### **✅ Recommended Approaches**
1. **Use Chained Processing**: รวมหลาย preprocessors สำหรับผลลัพธ์ที่ดีที่สุด
2. **Keep Thai-specific Features**: ไม่ลบ tone marks เพื่อรักษาความหมาย
3. **Domain-specific Customization**: ปรับแต่งตาม product categories
4. **Test Thoroughly**: ทดสอบกับข้อมูลจริงจากหลายแหล่ง

### **⚠️ Common Pitfalls**
1. **Over-aggressive Cleaning**: ลบข้อมูลสำคัญออกไปด้วย
2. **Ignoring Unicode**: ไม่จัดการ Unicode normalization
3. **Hard-coded Rules**: ใช้ rules ที่ fixed เกินไป
4. **No Testing**: ไม่ทดสอบกับข้อมูลจริง

---

**🧹 Text preprocessing เป็นศิลปะที่ต้องการความเข้าใจทั้งข้อมูลและโดเมน - การทำให้ดีต้องอาศัยการทดลองและปรับแต่งอย่างต่อเนื่อง!**

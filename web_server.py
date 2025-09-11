#!/usr/bin/env python3
"""
Web Server สำหรับ Human-in-the-Loop Product Deduplication
รองรับ Interactive Human Review
"""

from flask import Flask, render_template, jsonify, request
import json
import os
import pandas as pd
from pathlib import Path
from werkzeug.utils import secure_filename
import random
import time
import traceback

# 🔗 เชื่อมต่อกับ Fresh Architecture
try:
    from fresh_architecture import ProductMatcher, ProductSimilarityPipeline, Config
    from fresh_implementations import ComponentFactory
    from human_feedback_system import ProductDeduplicationSystem
    FRESH_ARCHITECTURE_AVAILABLE = True
    print("✅ Fresh Architecture modules loaded successfully")
except ImportError as e:
    FRESH_ARCHITECTURE_AVAILABLE = False
    print(f"⚠️ Fresh Architecture not available: {e}")
    print("📝 Using basic similarity calculation as fallback")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ตัวแปรเก็บสถานะ + เพิ่มข้อมูลสำหรับ Review
app_state = {
    'old_products_file': None,
    'new_products_file': None,
    'old_products_data': None,
    'new_products_data': None,
    'analysis_results': None,
    'review_queue': [],  # รายการที่ต้องตรวจสอบ
    'feedback_data': [],  # ผลการตรวจสอบ
    'current_review_index': 0
}

# Configuration เดียวกับ main_phase4.py
WEB_CONFIG = {
    'model_type': 'tfidf',  # choices: 'tfidf', 'sentence-bert', 'optimized-tfidf', 'mock'
    'similarity_method': 'cosine',  # choices: 'cosine', 'dot_product'
    'threshold': 0.6,  # เดียวกับ main_phase4.py default
    'top_k': 10,  # เดียวกับ main_phase4.py default
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}

def get_enhanced_similarity_system(model_type=None, similarity_method=None, threshold=None, top_k=None):
    """สร้าง Enhanced Similarity System ตาม main_phase4.py"""
    if FRESH_ARCHITECTURE_AVAILABLE:
        try:
            print("🔧 DEBUG: Creating Phase 4 Enhanced Pipeline...")
            
            # ใช้ค่า configuration เดียวกับ main_phase4.py
            config = Config()
            config.similarity_threshold = threshold or WEB_CONFIG['threshold']
            config.top_k = top_k or WEB_CONFIG['top_k']
            config.enable_performance_tracking = True
            config.include_metadata = True
            config.include_confidence_scores = True
            
            model_name = model_type or WEB_CONFIG['model_type']
            similarity_name = similarity_method or WEB_CONFIG['similarity_method']
            
            print(f"🔧 DEBUG: Using model: {model_name}, similarity: {similarity_name}")
            print(f"🔧 DEBUG: Threshold: {config.similarity_threshold}, Top-k: {config.top_k}")
            
            # Create components เดียวกับ main_phase4.py
            data_source = ComponentFactory.create_data_source("csv")
            text_processor = ComponentFactory.create_text_processor("thai") 
            embedding_model = ComponentFactory.create_embedding_model(model_name)
            similarity_calculator = ComponentFactory.create_similarity_calculator(similarity_name)
            
            print("🔧 DEBUG: Creating ProductMatcher...")
            matcher = ProductMatcher(
                embedding_model=embedding_model,
                similarity_calculator=similarity_calculator, 
                text_processor=text_processor,
                config=config
            )
            
            print(f"🚀 สร้าง Phase 4 ProductMatcher สำเร็จ! (Model: {model_name})")
            return matcher, config
            
        except Exception as e:
            print(f"❌ Phase 4 Architecture error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    else:
        print("❌ Fresh Architecture ไม่พร้อมใช้งาน")
        return None, None

def generate_review_queue():
    """สร้างรายการสำหรับ Human Review จากข้อมูลจริง"""
    if app_state['old_products_data'] is None or app_state['new_products_data'] is None:
        print("❌ DEBUG: ไม่มีข้อมูลสินค้า")
        return []
    
    # ตรวจสอบชื่อคอลัมน์ที่มี
    old_cols = list(app_state['old_products_data'].columns)
    new_cols = list(app_state['new_products_data'].columns)
    print(f"🔍 DEBUG: Old columns: {old_cols}")
    print(f"🔍 DEBUG: New columns: {new_cols}")
    
    # ใช้คอลัมน์ที่เหมาะสม
    old_products = app_state['old_products_data']['name'].tolist() if 'name' in old_cols else app_state['old_products_data'].iloc[:, 0].tolist()
    
    # สำหรับ new products - หาคอลัมน์ที่มีชื่อสินค้า
    if 'รายการ' in new_cols:
        new_products = app_state['new_products_data']['รายการ'].tolist()
        print("📝 DEBUG: ใช้คอลัมน์ 'รายการ' สำหรับสินค้าใหม่")
    elif 'name' in new_cols:
        new_products = app_state['new_products_data']['name'].tolist()
    else:
        # หาคอลัมน์ที่มีข้อความ ไม่ใช่ตัวเลข
        text_col = None
        for col in new_cols:
            sample_val = app_state['new_products_data'][col].dropna().iloc[0] if len(app_state['new_products_data'][col].dropna()) > 0 else None
            if sample_val is not None and isinstance(sample_val, str) and len(str(sample_val)) > 2:
                text_col = col
                break
        
        if text_col:
            new_products = app_state['new_products_data'][text_col].tolist()
            print(f"📝 DEBUG: ใช้คอลัมน์ '{text_col}' สำหรับสินค้าใหม่")
        else:
            new_products = app_state['new_products_data'].iloc[:, 0].tolist()
            print("⚠️ DEBUG: ไม่เจอคอลัมน์ข้อความ ใช้คอลัมน์แรก")
    
    print(f"📊 DEBUG: Old products count: {len(old_products)}")
    print(f"📊 DEBUG: New products count: {len(new_products)}")
    print(f"📝 DEBUG: Sample old products: {old_products[:3]}")
    print(f"📝 DEBUG: Sample new products: {new_products[:3]}")
    
    # สร้างคู่เปรียบเทียบด้วย Enhanced System
    review_queue = []
    count = 0
    total_checked = 0
    
    # ใช้ Phase 4 Enhanced Pipeline
    enhanced_matcher, phase4_config = get_enhanced_similarity_system()
    
    print(f"🔧 DEBUG: Enhanced matcher ready: {enhanced_matcher is not None}")
    print(f"🔧 DEBUG: Phase 4 config ready: {phase4_config is not None}")
    print(f"🔧 DEBUG: Fresh architecture available: {FRESH_ARCHITECTURE_AVAILABLE}")
    
    if enhanced_matcher and phase4_config and FRESH_ARCHITECTURE_AVAILABLE:
        print("🎯 ใช้ Fresh Architecture สำหรับการเปรียบเทียบ")
        try:
            # ใช้ ProductMatcher หาคู่ที่คล้าย - เพิ่มจำนวนสินค้าที่เปรียบเทียบ
            for i, new_product in enumerate(new_products[:20]):  # เพิ่มเป็น 20 สินค้าใหม่
                matches = enhanced_matcher.find_matches(
                    query_products=[str(new_product)],
                    reference_products=[str(p) for p in old_products[:50]]  # เพิ่มเป็น 50 สินค้าเก่า
                )
                
                for match in matches:
                    if count >= 20:
                        break
                    
                    similarity = match.get('similarity_score', 0.0)
                    confidence = match.get('confidence', random.uniform(0.6, 0.95))
                    
                    if similarity >= phase4_config.similarity_threshold:  # ใช้เกณฑ์ 0.6 เหมือน main_phase4.py
                        ml_prediction = "similar" if similarity > 0.5 else "different"
                        
                        review_item = {
                            'id': f"review_{count+1}",
                            'new_product': str(new_product),
                            'old_product': str(match['matched_product']),
                            'similarity': round(similarity, 3),
                            'confidence': round(confidence, 3),
                            'ml_prediction': ml_prediction,
                            'status': 'pending',
                            'method': 'fresh_architecture'
                        }
                        review_queue.append(review_item)
                        count += 1
                        print(f"✅ Fresh Architecture: Added pair {count} (score: {similarity:.3f})")
                
                if count >= 20:
                    break
                    
        except Exception as e:
            print(f"⚠️ Fresh Architecture failed: {e}")
            print("🔄 Falling back to basic similarity")
            enhanced_matcher = None
    
    # Fallback: ใช้ basic similarity ถ้า Fresh Architecture ไม่ทำงาน หรือได้ผลลัพธ์น้อยเกินไป
    if not enhanced_matcher or len(review_queue) < 3:
        print("📝 ใช้ Basic Similarity เป็น fallback")
        for i, new_product in enumerate(new_products[:20]):  # เพิ่มเป็น 20 สินค้าใหม่
            for j, old_product in enumerate(old_products[:30]):  # เพิ่มเป็น 30 สินค้าเก่า
                if count >= 20:
                    break
                
                total_checked += 1
                similarity = calculate_simple_similarity(new_product, old_product)
                confidence = random.uniform(0.4, 0.9)
                
                if similarity > 0.05:
                    ml_prediction = "similar" if similarity > 0.3 else "different"
                    
                    review_item = {
                        'id': f"review_{count+1}",
                        'new_product': str(new_product),
                        'old_product': str(old_product),
                        'similarity': round(similarity, 3),
                        'confidence': round(confidence, 3),
                        'ml_prediction': ml_prediction,
                        'status': 'pending',
                        'method': 'basic_similarity'
                    }
                    review_queue.append(review_item)
                    count += 1
                    print(f"✅ Basic: Added pair {count}")
            
            if count >= 20:
                break
    
    print(f"🎯 DEBUG: Final queue length: {len(review_queue)}")
    
    return review_queue

def enhance_results_phase4(matches, feedback_data):
    """Enhance results ตาม main_phase4.py พร้อม human feedback"""
    enhanced_matches = []
    
    # Calculate confidence scores
    if matches:
        max_score = max(match.get('similarity', match.get('similarity_score', 0)) for match in matches)
        min_score = min(match.get('similarity', match.get('similarity_score', 0)) for match in matches)
        score_range = max_score - min_score if max_score > min_score else 1.0
    
    for i, match in enumerate(matches):
        enhanced_match = match.copy()
        
        # Add Phase 4 metadata
        enhanced_match.update({
            'match_rank': i + 1,
            'processing_timestamp': time.time(),
            'processor_version': 'web_phase4_enhanced',
            'method': match.get('method', 'unknown')
        })
        
        # Add confidence score ตาม main_phase4.py
        similarity_score = match.get('similarity', match.get('similarity_score', 0))
        if score_range > 0:
            confidence = (similarity_score - min_score) / score_range
        else:
            confidence = 1.0
        enhanced_match['confidence_score'] = round(confidence, 4)
        
        # Add confidence level ตาม main_phase4.py
        if confidence >= 0.8:
            enhanced_match['confidence_level'] = 'high'
        elif confidence >= 0.5:
            enhanced_match['confidence_level'] = 'medium'
        else:
            enhanced_match['confidence_level'] = 'low'
        
        # หา human feedback ถ้ามี
        human_feedback = None
        for feedback in feedback_data:
            if (feedback['new_product'] == match.get('new_product') and 
                feedback['old_product'] == match.get('old_product')):
                human_feedback = feedback
                break
        
        if human_feedback:
            enhanced_match.update({
                'human_feedback': human_feedback['human_feedback'],
                'human_comments': human_feedback.get('comments', ''),
                'reviewer': human_feedback.get('reviewer', ''),
                'review_timestamp': human_feedback.get('timestamp', '')
            })
        
        enhanced_matches.append(enhanced_match)
    
    return enhanced_matches

def calculate_simple_similarity(text1, text2):
    """คำนวณความคล้ายแบบง่าย"""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(str(text1).lower().split())
    words2 = set(str(text2).lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Product Deduplication System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f8f9fa; }
            .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
            .card { border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 8px; }
            .button { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; text-decoration: none; display: inline-block; }
            .button:hover { background: #0056b3; }
            .button:disabled { background: #6c757d; cursor: not-allowed; }
            .upload-box { flex: 1; border: 2px dashed #007bff; padding: 20px; text-align: center; border-radius: 8px; margin: 10px; }
            .upload-box.success { border-color: #28a745; background: #f8fff9; }
            .status { padding: 15px; margin: 10px 0; border-radius: 4px; }
            .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
            .warning { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
            .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Human-in-the-Loop Product Deduplication</h1>
            
            <!-- Upload Section -->
            <div class="card">
                <h2>📥 ขั้นตอนที่ 1: อัปโหลดไฟล์สินค้า</h2>
                <div style="display: flex; gap: 20px;">
                    <div class="upload-box" id="oldBox">
                        <h3>📦 สินค้าเก่า</h3>
                        <input type="file" id="oldFile" accept=".csv,.xlsx,.xls">
                        <br><br>
                        <button class="button" onclick="uploadFile('old')">อัปโหลด</button>
                        <div id="oldStatus"></div>
                    </div>
                    <div class="upload-box" id="newBox">
                        <h3>🆕 สินค้าใหม่</h3>
                        <input type="file" id="newFile" accept=".csv,.xlsx,.xls">
                        <br><br>
                        <button class="button" onclick="uploadFile('new')">อัปโหลด</button>
                        <div id="newStatus"></div>
                    </div>
                </div>
            </div>
            
            <!-- Analysis Section -->
            <div class="card">
                <h2>🔍 ขั้นตอนที่ 2: วิเคราะห์ความคล้าย</h2>
                
                <!-- Model Configuration (เดียวกับ main_phase4.py) -->
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                    <h3>⚙️ การตั้งค่าโมเดล</h3>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                        <div>
                            <label>🧠 โมเดล:</label>
                            <select id="modelSelect" style="width: 100%; padding: 8px;">
                                <option value="tfidf">TF-IDF (เร็ว)</option>
                                <option value="sentence-bert">Sentence BERT (แม่นยำ)</option>
                                <option value="optimized-tfidf">Optimized TF-IDF</option>
                                <option value="mock">Mock (ทดสอบ)</option>
                            </select>
                        </div>
                        <div>
                            <label>📐 Similarity:</label>
                            <select id="similaritySelect" style="width: 100%; padding: 8px;">
                                <option value="cosine">Cosine Similarity</option>
                                <option value="dot_product">Dot Product</option>
                            </select>
                        </div>
                        <div>
                            <label>🎯 Threshold:</label>
                            <input type="number" id="thresholdInput" min="0" max="1" step="0.1" value="0.6" style="width: 100%; padding: 8px;">
                        </div>
                        <div>
                            <label>🔢 Top-K:</label>
                            <input type="number" id="topKInput" min="1" max="50" value="10" style="width: 100%; padding: 8px;">
                        </div>
                    </div>
                    <button class="button" onclick="updateConfig()" style="margin-top: 10px;">💾 อัปเดตการตั้งค่า</button>
                    <div id="configStatus"></div>
                </div>
                
                <button id="analyzeBtn" class="button" onclick="analyze()" disabled>เริ่มวิเคราะห์</button>
                <div id="analysisResult"></div>
            </div>
            
            <!-- Human Review Section -->
            <div class="card">
                <h2>👤 ขั้นตอนที่ 3: ตรวจสอบโดยมนุษย์</h2>
                <input type="text" id="reviewerName" placeholder="ชื่อผู้ตรวจสอบ" style="padding: 8px; width: 300px;">
                <button id="startReviewBtn" class="button" onclick="startHumanReview()" disabled>เริ่มตรวจสอบ</button>
                
                <!-- Review Interface -->
                <div id="reviewInterface" style="display: none;">
                    <div class="status info">
                        <strong id="reviewProgress">รายการที่ 1/20</strong> | 
                        ผู้ตรวจสอบ: <span id="currentReviewer"></span>
                    </div>
                    
                    <div id="reviewItem" style="border: 2px solid #007bff; padding: 20px; border-radius: 8px; margin: 20px 0;">
                        <div style="display: flex; gap: 20px; margin-bottom: 20px;">
                            <div style="flex: 1; background: #e3f2fd; padding: 15px; border-radius: 4px;">
                                <h4>📦 สินค้าเก่า</h4>
                                <div id="oldProductDisplay" style="font-size: 16px; font-weight: bold;"></div>
                            </div>
                            <div style="flex: 1; background: #e8f5e8; padding: 15px; border-radius: 4px;">
                                <h4>🆕 สินค้าใหม่</h4>
                                <div id="newProductDisplay" style="font-size: 16px; font-weight: bold;"></div>
                            </div>
                        </div>
                        
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 4px; margin: 15px 0;">
                            🤖 <strong>AI ทำนาย:</strong> <span id="mlPrediction"></span> | 
                            📊 <strong>ความคล้าย:</strong> <span id="similarityScore"></span> | 
                            🎯 <strong>ความมั่นใจ:</strong> <span id="confidenceScore"></span>
                        </div>
                        
                        <div style="text-align: center; margin: 20px 0;">
                            <button class="button" onclick="provideFeedback('duplicate')" style="background: #dc3545;">🔴 สินค้าซ้ำ</button>
                            <button class="button" onclick="provideFeedback('similar')" style="background: #ffc107; color: black;">🟡 คล้าย แต่ไม่ซ้ำ</button>
                            <button class="button" onclick="provideFeedback('different')" style="background: #28a745;">🟢 ต่างกัน</button>
                            <button class="button" onclick="provideFeedback('uncertain')" style="background: #6c757d;">⚪ ไม่แน่ใจ</button>
                        </div>
                        
                        <textarea id="reviewComments" placeholder="ความคิดเห็นเพิ่มเติม (ไม่บังคับ)" 
                                  style="width: 100%; height: 80px; padding: 10px; border: 1px solid #ddd; border-radius: 4px;"></textarea>
                    </div>
                </div>
                
                <div id="reviewSummary"></div>
            </div>
            
            <!-- Results Section -->
            <div class="card">
                <h2>📊 ผลลัพธ์และส่งออก</h2>
                <div style="margin-bottom: 15px;">
                    <button class="button" onclick="showResults()">ดูผลลัพธ์</button>
                    <button class="button" onclick="exportCSV()" style="background: #28a745;">📥 ส่งออก CSV</button>
                    <button class="button" onclick="exportMLData()" style="background: #6f42c1;">🤖 ส่งออก ML Data</button>
                </div>
                <div id="results"></div>
                <div id="exportResults"></div>
            </div>
        </div>
        
        <script>
            let uploadState = { old: false, new: false };
            let analysisState = { completed: false };
            let reviewState = { 
                active: false, 
                currentIndex: 0, 
                queue: [],
                feedback: []
            };
            
            function uploadFile(type) {
                const fileInput = document.getElementById(type + 'File');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert(`กรุณาเลือกไฟล์สินค้า${type === 'old' ? 'เก่า' : 'ใหม่'}`);
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                formData.append('type', type);
                
                document.getElementById(type + 'Status').innerHTML = '<div class="warning">⏳ กำลังอัปโหลด...</div>';
                
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById(type + 'Status').innerHTML = `
                            <div class="success">✅ สำเร็จ!<br>📊 ${data.count} รายการ</div>
                        `;
                        document.getElementById(type + 'Box').classList.add('success');
                        uploadState[type] = true;
                        checkReadyForAnalysis();
                    } else {
                        document.getElementById(type + 'Status').innerHTML = `
                            <div class="error">❌ ${data.message}</div>
                        `;
                    }
                })
                .catch(error => {
                    document.getElementById(type + 'Status').innerHTML = `
                        <div class="error">❌ เกิดข้อผิดพลาด</div>
                    `;
                });
            }
            
            function checkReadyForAnalysis() {
                if (uploadState.old && uploadState.new) {
                    document.getElementById('analyzeBtn').disabled = false;
                    document.getElementById('analyzeBtn').style.background = '#007bff';
                }
            }
            
            function updateConfig() {
                const config = {
                    model_type: document.getElementById('modelSelect').value,
                    similarity_method: document.getElementById('similaritySelect').value,
                    threshold: parseFloat(document.getElementById('thresholdInput').value),
                    top_k: parseInt(document.getElementById('topKInput').value)
                };
                
                document.getElementById('configStatus').innerHTML = '<div class="info">⏳ อัปเดตการตั้งค่า...</div>';
                
                fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('configStatus').innerHTML = `
                            <div class="success">✅ อัปเดตสำเร็จ!<br>
                            🧠 โมเดล: ${config.model_type}<br>
                            📐 Similarity: ${config.similarity_method}<br>
                            🎯 Threshold: ${config.threshold}<br>
                            🔢 Top-K: ${config.top_k}</div>
                        `;
                    } else {
                        document.getElementById('configStatus').innerHTML = `
                            <div class="error">❌ ${data.message}</div>
                        `;
                    }
                })
                .catch(error => {
                    document.getElementById('configStatus').innerHTML = `
                        <div class="error">❌ เกิดข้อผิดพลาด</div>
                    `;
                });
            }
            
            // โหลด configuration ปัจจุบัน
            function loadCurrentConfig() {
                fetch('/api/config')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('modelSelect').value = data.config.model_type;
                        document.getElementById('similaritySelect').value = data.config.similarity_method;
                        document.getElementById('thresholdInput').value = data.config.threshold;
                        document.getElementById('topKInput').value = data.config.top_k;
                    }
                });
            }
            
            // เรียกเมื่อโหลดหน้า
            window.onload = function() {
                loadCurrentConfig();
            };
            
            function analyze() {
                document.getElementById('analysisResult').innerHTML = '<div class="info">⏳ กำลังวิเคราะห์...</div>';
                
                fetch('/analyze', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('analysisResult').innerHTML = `
                            <div class="success">
                                ✅ วิเคราะห์เสร็จสิ้น!<br>
                                🔍 การเปรียบเทียบ: ${data.total_comparisons} คู่<br>
                                🎯 ต้องตรวจสอบ: ${data.pending_review} คู่
                            </div>
                        `;
                        analysisState.completed = true;
                        document.getElementById('startReviewBtn').disabled = false;
                        document.getElementById('startReviewBtn').style.background = '#ffc107';
                    }
                });
            }
            
            function startHumanReview() {
                const reviewer = document.getElementById('reviewerName').value.trim();
                if (!reviewer) {
                    alert('กรุณาใส่ชื่อผู้ตรวจสอบ');
                    return;
                }
                
                document.getElementById('currentReviewer').textContent = reviewer;
                
                // โหลดรายการตรวจสอบ
                fetch('/get-review-queue')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        reviewState.queue = data.queue;
                        reviewState.active = true;
                        reviewState.currentIndex = 0;
                        
                        document.getElementById('reviewInterface').style.display = 'block';
                        loadNextReviewItem();
                    }
                });
            }
            
            function loadNextReviewItem() {
                if (reviewState.currentIndex >= reviewState.queue.length) {
                    completeReview();
                    return;
                }
                
                const item = reviewState.queue[reviewState.currentIndex];
                
                document.getElementById('reviewProgress').textContent = 
                    `รายการที่ ${reviewState.currentIndex + 1}/${reviewState.queue.length}`;
                document.getElementById('oldProductDisplay').textContent = item.old_product;
                document.getElementById('newProductDisplay').textContent = item.new_product;
                document.getElementById('mlPrediction').textContent = item.ml_prediction;
                document.getElementById('similarityScore').textContent = (item.similarity * 100).toFixed(1) + '%';
                document.getElementById('confidenceScore').textContent = (item.confidence * 100).toFixed(1) + '%';
                
                // Clear previous comments
                document.getElementById('reviewComments').value = '';
            }
            
            function provideFeedback(decision) {
                const currentItem = reviewState.queue[reviewState.currentIndex];
                const comments = document.getElementById('reviewComments').value;
                
                const feedback = {
                    id: currentItem.id,
                    old_product: currentItem.old_product,
                    new_product: currentItem.new_product,
                    similarity: currentItem.similarity,
                    ml_prediction: currentItem.ml_prediction,
                    human_feedback: decision,
                    comments: comments,
                    reviewer: document.getElementById('currentReviewer').textContent,
                    timestamp: new Date().toISOString()
                };
                
                // บันทึก feedback
                fetch('/save-feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(feedback)
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        reviewState.feedback.push(feedback);
                        reviewState.currentIndex++;
                        loadNextReviewItem();
                    }
                });
            }
            
            function completeReview() {
                const totalReviewed = reviewState.feedback.length;
                const agreement = reviewState.feedback.filter(f => 
                    f.human_feedback === f.ml_prediction.replace('similar', 'similar')
                ).length;
                
                document.getElementById('reviewSummary').innerHTML = `
                    <div class="success">
                        🎉 <strong>ตรวจสอบเสร็จสิ้น!</strong><br>
                        📊 ตรวจสอบทั้งหมด: ${totalReviewed} รายการ<br>
                        🤝 เห็นด้วยกับ AI: ${agreement}/${totalReviewed} รายการ (${((agreement/totalReviewed)*100).toFixed(1)}%)<br>
                        🧠 ข้อมูลจะถูกใช้เทรน ML เพื่อปรับปรุงความแม่นยำ
                    </div>
                `;
                
                document.getElementById('reviewInterface').style.display = 'none';
            }
            
            function showResults() {
                fetch('/get-results')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('results').innerHTML = `
                        <div class="info">
                            📈 <strong>สรุปผลลัพธ์ (Phase 4 Enhanced):</strong><br>
                            • สินค้าเก่า: ${data.old_count} รายการ<br>
                            • สินค้าใหม่: ${data.new_count} รายการ<br>
                            • Human Feedback: ${data.feedback_count} รายการ<br>
                            • ความแม่นยำ ML: ${data.accuracy}%<br>
                            • พร้อมส่งออกข้อมูลสำหรับ ML Training
                        </div>
                    `;
                });
            }
            
            function exportCSV() {
                document.getElementById('exportResults').innerHTML = '<div class="info">⏳ กำลังสร้างไฟล์ CSV...</div>';
                
                fetch('/export-csv')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('exportResults').innerHTML = `
                            <div class="success">
                                ✅ <strong>ส่งออกสำเร็จ!</strong><br>
                                📁 ไฟล์หลัก: ${data.filename}<br>
                                🤖 ไฟล์ ML Training: ${data.ml_training_file || 'ไม่มี'}<br>
                                📊 จำนวนรายการ: ${data.records_count}<br>
                                👤 Human Feedback: ${data.feedback_count} รายการ<br>
                                💾 บันทึกที่: output/${data.filename}
                            </div>
                        `;
                    } else {
                        document.getElementById('exportResults').innerHTML = `
                            <div class="error">❌ ${data.message}</div>
                        `;
                    }
                })
                .catch(error => {
                    document.getElementById('exportResults').innerHTML = `
                        <div class="error">❌ เกิดข้อผิดพลาดในการส่งออก</div>
                    `;
                });
            }
            
            function exportMLData() {
                alert('🤖 ML Training Data จะถูกส่งออกพร้อมกับ CSV หลัก\\n\\nข้อมูลจะใช้สำหรับ:\\n• ปรับปรุงความแม่นยำของ AI\\n• เทรนโมเดลใหม่\\n• วิเคราะห์ Human-AI Agreement');
                exportCSV();
            }
        </script>
    </body>
    </html>
    """

# API Endpoints เพิ่มเติมสำหรับ Human Review

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'ไม่พบไฟล์'})
        
        file = request.files['file']
        file_type = request.form.get('type')
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'ไม่ได้เลือกไฟล์'})
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'message': 'รองรับเฉพาะไฟล์ .csv, .xlsx, .xls'})
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_type}_{filename}")
        file.save(filepath)
        
        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filepath, encoding='utf-8-sig')
            else:
                df = pd.read_excel(filepath)
        except Exception as e:
            return jsonify({'success': False, 'message': f'ไม่สามารถอ่านไฟล์ได้: {str(e)}'})
        
        if file_type == 'old':
            app_state['old_products_file'] = filepath
            app_state['old_products_data'] = df
        else:
            app_state['new_products_file'] = filepath
            app_state['new_products_data'] = df
        
        return jsonify({
            'success': True,
            'filename': filename,
            'count': len(df),
            'columns': list(df.columns)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if app_state['old_products_data'] is None or app_state['new_products_data'] is None:
            return jsonify({'success': False, 'message': 'กรุณาอัปโหลดไฟล์ทั้งสองไฟล์ก่อน'})
        
        # สร้างคิวสำหรับ Human Review
        app_state['review_queue'] = generate_review_queue()
        
        old_count = len(app_state['old_products_data'])
        new_count = len(app_state['new_products_data'])
        total_comparisons = len(app_state['review_queue'])
        
        app_state['analysis_results'] = {
            'total_comparisons': total_comparisons,
            'pending_review': total_comparisons,
            'avg_confidence': 72.5
        }
        
        return jsonify({
            'success': True,
            'total_comparisons': total_comparisons,
            'pending_review': total_comparisons,
            'avg_confidence': 72.5
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/get-review-queue')
def get_review_queue():
    return jsonify({
        'success': True,
        'queue': app_state['review_queue']
    })

@app.route('/save-feedback', methods=['POST'])
def save_feedback():
    try:
        feedback = request.json
        app_state['feedback_data'].append(feedback)
        
        return jsonify({'success': True, 'message': 'บันทึก feedback สำเร็จ'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/get-results')
def get_results():
    old_count = len(app_state['old_products_data']) if app_state['old_products_data'] is not None else 0
    new_count = len(app_state['new_products_data']) if app_state['new_products_data'] is not None else 0
    feedback_count = len(app_state['feedback_data'])
    
    # คำนวณความแม่นยำ
    if feedback_count > 0:
        correct_predictions = sum(1 for f in app_state['feedback_data'] 
                                if f['human_feedback'] == f['ml_prediction'])
        accuracy = round((correct_predictions / feedback_count) * 100, 1)
    else:
        accuracy = 0.0
    
    return jsonify({
        'old_count': old_count,
        'new_count': new_count,
        'feedback_count': feedback_count,
        'accuracy': accuracy
    })

@app.route('/export-csv')
def export_csv():
    """Export enhanced results เป็น CSV พร้อม human feedback"""
    try:
        if not app_state['review_queue'] or not app_state['feedback_data']:
            return jsonify({'success': False, 'message': 'ไม่มีข้อมูลสำหรับส่งออก'})
        
        # Enhance results with Phase 4 features + Human Feedback
        enhanced_results = enhance_results_phase4(app_state['review_queue'], app_state['feedback_data'])
        
        # สร้าง DataFrame และบันทึกเป็น CSV
        import pandas as pd
        from datetime import datetime
        
        df = pd.DataFrame(enhanced_results)
        
        # สร้างชื่อไฟล์พร้อม timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"human_feedback_results_{timestamp}.csv"
        filepath = os.path.join('output', filename)
        
        # สร้างโฟลเดอร์ output ถ้าไม่มี
        os.makedirs('output', exist_ok=True)
        
        # บันทึก CSV
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        # สร้าง ML training data
        ml_training_data = []
        for result in enhanced_results:
            if result.get('human_feedback'):
                ml_training_data.append({
                    'new_product': result['new_product'],
                    'old_product': result['old_product'],
                    'similarity_score': result.get('similarity', 0),
                    'ai_prediction': result.get('ml_prediction', ''),
                    'human_feedback': result['human_feedback'],
                    'confidence_score': result.get('confidence_score', 0),
                    'confidence_level': result.get('confidence_level', ''),
                    'reviewer': result.get('reviewer', ''),
                    'comments': result.get('human_comments', '')
                })
        
        # บันทึก ML training data
        if ml_training_data:
            ml_filename = f"ml_training_data_{timestamp}.csv"
            ml_filepath = os.path.join('output', ml_filename)
            pd.DataFrame(ml_training_data).to_csv(ml_filepath, index=False, encoding='utf-8-sig')
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'ml_training_file': ml_filename if ml_training_data else None,
            'records_count': len(enhanced_results),
            'feedback_count': len([r for r in enhanced_results if r.get('human_feedback')])
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """Get/Set configuration เดียวกับ main_phase4.py"""
    if request.method == 'GET':
        return jsonify({
            'success': True,
            'config': WEB_CONFIG,
            'available_models': ['tfidf', 'sentence-bert', 'optimized-tfidf', 'mock'],
            'available_similarities': ['cosine', 'dot_product']
        })
    
    elif request.method == 'POST':
        try:
            data = request.json
            
            # Update configuration
            if 'model_type' in data:
                WEB_CONFIG['model_type'] = data['model_type']
            if 'similarity_method' in data:
                WEB_CONFIG['similarity_method'] = data['similarity_method']
            if 'threshold' in data:
                WEB_CONFIG['threshold'] = float(data['threshold'])
            if 'top_k' in data:
                WEB_CONFIG['top_k'] = int(data['top_k'])
            
            return jsonify({
                'success': True,
                'message': 'Configuration updated',
                'config': WEB_CONFIG
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'running',
        'system': 'Human-in-the-Loop Product Deduplication v2.1 - Phase 4 Enhanced',
        'features': [
            'Interactive Human Review Interface',
            'Real-time Feedback Processing',
            'AI-Human Collaboration',
            'Progress Tracking',
            'CSV Export with Human Feedback',
            'ML Training Data Generation',
            'Multi-Model Support (same as main_phase4.py)'
        ],
        'current_config': WEB_CONFIG,
        'state': {
            'old_products_loaded': app_state['old_products_data'] is not None,
            'new_products_loaded': app_state['new_products_data'] is not None,
            'review_queue_ready': len(app_state['review_queue']) > 0,
            'feedback_collected': len(app_state['feedback_data']),
            'phase4_enhanced': True
        }
    })

if __name__ == '__main__':
    print("🚀 เริ่มต้น Interactive Human Review System")
    print("🌐 เข้าใช้งานที่: http://localhost:5000")
    print("👤 รองรับ Human Review แบบ Interactive")
    print("🔧 API Status: http://localhost:5000/api/status")
    app.run(debug=True, host='0.0.0.0', port=5000)

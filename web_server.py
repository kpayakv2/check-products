#!/usr/bin/env python3
"""
Web Server สำหรับ Human-in-the-Loop Product Deduplication
รองรับ Interactive Human Review
ปรับปรุงแล้วให้ใช้ Fresh Architecture และ existing components
"""

from flask import Flask, request, jsonify, send_from_directory, render_template
import json
import os
import pandas as pd
from werkzeug.utils import secure_filename
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

# 🔗 ใช้ Fresh Architecture components ที่มีอยู่แล้ว
FRESH_ARCHITECTURE_AVAILABLE = False
logger = logging.getLogger(__name__)

try:
    from fresh_architecture import ProductMatcher, ProductSimilarityPipeline, Config
    from fresh_implementations import ComponentFactory
    from human_feedback_system import ProductDeduplicationSystem, FeedbackType
    from main import Phase4Config, enhance_results, generate_performance_report, create_enhanced_pipeline
    FRESH_ARCHITECTURE_AVAILABLE = True
    logger.info("Fresh Architecture modules loaded successfully")
except ImportError as e:
    logger.warning(f"Fresh Architecture not available: {e}")
    logger.info("Using basic similarity calculation as fallback")

# 🔄 Import shared utilities to eliminate code duplication
try:
    from utils.product_data_utils import (
        extract_product_names, 
        classify_products, 
        calculate_simple_similarity,
        ThresholdConfig,
        ColumnNames
    )
    SHARED_UTILS_AVAILABLE = True
    logger.info("Shared utilities loaded successfully")
except ImportError as e:
    logger.warning(f"Shared utilities not available: {e}")
    SHARED_UTILS_AVAILABLE = False
    
    # Fallback constants only if shared utils unavailable
    class ThresholdConfig:
        HIGH_SIMILARITY = 0.8
        LOW_SIMILARITY = 0.3   
        MAX_CONFIDENCE = 0.95
        HIGH_CONFIDENCE = 0.8

    class ColumnNames:
        THAI_COLUMNS = ['รายการ', 'ชื่อสินค้า']
        ENGLISH_COLUMNS = ['name', 'product_name']

# 🗄️ Import Global Model Cache
try:
    from model_cache_manager import (
        get_global_cache, 
        create_cached_model, 
        clear_model_cache, 
        get_cache_stats
    )
    MODEL_CACHE_AVAILABLE = True
    logger.info("Global Model Cache loaded successfully")
except ImportError as e:
    logger.warning(f"Model Cache not available: {e}")
    MODEL_CACHE_AVAILABLE = False

# Constants and Configuration
class Messages:
    NO_FILE = 'ไม่พบไฟล์'
    NO_FILE_SELECTED = 'ไม่ได้เลือกไฟล์'
    INVALID_FILE_TYPE = 'รองรับเฉพาะไฟล์ .csv, .xlsx, .xls'
    UPLOAD_BOTH_FILES = 'กรุณาอัปโหลดไฟล์ทั้งสองไฟล์ก่อน'
    CANNOT_READ_FILE = 'ไม่สามารถอ่านไฟล์ได้'

# Configuration เดียวกับ main_phase4.py
WEB_CONFIG = {
    'model_type': 'tfidf',  # choices: 'tfidf', 'sentence-bert', 'optimized-tfidf', 'mock'
    'similarity_method': 'cosine',  # choices: 'cosine', 'dot_product'
    'threshold': 0.75,  # เพิ่มจาก 60% เป็น 75% เพื่อลดจำนวนที่ต้องตรวจสอบ
    'top_k': 10,  # เดียวกับ main_phase4.py default
}

# Custom Exceptions
class PipelineError(Exception):
    """Custom exception for ML pipeline errors."""
    pass

class ValidationError(Exception):
    """Custom exception for input validation errors."""
    pass

@dataclass
class AppState:
    """Application state management with type safety."""
    old_products_file: Optional[str] = None
    new_products_file: Optional[str] = None
    old_products_data: Optional[pd.DataFrame] = None
    new_products_data: Optional[pd.DataFrame] = None
    analysis_results: Optional[Dict[str, Any]] = None
    review_queue: List[Dict[str, Any]] = field(default_factory=list)
    feedback_data: List[Dict[str, Any]] = field(default_factory=list)
    current_review_index: int = 0
    # Add missing fields that are used throughout the code
    unique_products: Optional[List[Dict[str, Any]]] = None
    duplicate_check_needed: Optional[List[Dict[str, Any]]] = None
    excluded_duplicates: Optional[List[Dict[str, Any]]] = None  # New field for auto-excluded items
    
    def reset(self) -> None:
        """Reset all state to initial values."""
        self.__init__()
    
    def has_data(self) -> bool:
        """Check if both datasets are loaded."""
        return (self.old_products_data is not None and 
                self.new_products_data is not None)

app = Flask(__name__, template_folder='web')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create application state instance
app_state = AppState()

# Configuration Classes
class PipelineConfig:
    """Configuration wrapper for ML pipeline creation."""
    
    def __init__(self, web_config: Dict[str, Any]):
        self.model = web_config.get('model_type', 'tfidf')
        self.similarity = web_config.get('similarity_method', 'cosine')
        self.threshold = web_config.get('threshold', 0.6)
        self.top_k = web_config.get('top_k', 10)
        self.enhanced = True
        self.track_performance = True
        self.include_metadata = True
        self.confidence_scores = True
        self.export_report = False

# Helper Functions
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}

def create_ml_pipeline(config: PipelineConfig) -> Tuple[Any, Any]:
    """Create ML pipeline from configuration with caching."""
    if not FRESH_ARCHITECTURE_AVAILABLE:
        raise PipelineError("Fresh Architecture components not available")
    
    try:
        # ใช้ Global Model Cache ถ้ามี
        if MODEL_CACHE_AVAILABLE:
            logger.info(f"🗄️ Using Global Model Cache for model: {config.model}")
            
            # สร้าง config dict สำหรับ cache key
            cache_config = {
                'model_type': config.model,
                'similarity_method': config.similarity, 
                'threshold': config.threshold,
                'top_k': config.top_k
            }
            
            # ลองดึง model จาก cache
            cached_model = create_cached_model(config.model, cache_config)
            
            if cached_model:
                # สร้าง ProductMatcher wrapper สำหรับ cached model
                from fresh_architecture import ProductMatcher
                from fresh_implementations import CosineSimilarityCalculator
                
                # สร้าง similarity calculator
                similarity_calc = CosineSimilarityCalculator()
                
                # สร้าง ProductMatcher ที่ wrap cached model
                matcher = ProductMatcher(
                    embedding_model=cached_model,
                    similarity_calculator=similarity_calc
                )
                
                # สร้าง system config
                system_config_dict = {
                    'similarity_threshold': config.threshold,
                    'top_k': config.top_k,
                    'model_type': config.model
                }
                
                # Mock system config object
                class MockSystemConfig:
                    def __init__(self, config_dict):
                        for key, value in config_dict.items():
                            setattr(self, key, value)
                
                system_config = MockSystemConfig(system_config_dict)
                
                logger.info(f"✅ Using cached model with ProductMatcher wrapper: {config.model}")
                return matcher, system_config
        
        # Fallback: ใช้ original pipeline creation
        pipeline, system_config = create_enhanced_pipeline(config)
        if not pipeline or not hasattr(pipeline, 'product_matcher'):
            raise PipelineError("Invalid pipeline structure")
        
        logger.info(f"Pipeline created successfully with model: {config.model}")
        return pipeline.product_matcher, system_config
    
    except Exception as e:
        logger.error(f"Failed to create ML pipeline: {e}")
        raise PipelineError(f"Pipeline creation failed: {e}")

def find_unique_new_products(state: AppState) -> Tuple[List[Dict], List[Dict]]:
    """Find unique new products that don't duplicate existing ones."""
    if not state.has_data():
        logger.warning("No product data available for analysis")
        return [], []
    
    try:
        # Create pipeline configuration
        config = PipelineConfig(WEB_CONFIG.copy())
        matcher, system_config = create_ml_pipeline(config)
        
        # Extract product names using shared utility
        old_products = extract_product_names(state.old_products_data)
        new_products = extract_product_names(state.new_products_data)
        
        logger.info(f"Processing {len(new_products)} new products vs {len(old_products)} old products")
        
        # Process products using ML pipeline
        matches = matcher.find_matches(
            query_products=new_products,
            reference_products=old_products
        )
        
        # Classify results using shared utility
        unique_products, duplicate_check_needed, excluded_duplicates = classify_products(
            matches, new_products, system_config.similarity_threshold
        )
        
        return unique_products, duplicate_check_needed, excluded_duplicates
        
    except PipelineError:
        raise  # Re-raise pipeline errors
    except Exception as e:
        logger.error(f"Error in find_unique_new_products: {e}")
        raise PipelineError(f"Product analysis failed: {e}")

def create_human_review_queue(state: AppState) -> List[Dict[str, Any]]:
    """สร้างรายการสำหรับ Human Review (เฉพาะสินค้าที่ต้องตรวจสอบ)"""
    # ใช้ผลลัพธ์จาก find_unique_new_products
    if not hasattr(state, 'duplicate_check_needed') or state.duplicate_check_needed is None or not state.duplicate_check_needed:
        print("❌ ไม่มีสินค้าที่ต้องตรวจสอบ หรือยังไม่ได้วิเคราะห์")
        return []
    
    review_queue = []
    for i, item in enumerate(state.duplicate_check_needed):
        review_item = {
            'id': f"review_{i+1}",
            'new_product': item['สินค้าใหม่'],
            'old_product': item['สินค้าเก่าที่คล้ายที่สุด'],
            'similarity': float(item.get('ความคล้าย_%', '0%').replace('%', '')) / 100.0,
            'confidence': min(0.95, float(item.get('ความคล้าย_%', '0%').replace('%', '')) / 100.0 + 0.2),
            'ml_prediction': "similar" if float(item.get('ความคล้าย_%', '0%').replace('%', '')) > 50 else "different",
            'status': 'pending',
            'reason': item['คำแนะนำ']
        }
        review_queue.append(review_item)
    
    print(f"📋 สร้างคิวสำหรับ Human Review: {len(review_queue)} รายการ")
    return review_queue

def merge_human_feedback_with_review_queue(review_queue, feedback_data):
    """รวม human feedback เข้ากับ review queue เพื่อสร้างข้อมูลสำหรับ export"""
    if not feedback_data:
        logger.info("ไม่มี human feedback สำหรับการรวม")
        return review_queue
    
    # สร้าง lookup dict สำหรับ feedback โดยใช้คู่ old_product, new_product
    feedback_lookup = {}
    for feedback in feedback_data:
        key = (feedback.get('old_product', '').strip(), feedback.get('new_product', '').strip())
        feedback_lookup[key] = feedback
    
    # รวม feedback เข้ากับ review_queue
    enhanced_queue = []
    for item in review_queue:
        old_product = item.get('old_product', '').strip()
        new_product = item.get('new_product', '').strip()
        feedback_key = (old_product, new_product)
        
        # ค้นหา human feedback ที่ตรงกัน
        human_feedback = feedback_lookup.get(feedback_key, {})
        
        # เพิ่มข้อมูล human feedback เข้าไป
        enhanced_item = item.copy()
        enhanced_item.update({
            'การตัดสินใจของมนุษย์': human_feedback.get('human_feedback', 'ยังไม่ตัดสินใจ'),
            'ความคิดเห็นเพิ่มเติม': human_feedback.get('comments', ''),
            'ผู้ตรวจสอบ': human_feedback.get('reviewer', ''),
            'วันที่ตรวจสอบ': human_feedback.get('timestamp', ''),
            'สถานะการตรวจสอบ': 'เสร็จแล้ว' if human_feedback else 'ยังไม่ได้ตรวจสอบ'
        })
        
        enhanced_queue.append(enhanced_item)
    
    logger.info(f"รวม human feedback สำเร็จ: {len([f for f in enhanced_queue if f.get('การตัดสินใจของมนุษย์', 'ยังไม่ตัดสินใจ') != 'ยังไม่ตัดสินใจ'])} จาก {len(enhanced_queue)} รายการ")
    return enhanced_queue

def enhance_results_phase4(matches, feedback_data):
    """Wrapper for main_phase4.enhance_results with Thai column formatting and human feedback."""
    if not FRESH_ARCHITECTURE_AVAILABLE:
        logger.warning("Fresh Architecture not available, using basic enhancement")
        # รวม human feedback แม้ว่าจะไม่มี Fresh Architecture
        return merge_human_feedback_with_review_queue(matches, feedback_data)
    
    try:
        # Create Phase4Config for enhancement
        config = Phase4Config()
        config.include_confidence_scores = True
        config.include_metadata = True
        
        # Use the original enhance_results from main.py
        enhanced_matches = enhance_results(matches, config)
        
        # Convert to Thai column format for web display
        thai_enhanced_matches = []
        for i, match in enumerate(enhanced_matches):
            thai_match = {
                'รหัส': match.get('id', f"review_{i+1}"),
                'สินค้าใหม่': match.get('query_product', match.get('new_product', '')),
                'สินค้าเก่าที่คล้ายที่สุด': match.get('matched_product', match.get('old_product', '')),
                'ความคล้าย_%': f"{match.get('similarity_score', 0)*100:.1f}%",
                'ความมั่นใจ_AI': f"{match.get('confidence_score', 0)*100:.1f}%",
                'คาดการณ์_AI': 'คล้าย' if match.get('ml_prediction') == 'similar' else 'แตกต่าง',
                'สถานะระบบ': 'รอตรวจสอบ' if match.get('status') == 'pending' else match.get('status', 'รอตรวจสอบ'),
                'คำแนะนำเบื้องต้น': match.get('reason', ''),
                'อันดับ': match.get('match_rank', i + 1),
                'เวลาประมวลผล': match.get('processing_timestamp', time.time()),
                'เวอร์ชันระบบ': match.get('processor_version', 'web_phase4_enhanced'),
                'วิธีการวิเคราะห์': match.get('method', 'fresh_architecture'),
                'คะแนนความมั่นใจ': match.get('confidence_score', 0),
                'ระดับความมั่นใจ': _translate_confidence_level(match.get('confidence_level', 'low'))
            }
            thai_enhanced_matches.append(thai_match)
        
        # 🚨 สำคัญ! รวม human feedback เข้าไป
        final_enhanced_results = merge_human_feedback_with_review_queue(thai_enhanced_matches, feedback_data)
        
        return final_enhanced_results
        
    except Exception as e:
        logger.error(f"Error in enhance_results_phase4: {e}")
        # Fallback: ใช้การรวม feedback อย่างง่าย
        return merge_human_feedback_with_review_queue(matches, feedback_data)

def _translate_confidence_level(level: str) -> str:
    """Translate confidence level to Thai."""
    translations = {
        'high': 'สูง',
        'medium': 'ปานกลาง', 
        'low': 'ต่ำ'
    }
    return translations.get(level, 'ต่ำ')

@app.route('/')
def index():
    """Product deduplication main interface."""
    try:
        return render_template('product_deduplication.html')
    except Exception as e:
        return f"<h1>Template Error</h1><p>{str(e)}</p>"

@app.route('/human-review')
def human_review():
    """ใช้ web/human_review.html template สำหรับ human review interface"""
    return send_from_directory('web', 'human_review.html')

# API Endpoints เพิ่มเติมสำหรับ Human Review

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process product files with validation."""
    try:
        # Input validation
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': Messages.NO_FILE})
        
        file = request.files['file']
        file_type = request.form.get('type')
        
        if not file_type or file_type not in ['old', 'new']:
            return jsonify({'success': False, 'message': 'Invalid file type'})
        
        if file.filename == '':
            return jsonify({'success': False, 'message': Messages.NO_FILE_SELECTED})
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'message': Messages.INVALID_FILE_TYPE})
        
        # Process file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_type}_{filename}")
        file.save(filepath)
        
        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filepath, encoding='utf-8-sig')
            else:
                df = pd.read_excel(filepath)
        except Exception as e:
            logger.error(f"Failed to read file {filename}: {e}")
            return jsonify({'success': False, 'message': f'{Messages.CANNOT_READ_FILE}: {str(e)}'})
        
        # Validate data
        if df.empty:
            return jsonify({'success': False, 'message': 'File is empty'})
        
        # Update state safely
        if file_type == 'old':
            app_state.old_products_file = filepath
            app_state.old_products_data = df
        else:
            app_state.new_products_file = filepath
            app_state.new_products_data = df
        
        logger.info(f"Successfully uploaded {file_type} products file: {len(df)} records")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'count': len(df),
            'columns': list(df.columns)
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze products for duplicates."""
    try:
        # Validate state
        if not app_state.has_data():
            return jsonify({'success': False, 'message': Messages.UPLOAD_BOTH_FILES})
        
        # Perform analysis
        unique_products, duplicate_check_needed, excluded_duplicates = find_unique_new_products(app_state)
        
        # Update state first
        app_state.unique_products = unique_products
        app_state.duplicate_check_needed = duplicate_check_needed
        app_state.excluded_duplicates = excluded_duplicates
        
        # Create review queue
        app_state.review_queue = create_human_review_queue(app_state)
        
        # Calculate statistics
        old_count = len(app_state.old_products_data)
        new_count = len(app_state.new_products_data)
        unique_count = len(unique_products)
        duplicate_check_count = len(duplicate_check_needed)
        excluded_count = len(excluded_duplicates)
        
        # Update state
        app_state.analysis_results = {
            'old_count': old_count,
            'new_count': new_count,
            'unique_count': unique_count,
            'duplicate_check_count': duplicate_check_count,
            'excluded_count': excluded_count,
            'unique_products': unique_products,
            'duplicate_check_needed': duplicate_check_needed,
            'excluded_duplicates': excluded_duplicates
        }
        
        logger.info(f"Analysis complete: {unique_count} unique, {duplicate_check_count} need review, {excluded_count} auto-excluded")
        
        return jsonify({
            'success': True,
            'old_count': old_count,
            'new_count': new_count,
            'unique_count': unique_count,
            'duplicate_check_count': duplicate_check_count,
            'excluded_count': excluded_count,
            'pending_review': len(app_state.review_queue),
            'summary': f"จาก {new_count} สินค้าใหม่: {unique_count} ไม่ซ้ำ, {duplicate_check_count} ต้องตรวจสอบ, {excluded_count} ซ้ำมาก (ตัดออกอัตโนมัติ)"
        })
        
    except PipelineError as e:
        logger.error(f"Pipeline error: {e}")
        return jsonify({'success': False, 'message': f"Pipeline error: {str(e)}"})
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/get-review-queue')
def get_review_queue():
    return jsonify({
        'success': True,
        'queue': app_state.review_queue
    })

@app.route('/save-feedback', methods=['POST'])
def save_feedback():
    """บันทึก Human Feedback โดยใช้ ProductDeduplicationSystem"""
    try:
        feedback = request.json
        
        # Validate input
        if not feedback:
            return jsonify({'success': False, 'message': 'ไม่มีข้อมูล feedback'})
        
        required_fields = ['old_product', 'new_product', 'human_feedback']
        for field in required_fields:
            if field not in feedback:
                return jsonify({'success': False, 'message': f'ขาดข้อมูล: {field}'})
        
        # เพิ่ม timestamp และข้อมูลเพิ่มเติม
        feedback['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        feedback['reviewer'] = feedback.get('reviewer', 'anonymous')
        feedback['comments'] = feedback.get('comments', '')
        
        # 🛡️ บันทึกลง local file เพื่อป้องกันข้อมูลหาย
        try:
            backup_feedback_to_file(feedback)
        except Exception as backup_error:
            logger.warning(f"Failed to backup feedback to file: {backup_error}")
        
        # ใช้ human_feedback_system.py
        if FRESH_ARCHITECTURE_AVAILABLE:
            try:
                # Create deduplication system instance with configurable embedding model
                # 🚀 เปลี่ยน embedding_model_type ตามความต้องการ:
                # - "mock": เร็วมาก เหมาะสำหรับ development/testing
                # - "optimized-tfidf": สมดุลระหว่างเร็วและแม่นยำ  
                # - "sentence-bert": แม่นยำสูงสุด เหมาะสำหรับ production
                
                embedding_model_type = "mock"  # 🔧 เปลี่ยนตรงนี้เพื่อเลือก model
                
                dedup_system = ProductDeduplicationSystem(
                    similarity_threshold=0.8,
                    embedding_model_type=embedding_model_type
                )
                
                # Convert feedback to proper format
                feedback_type = FeedbackType(feedback.get('human_feedback', 'uncertain'))
                
                result = dedup_system.record_human_feedback(
                    product1=feedback['old_product'],
                    product2=feedback['new_product'], 
                    similarity=feedback.get('similarity', 0.0),
                    human_decision=feedback_type,
                    reviewer=feedback.get('reviewer', 'anonymous'),
                    comments=feedback.get('comments', '')
                )
                
                if result:
                    app_state.feedback_data.append(feedback)  # Keep for web compatibility
                    logger.info(f"✅ Feedback saved via ProductDeduplicationSystem: {feedback_type}")
                    return jsonify({
                        'success': True, 
                        'message': 'บันทึก feedback สำเร็จ',
                        'feedback_type': feedback_type.value,
                        'total_feedback': len(app_state.feedback_data),
                        'reviewer': feedback['reviewer'],
                        'timestamp': feedback['timestamp']
                    })
                else:
                    logger.warning("ProductDeduplicationSystem failed, using fallback")
            
            except Exception as system_error:
                logger.error(f"ProductDeduplicationSystem error: {system_error}")
                # Continue to fallback
        
        # Fallback: simple storage
        app_state.feedback_data.append(feedback)
        logger.info(f"✅ Feedback saved via fallback: {feedback.get('human_feedback', 'unknown')}")
        return jsonify({
            'success': True, 
            'message': 'บันทึก feedback สำเร็จ (fallback mode)',
            'total_feedback': len(app_state.feedback_data),
            'reviewer': feedback['reviewer'],
            'timestamp': feedback['timestamp']
        })
        
    except Exception as e:
        logger.error(f"save_feedback error: {e}")
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/feedback/status')
def api_feedback_status():
    """ตรวจสอบสถานะ human feedback ทั้งหมด"""
    try:
        # รวม feedback จาก memory และ backup file
        memory_feedback = app_state.feedback_data
        backup_feedback = load_feedback_from_backup()
        
        # หาข้อมูลที่ไม่ซ้ำกัน
        all_feedback = {}
        
        # เพิ่ม backup feedback ก่อน
        for fb in backup_feedback:
            key = (fb.get('old_product', ''), fb.get('new_product', ''))
            all_feedback[key] = fb
        
        # เขียนทับด้วย memory feedback (ล่าสุดกว่า)
        for fb in memory_feedback:
            key = (fb.get('old_product', ''), fb.get('new_product', ''))
            all_feedback[key] = fb
        
        unique_feedback = list(all_feedback.values())
        
        # สถิติ
        feedback_stats = {
            'total_feedback': len(unique_feedback),
            'similar_decisions': len([f for f in unique_feedback if f.get('human_feedback') == 'similar']),
            'different_decisions': len([f for f in unique_feedback if f.get('human_feedback') == 'different']),
            'duplicate_decisions': len([f for f in unique_feedback if f.get('human_feedback') == 'duplicate']),
            'uncertain_decisions': len([f for f in unique_feedback if f.get('human_feedback') == 'uncertain']),
            'memory_feedback_count': len(memory_feedback),
            'backup_feedback_count': len(backup_feedback),
            'latest_feedback_time': max([f.get('timestamp', '') for f in unique_feedback], default='')
        }
        
        return jsonify({
            'success': True,
            'feedback_status': feedback_stats,
            'data_sources': {
                'memory_available': len(memory_feedback) > 0,
                'backup_file_available': len(backup_feedback) > 0,
                'data_integrity': 'OK' if len(unique_feedback) > 0 else 'NO_DATA'
            },
            'recommendations': [
                "✅ Feedback data is being backed up automatically",
                f"📊 {feedback_stats['total_feedback']} human decisions recorded",
                "💾 Data is preserved across server restarts"
            ] if len(unique_feedback) > 0 else [
                "⚠️ No human feedback recorded yet",
                "🚀 Start reviewing products to build training data"
            ]
        })
        
    except Exception as e:
        logger.error(f"api_feedback_status error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/review-summary')
def api_review_summary():
    """สรุปผลการตัดสินใจของมนุษย์สำหรับแสดงใน web UI ก่อนการส่งออก"""
    try:
        # รวม feedback data จาก memory และ backup file
        all_feedback_data = []
        try:
            backup_feedback = load_feedback_from_backup()
            
            # รวมข้อมูลโดยไม่ให้ซ้ำกัน
            feedback_dict = {}
            
            # เพิ่ม backup feedback ก่อน
            for fb in backup_feedback:
                key = (fb.get('old_product', ''), fb.get('new_product', ''))
                feedback_dict[key] = fb
            
            # เขียนทับด้วย memory feedback (ล่าสุดกว่า)
            for fb in app_state.feedback_data:
                key = (fb.get('old_product', ''), fb.get('new_product', ''))
                feedback_dict[key] = fb
            
            all_feedback_data = list(feedback_dict.values())
            
        except Exception as e:
            logger.warning(f"Failed to load backup feedback: {e}")
            all_feedback_data = app_state.feedback_data
        
        # คำนวณสถิติการตัดสินใจ
        total_items = len(app_state.review_queue) if app_state.review_queue else 0
        reviewed_items = len(all_feedback_data)
        pending_items = total_items - reviewed_items
        
        # แยกสถิติตามการตัดสินใจ
        similar_count = len([f for f in all_feedback_data if f.get('human_feedback') == 'similar'])
        different_count = len([f for f in all_feedback_data if f.get('human_feedback') == 'different'])
        duplicate_count = len([f for f in all_feedback_data if f.get('human_feedback') == 'duplicate'])
        uncertain_count = len([f for f in all_feedback_data if f.get('human_feedback') == 'uncertain'])
        
        # คำนวณความแม่นยำของ AI vs Human
        correct_predictions = 0
        total_predictions = 0
        
        for feedback in all_feedback_data:
            if 'ml_prediction' in feedback and 'human_feedback' in feedback:
                total_predictions += 1
                ml_pred = feedback.get('ml_prediction', '').lower()
                human_decision = feedback.get('human_feedback', '').lower()
                
                # แปลง human decision ให้ตรงกับ ml_prediction format
                if human_decision == 'similar' and ml_pred == 'similar':
                    correct_predictions += 1
                elif human_decision == 'different' and ml_pred == 'different':
                    correct_predictions += 1
                elif human_decision == 'duplicate' and ml_pred == 'similar':
                    # duplicate ถือว่าเป็น similar สำหรับ AI accuracy
                    correct_predictions += 1
        
        ai_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        # สร้างรายละเอียดการตัดสินใจ
        decision_details = []
        for i, feedback in enumerate(all_feedback_data):
            detail = {
                'index': i + 1,
                'old_product': feedback.get('old_product', ''),
                'new_product': feedback.get('new_product', ''),
                'human_decision': feedback.get('human_feedback', 'ยังไม่ตัดสินใจ'),
                'ml_prediction': feedback.get('ml_prediction', 'ไม่ทราบ'),
                'similarity': feedback.get('similarity', 0),
                'comments': feedback.get('comments', ''),
                'reviewer': feedback.get('reviewer', 'anonymous'),
                'timestamp': feedback.get('timestamp', ''),
                'agreement': 'ตรงกัน' if (
                    feedback.get('human_feedback', '').lower() == 'similar' and feedback.get('ml_prediction', '').lower() == 'similar'
                ) or (
                    feedback.get('human_feedback', '').lower() == 'different' and feedback.get('ml_prediction', '').lower() == 'different'  
                ) or (
                    feedback.get('human_feedback', '').lower() == 'duplicate' and feedback.get('ml_prediction', '').lower() == 'similar'
                ) else 'ไม่ตรงกัน'
            }
            decision_details.append(detail)
        
        # แปลงการตัดสินใจเป็นภาษาไทย
        def translate_decision(decision):
            translations = {
                'similar': 'คล้ายกัน',
                'different': 'แตกต่างกัน',
                'duplicate': 'ซ้ำกัน', 
                'uncertain': 'ไม่แน่ใจ'
            }
            return translations.get(decision.lower(), decision)
        
        # ใส่การแปลงในรายละเอียด
        for detail in decision_details:
            detail['human_decision_thai'] = translate_decision(detail['human_decision'])
            detail['ml_prediction_thai'] = translate_decision(detail['ml_prediction']) if detail['ml_prediction'] != 'ไม่ทราบ' else 'ไม่ทราบ'
        
        return jsonify({
            'success': True,
            'review_summary': {
                'total_items': total_items,
                'reviewed_items': reviewed_items,
                'pending_items': pending_items,
                'completion_percentage': round((reviewed_items / total_items * 100), 1) if total_items > 0 else 0,
                
                'decision_statistics': {
                    'similar_count': similar_count,
                    'different_count': different_count,
                    'duplicate_count': duplicate_count,
                    'uncertain_count': uncertain_count,
                    'similar_percentage': round((similar_count / reviewed_items * 100), 1) if reviewed_items > 0 else 0,
                    'different_percentage': round((different_count / reviewed_items * 100), 1) if reviewed_items > 0 else 0,
                    'duplicate_percentage': round((duplicate_count / reviewed_items * 100), 1) if reviewed_items > 0 else 0,
                    'uncertain_percentage': round((uncertain_count / reviewed_items * 100), 1) if reviewed_items > 0 else 0
                },
                
                'ai_performance': {
                    'accuracy_percentage': round(ai_accuracy, 1),
                    'total_predictions': total_predictions,
                    'correct_predictions': correct_predictions,
                    'status': 'ดีเยี่ยม' if ai_accuracy >= 80 else 'ดี' if ai_accuracy >= 60 else 'ปรับปรุงได้'
                },
                
                'decision_details': decision_details,
                
                'export_readiness': {
                    'can_export': reviewed_items > 0,
                    'recommendation': 'พร้อมส่งออก' if reviewed_items >= total_items * 0.8 else 'ควรตรวจสอบเพิ่มเติม',
                    'missing_reviews': max(0, total_items - reviewed_items)
                }
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        logger.error(f"api_review_summary error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/prepare-export', methods=['POST'])
def api_prepare_export():
    """ประมวลผลและเตรียมข้อมูลก่อนการส่งออก รวมถึงการแสดงสถิติและสรุป"""
    try:
        # ตรวจสอบว่ามีข้อมูลที่จำเป็นหรือไม่
        if not hasattr(app_state, 'unique_products') or not app_state.unique_products:
            return jsonify({'success': False, 'message': 'ยังไม่ได้วิเคราะห์ข้อมูล กรุณาวิเคราะห์ก่อน'})
        
        # รวม feedback data จาก memory และ backup file
        all_feedback_data = []
        try:
            backup_feedback = load_feedback_from_backup()
            
            # รวมข้อมูลโดยไม่ให้ซ้ำกัน
            feedback_dict = {}
            
            # เพิ่ม backup feedback ก่อน
            for fb in backup_feedback:
                key = (fb.get('old_product', ''), fb.get('new_product', ''))
                feedback_dict[key] = fb
            
            # เขียนทับด้วย memory feedback (ล่าสุดกว่า)
            for fb in app_state.feedback_data:
                key = (fb.get('old_product', ''), fb.get('new_product', ''))
                feedback_dict[key] = fb
            
            all_feedback_data = list(feedback_dict.values())
            
        except Exception as e:
            logger.warning(f"Failed to load backup feedback: {e}")
            all_feedback_data = app_state.feedback_data
        
        # ประมวลผลข้อมูลขั้นสุดท้าย
        logger.info("🔄 เริ่มประมวลผลข้อมูลสำหรับการส่งออก...")
        
        # Enhanced results with human feedback
        enhanced_results = []
        if all_feedback_data:
            enhanced_results = enhance_results_phase4(app_state.review_queue, all_feedback_data)
        
        # คำนวณสถิติสำหรับการส่งออก
        export_statistics = {
            'total_old_products': len(app_state.old_products_data) if app_state.old_products_data is not None else 0,
            'total_new_products': len(app_state.new_products_data) if app_state.new_products_data is not None else 0,
            'unique_new_products': len(app_state.unique_products),
            'need_review_products': len(app_state.duplicate_check_needed) if app_state.duplicate_check_needed else 0,
            'auto_excluded_duplicates': len(app_state.excluded_duplicates) if app_state.excluded_duplicates else 0,
            'human_reviewed_items': len(all_feedback_data),
            'enhanced_results_count': len(enhanced_results)
        }
        
        # คำนวณเปอร์เซ็นต์
        export_statistics['unique_percentage'] = round(
            (export_statistics['unique_new_products'] / export_statistics['total_new_products'] * 100), 1
        ) if export_statistics['total_new_products'] > 0 else 0
        
        export_statistics['review_completion_percentage'] = round(
            (export_statistics['human_reviewed_items'] / export_statistics['need_review_products'] * 100), 1
        ) if export_statistics['need_review_products'] > 0 else 100
        
        # สถิติการตัดสินใจของมนุษย์
        human_decisions = {
            'similar': len([f for f in all_feedback_data if f.get('human_feedback') == 'similar']),
            'different': len([f for f in all_feedback_data if f.get('human_feedback') == 'different']),
            'duplicate': len([f for f in all_feedback_data if f.get('human_feedback') == 'duplicate']),
            'uncertain': len([f for f in all_feedback_data if f.get('human_feedback') == 'uncertain'])
        }
        
        # สถิติความแม่นยำของ AI
        ai_human_agreement = 0
        total_comparisons = 0
        
        for feedback in all_feedback_data:
            if 'ml_prediction' in feedback and 'human_feedback' in feedback:
                total_comparisons += 1
                ml_pred = feedback.get('ml_prediction', '').lower()
                human_decision = feedback.get('human_feedback', '').lower()
                
                # Check for agreement including duplicate = similar
                if (ml_pred == human_decision) or (ml_pred == 'similar' and human_decision == 'duplicate'):
                    ai_human_agreement += 1
        
        ai_accuracy = (ai_human_agreement / total_comparisons * 100) if total_comparisons > 0 else 0
        
        # 🚀 ขั้นตอนประมวลผลและคัดแยกสินค้า
        logger.info("🔄 เริ่มกระบวนการคัดแยกสินค้าตาม human feedback...")
        
        product_classification = classify_products_by_human_feedback(
            app_state.unique_products,
            app_state.review_queue,
            all_feedback_data
        )
        
        # อัปเดต export statistics ด้วยข้อมูลการคัดแยก
        export_statistics.update({
            'approved_for_import': product_classification['classification_stats']['approved_for_import_count'],
            'rejected_duplicates': product_classification['classification_stats']['rejected_duplicates_count'],
            'pending_review_items': product_classification['classification_stats']['pending_review_count'],
            'approval_rate': product_classification['classification_stats']['approval_rate'],
            'human_review_coverage': product_classification['classification_stats']['human_review_coverage']
        })
        
        # เตรียมข้อมูลสำหรับแสดงผลใน UI ✨
        export_preview = {
            'files_to_be_generated': [
                {
                    'name': 'approved_products_for_import',
                    'description': 'สินค้าที่อนุมัติให้นำเข้าระบบ (ไฟล์หลัก) ⭐',
                    'count': len(product_classification['approved_for_import']),
                    'purpose': 'นำเข้าระบบได้ทันที',
                    'confidence': 'สูง',
                    'categories': {
                        'unique_products': len([p for p in product_classification['approved_for_import'] if p['category'] == 'unique']),
                        'human_approved': len([p for p in product_classification['approved_for_import'] if p['category'] == 'human_approved']),
                        'similar_approved': len([p for p in product_classification['approved_for_import'] if p['category'] == 'similar_approved'])
                    }
                },
                {
                    'name': 'rejected_duplicates',
                    'description': 'สินค้าที่ปฏิเสธเนื่องจากซ้ำแบบ exact',
                    'count': len(product_classification['rejected_duplicates']),
                    'purpose': 'บันทึกเพื่อไม่ให้นำเข้าซ้ำ',
                    'confidence': 'สูงมาก',
                    'categories': {
                        'exact_duplicates': len([p for p in product_classification['rejected_duplicates'] if p['category'] == 'exact_duplicate'])
                    }
                },
                {
                    'name': 'pending_review_products',
                    'description': 'สินค้าที่ต้องตรวจสอบเพิ่มเติม',
                    'count': len(product_classification['pending_review']),
                    'purpose': 'รอการตัดสินใจเพิ่มเติม',
                    'confidence': 'ต่ำ',
                    'action_needed': True
                },
                {
                    'name': 'analysis_summary',
                    'description': 'สรุปการวิเคราะห์และคัดแยก',
                    'count': 1,
                    'purpose': 'รายงานผลการทำงาน'
                }
            ],
            'conditional_files': []
        }
        
        # เพิ่มไฟล์ human feedback ถ้ามีการตัดสินใจ
        if all_feedback_data:
            export_preview['conditional_files'].append({
                'name': 'human_feedback_results',
                'description': 'ผลการตัดสินใจของมนุษย์ พร้อมข้อมูลการเรียนรู้สำหรับ AI',
                'count': len(all_feedback_data),
                'purpose': 'Training data สำหรับปรับปรุง AI'
            })
        
        # สร้างคำแนะนำที่ชาญฉลาด
        recommendations = []
        
        if len(product_classification['approved_for_import']) > 0:
            unique_count = len([p for p in product_classification['approved_for_import'] if p['category'] == 'unique'])
            different_count = len([p for p in product_classification['approved_for_import'] if p['category'] == 'human_approved'])
            similar_count = len([p for p in product_classification['approved_for_import'] if p['category'] == 'similar_approved'])
            
            recommendations.append(f"✅ พร้อมนำเข้า {len(product_classification['approved_for_import'])} สินค้า ({product_classification['classification_stats']['approval_rate']}% ของที่ตรวจสอบแล้ว)")
            if similar_count > 0:
                recommendations.append(f"   - รวม {similar_count} สินค้าที่คล้ายกันแต่อนุมัติให้นำเข้า")
        
        if len(product_classification['rejected_duplicates']) > 0:
            exact_count = len([p for p in product_classification['rejected_duplicates'] if p['category'] == 'exact_duplicate'])
            recommendations.append(f"❌ ปฏิเสธ {len(product_classification['rejected_duplicates'])} สินค้าที่ซ้ำแบบ exact")
            
        if len(product_classification['pending_review']) > 0:
            recommendations.append(f"⏳ ยังต้องตรวจสอบ {len(product_classification['pending_review'])} สินค้า")
            
        recommendations.extend([
            f"📊 มีการตัดสินใจของมนุษย์ {len(all_feedback_data)} รายการ ({export_statistics['human_review_coverage']}% ของที่ต้องตรวจสอบ)",
            f"🤖 AI มีความแม่นยำ {round(ai_accuracy, 1)}% เมื่อเทียบกับมนุษย์",
            "💾 ข้อมูลพร้อมส่งออกในรูปแบบ CSV"
        ])
        
        # แสดง next steps ที่ชาญฉลาด
        next_steps = []
        
        if len(product_classification['approved_for_import']) > 0:
            next_steps.append(f"1. ✅ นำเข้าสินค้า {len(product_classification['approved_for_import'])} รายการที่ได้รับอนุมัติ")
        
        if len(product_classification['pending_review']) > 0:
            next_steps.append(f"2. ⏳ ตรวจสอบสินค้า {len(product_classification['pending_review'])} รายการที่ยังรออนุมัติ")
            
        next_steps.extend([
            "3. 📊 ตรวจสอบสถิติและตัวเลข",
            "4. 💾 กดปุ่ม 'ส่งออก CSV' เพื่อสร้างไฟล์",
            "5. 📁 ดาวน์โหลดไฟล์จากโฟลเดอร์ output/"
        ])
        
        # บันทึกผลการคัดแยกใน app_state เพื่อใช้ใน export
        app_state.product_classification = product_classification
        
        return jsonify({
            'success': True,
            'export_ready': True,
            'processing_complete': True,
            'classification_complete': True,
            'export_statistics': export_statistics,
            'human_decisions': human_decisions,
            'product_classification': {
                'approved_for_import': len(product_classification['approved_for_import']),
                'rejected_duplicates': len(product_classification['rejected_duplicates']),
                'pending_review': len(product_classification['pending_review']),
                'approval_rate': product_classification['classification_stats']['approval_rate'],
                'review_coverage': product_classification['classification_stats']['human_review_coverage']
            },
            'ai_performance': {
                'accuracy_percentage': round(ai_accuracy, 1),
                'total_comparisons': total_comparisons,
                'agreements': ai_human_agreement,
                'status': 'ดีเยี่ยม' if ai_accuracy >= 80 else 'ดี' if ai_accuracy >= 60 else 'ปรับปรุงได้'
            },
            'export_preview': export_preview,
            'recommendations': recommendations,
            'next_steps': next_steps,
            'business_insights': {
                'import_readiness': 'พร้อมนำเข้า' if len(product_classification['approved_for_import']) > 0 else 'ยังไม่พร้อม',
                'duplicate_prevention': f"ป้องกันสินค้าซ้ำ {len(product_classification['rejected_duplicates'])} รายการ",
                'quality_assurance': f"ตรวจสอบคุณภาพ {export_statistics['human_review_coverage']}%"
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        logger.error(f"api_prepare_export error: {e}")
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาดในการเตรียมข้อมูล: {str(e)}'})

def classify_products_by_human_feedback(unique_products, review_queue, all_feedback_data):
    """
    คัดแยกสินค้าตาม human feedback เพื่อตัดสินใจว่าสินค้าไหนสามารถนำเข้าระบบได้
    
    Categories:
    1. approved_for_import: สินค้าที่ได้รับอนุมัติให้นำเข้า (รวม unique + different)
    2. rejected_duplicates: สินค้าที่ถูกปฏิเสธเพราะซ้ำ (similar)
    3. pending_review: สินค้าที่ยังรออนุมัติ (uncertain หรือไม่มี feedback)
    """
    
    # สร้าง lookup dict สำหรับ feedback
    feedback_lookup = {}
    for feedback in all_feedback_data:
        new_product = feedback.get('new_product', '').strip()
        if new_product:
            feedback_lookup[new_product] = feedback
    
    # สินค้าที่นำเข้าได้แน่นอน (unique products)
    approved_for_import = []
    for product in unique_products:
        product_name = str(product).strip() if isinstance(product, str) else str(product.get('สินค้าใหม่', str(product))).strip()
        approved_for_import.append({
            'product_name': product_name,
            'category': 'unique',
            'reason': 'ไม่ซ้ำกับสินค้าเก่า',
            'confidence': 'สูง',
            'source': 'AI_analysis'
        })
    
    # คัดแยกสินค้าที่ผ่าน human review
    rejected_duplicates = []
    pending_review = []
    
    # สร้าง lookup สำหรับ review queue
    review_lookup = {}
    for item in review_queue:
        new_product = item.get('new_product', '').strip()
        if new_product:
            review_lookup[new_product] = item
    
    # ประมวลผลสินค้าที่ผ่าน human review
    for feedback in all_feedback_data:
        new_product = feedback.get('new_product', '').strip()
        human_decision = feedback.get('human_feedback', '').lower()
        old_product = feedback.get('old_product', '')
        
        if not new_product:
            continue
            
        # ข้อมูลเพิ่มเติมจาก review queue
        review_item = review_lookup.get(new_product, {})
        similarity = feedback.get('similarity', review_item.get('similarity', 0))
        
        if human_decision == 'different':
            # มนุษย์ตัดสินใจว่าแตกต่าง → นำเข้าได้
            approved_for_import.append({
                'product_name': new_product,
                'category': 'human_approved',
                'reason': f'มนุษย์ตัดสินใจว่าแตกต่างจาก "{old_product}"',
                'confidence': 'สูง',
                'similarity_score': similarity,
                'reviewer': feedback.get('reviewer', 'anonymous'),
                'review_date': feedback.get('timestamp', ''),
                'comments': feedback.get('comments', ''),
                'source': 'human_review'
            })
            
        elif human_decision == 'duplicate':
            # มนุษย์ตัดสินใจว่าซ้ำแบบ exact → ปฏิเสธอย่างแน่นอน
            rejected_duplicates.append({
                'product_name': new_product,
                'category': 'exact_duplicate',
                'reason': f'มนุษย์ตัดสินใจว่าซ้ำกับ "{old_product}" (duplicate)',
                'confidence': 'สูงมาก',
                'similarity_score': similarity,
                'duplicate_of': old_product,
                'reviewer': feedback.get('reviewer', 'anonymous'),
                'review_date': feedback.get('timestamp', ''),
                'comments': feedback.get('comments', ''),
                'source': 'human_review',
                'duplicate_type': 'exact'
            })
            
        elif human_decision == 'similar':
            # มนุษย์ตัดสินใจว่าคล้าย → นำเข้าได้ (แก้ไขให้อนุมัติ)
            approved_for_import.append({
                'product_name': new_product,
                'category': 'similar_approved',
                'reason': f'มนุษย์ตัดสินใจว่าคล้ายกับ "{old_product}" แต่อนุมัติให้นำเข้า',
                'confidence': 'ปานกลาง',
                'similarity_score': similarity,
                'similar_to': old_product,
                'reviewer': feedback.get('reviewer', 'anonymous'),
                'review_date': feedback.get('timestamp', ''),
                'comments': feedback.get('comments', ''),
                'source': 'human_review',
                'approval_type': 'similar_but_approved'
            })
            
        elif human_decision == 'uncertain':
            # มนุษย์ไม่แน่ใจ → รอการตรวจสอบเพิ่มเติม
            pending_review.append({
                'product_name': new_product,
                'category': 'uncertain',
                'reason': f'มนุษย์ไม่แน่ใจเกี่ยวกับความคล้ายกับ "{old_product}"',
                'confidence': 'ต่ำ',
                'similarity_score': similarity,
                'potentially_similar_to': old_product,
                'reviewer': feedback.get('reviewer', 'anonymous'),
                'review_date': feedback.get('timestamp', ''),
                'comments': feedback.get('comments', ''),
                'source': 'human_review',
                'action_needed': 'ต้องการการตรวจสอบเพิ่มเติม'
            })
    
    # หาสินค้าที่ยังไม่ได้รับการตรวจสอบ
    reviewed_products = set(feedback_lookup.keys())
    
    for item in review_queue:
        new_product = item.get('new_product', '').strip()
        
        if new_product and new_product not in reviewed_products:
            # สินค้าที่ยังไม่ได้ human feedback
            pending_review.append({
                'product_name': new_product,
                'category': 'not_reviewed',
                'reason': f'ยังไม่ได้รับการตรวจสอบจากมนุษย์ (อาจคล้ายกับ "{item.get("old_product", "")}")',
                'confidence': 'ต่ำ',
                'similarity_score': item.get('similarity', 0),
                'potentially_similar_to': item.get('old_product', ''),
                'ml_prediction': item.get('ml_prediction', ''),
                'source': 'AI_analysis',
                'action_needed': 'ต้องการการตรวจสอบจากมนุษย์'
            })
    
    # สถิติ
    classification_stats = {
        'approved_for_import_count': len(approved_for_import),
        'rejected_duplicates_count': len(rejected_duplicates),
        'pending_review_count': len(pending_review),
        'total_processed': len(approved_for_import) + len(rejected_duplicates) + len(pending_review),
        'approval_rate': round((len(approved_for_import) / (len(approved_for_import) + len(rejected_duplicates)) * 100), 1) if (len(approved_for_import) + len(rejected_duplicates)) > 0 else 0,
        'human_review_coverage': round((len(all_feedback_data) / len(review_queue) * 100), 1) if len(review_queue) > 0 else 100
    }
    
    logger.info(f"🔄 Product Classification Complete:")
    logger.info(f"   ✅ Approved for import: {len(approved_for_import)}")
    logger.info(f"   ❌ Rejected duplicates: {len(rejected_duplicates)}")  
    logger.info(f"   ⏳ Pending review: {len(pending_review)}")
    logger.info(f"   📊 Approval rate: {classification_stats['approval_rate']}%")
    
    return {
        'approved_for_import': approved_for_import,
        'rejected_duplicates': rejected_duplicates,
        'pending_review': pending_review,
        'classification_stats': classification_stats
    }

def backup_feedback_to_file(feedback: Dict[str, Any]) -> None:
    """บันทึก human feedback ลง local file เพื่อป้องกันข้อมูลหาย"""
    try:
        backup_dir = 'output'
        os.makedirs(backup_dir, exist_ok=True)
        
        backup_file = os.path.join(backup_dir, 'human_feedback_backup.jsonl')
        
        # เขียนทีละบรรทัด (JSONL format) เพื่อป้องกัน corruption
        with open(backup_file, 'a', encoding='utf-8') as f:
            json.dump(feedback, f, ensure_ascii=False)
            f.write('\n')
        
        logger.debug(f"Backup feedback to {backup_file}")
        
    except Exception as e:
        logger.error(f"Failed to backup feedback: {e}")
        raise

def load_feedback_from_backup() -> List[Dict[str, Any]]:
    """โหลด feedback จาก backup file"""
    try:
        backup_file = os.path.join('output', 'human_feedback_backup.jsonl')
        
        if not os.path.exists(backup_file):
            return []
        
        feedback_list = []
        with open(backup_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        feedback = json.loads(line.strip())
                        feedback_list.append(feedback)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skip invalid feedback line: {e}")
                        continue
        
        logger.info(f"Loaded {len(feedback_list)} feedback items from backup")
        return feedback_list
        
    except Exception as e:
        logger.error(f"Failed to load feedback from backup: {e}")
        return []

def _get_cache_recommendations(stats: dict) -> list:
    """สร้างคำแนะนำจากสถิติ cache"""
    recommendations = []
    
    if stats['memory_utilization'] > 80:
        recommendations.append("⚠️ Memory usage high - consider clearing cache")
    
    if stats['cached_models'] > 3:
        recommendations.append("💡 Multiple models cached - good for performance")
        
    if stats['cached_models'] == 0:
        recommendations.append("🔄 No cached models - first analysis will be slower")
        
    return recommendations

@app.route('/api/cache/stats')
def api_cache_stats():
    """ดูสถิติของ model cache"""
    if not MODEL_CACHE_AVAILABLE:
        return jsonify({
            'success': False, 
            'message': 'Model cache not available'
        })
    
    try:
        stats = get_cache_stats()
        return jsonify({
            'success': True,
            'cache_stats': stats,
            'recommendations': _get_cache_recommendations(stats)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/cache/clear', methods=['POST'])
def api_cache_clear():
    """ล้าง model cache"""
    if not MODEL_CACHE_AVAILABLE:
        return jsonify({
            'success': False, 
            'message': 'Model cache not available'
        })
    
    try:
        clear_model_cache()
        return jsonify({
            'success': True,
            'message': 'Model cache cleared successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/memory/cleanup', methods=['POST']) 
def api_memory_cleanup():
    """ทำความสะอาด memory และ app state"""
    try:
        # Clear app state data
        old_count = len(app_state.old_products_data) if app_state.old_products_data is not None else 0
        new_count = len(app_state.new_products_data) if app_state.new_products_data is not None else 0
        
        # Clear large DataFrames
        app_state.old_products_data = None
        app_state.new_products_data = None
        app_state.analysis_results = None
        app_state.review_queue.clear()
        app_state.feedback_data.clear()
        
        # Clear model cache
        if MODEL_CACHE_AVAILABLE:
            clear_model_cache()
        
        # Force garbage collection
        import gc
        collected = gc.collect()
        
        return jsonify({
            'success': True,
            'message': 'Memory cleanup completed',
            'freed_data': {
                'old_products_cleared': old_count,
                'new_products_cleared': new_count,
                'review_queue_cleared': len(app_state.review_queue),
                'feedback_cleared': len(app_state.feedback_data),
                'model_cache_cleared': MODEL_CACHE_AVAILABLE,
                'gc_objects_collected': collected
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/get-results')
def get_results():
    old_count = len(app_state.old_products_data) if app_state.old_products_data is not None else 0
    new_count = len(app_state.new_products_data) if app_state.new_products_data is not None else 0
    feedback_count = len(app_state.feedback_data)
    
    # ข้อมูลสินค้าใหม่ที่ไม่ซ้ำ
    unique_count = len(app_state.unique_products) if app_state.unique_products else 0
    duplicate_check_count = len(app_state.duplicate_check_needed) if app_state.duplicate_check_needed else 0
    
    # คำนวณความแม่นยำ
    if feedback_count > 0:
        correct_predictions = sum(1 for f in app_state.feedback_data 
                                if f.get('human_feedback') == f.get('ml_prediction'))
        accuracy = round((correct_predictions / feedback_count) * 100, 1)
    else:
        accuracy = 0.0
    
    return jsonify({
        'old_count': old_count,
        'new_count': new_count,
        'unique_count': unique_count,  # สินค้าใหม่ที่ไม่ซ้ำ
        'duplicate_check_count': duplicate_check_count,  # สินค้าที่ต้องตรวจสอบ
        'feedback_count': feedback_count,
        'accuracy': accuracy,
        'unique_percentage': round((unique_count / new_count) * 100, 1) if new_count > 0 else 0
    })

@app.route('/export-csv')
def export_csv():
    """Export สินค้าคัดแยกแล้วตาม Human Feedback (ทำงานหลังจากการประมวลผลเสร็จแล้ว)"""
    try:
        # ตรวจสอบว่าได้ทำการประมวลผลและคัดแยกแล้วหรือไม่
        if not hasattr(app_state, 'product_classification') or app_state.product_classification is None:
            return jsonify({'success': False, 'message': 'ยังไม่ได้ประมวลผลข้อมูล กรุณาเรียก /api/prepare-export ก่อน'})
        
        if not hasattr(app_state, 'unique_products') or app_state.unique_products is None or not app_state.unique_products:
            return jsonify({'success': False, 'message': 'ยังไม่ได้วิเคราะห์ข้อมูล กรุณาวิเคราะห์ก่อน'})
        
        logger.info("🔄 เริ่มสร้างไฟล์ CSV จากข้อมูลที่คัดแยกแล้ว...")
        
        # สร้างโฟลเดอร์ output ถ้าไม่มี
        os.makedirs('output', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ใช้ข้อมูลการคัดแยกจาก prepare-export
        product_classification = app_state.product_classification
        
        # รวม feedback data จาก memory และ backup file (เช่นเดียวกับ prepare-export)
        all_feedback_data = []
        try:
            backup_feedback = load_feedback_from_backup()
            
            # รวมข้อมูลโดยไม่ให้ซ้ำกัน
            feedback_dict = {}
            
            # เพิ่ม backup feedback ก่อน
            for fb in backup_feedback:
                key = (fb.get('old_product', ''), fb.get('new_product', ''))
                feedback_dict[key] = fb
            
            # เขียนทับด้วย memory feedback (ล่าสุดกว่า)
            for fb in app_state.feedback_data:
                key = (fb.get('old_product', ''), fb.get('new_product', ''))
                feedback_dict[key] = fb
            
            all_feedback_data = list(feedback_dict.values())
            logger.info(f"รวม feedback data: {len(all_feedback_data)} รายการ (memory: {len(app_state.feedback_data)}, backup: {len(backup_feedback)})")
            
        except Exception as e:
            logger.warning(f"Failed to load backup feedback, using memory only: {e}")
            all_feedback_data = app_state.feedback_data
        
        generated_files = []
        
        # 1. Export สินค้าที่อนุมัติให้นำเข้าระบบ (ไฟล์หลัก) ⭐⭐⭐
        logger.info("📝 กำลังสร้างไฟล์สินค้าที่อนุมัติให้นำเข้า...")
        approved_products = product_classification['approved_for_import']
        
        if approved_products:
            approved_df = pd.DataFrame(approved_products)
            approved_filename = f"approved_products_for_import_{timestamp}.csv"
            approved_filepath = os.path.join('output', approved_filename)
            approved_df.to_csv(approved_filepath, index=False, encoding='utf-8-sig')
            
            generated_files.append({
                'name': approved_filename,
                'type': 'primary',
                'description': 'สินค้าที่อนุมัติให้นำเข้าระบบ (ไฟล์หลัก)',
                'count': len(approved_products),
                'categories': {
                    'unique_products': len([p for p in approved_products if p['category'] == 'unique']),
                    'human_approved': len([p for p in approved_products if p['category'] == 'human_approved'])
                }
            })
        
        # 2. Export สินค้าที่ปฏิเสธเนื่องจากซ้ำ
        logger.info("📝 กำลังสร้างไฟล์สินค้าที่ปฏิเสธ...")
        rejected_products = product_classification['rejected_duplicates']
        
        if rejected_products:
            rejected_df = pd.DataFrame(rejected_products)
            rejected_filename = f"rejected_duplicates_{timestamp}.csv"
            rejected_filepath = os.path.join('output', rejected_filename)
            rejected_df.to_csv(rejected_filepath, index=False, encoding='utf-8-sig')
            
            generated_files.append({
                'name': rejected_filename,
                'type': 'reference',
                'description': 'สินค้าที่ปฏิเสธเนื่องจากซ้ำ',
                'count': len(rejected_products)
            })
        
        # 3. Export สินค้าที่ต้องตรวจสอบเพิ่มเติม
        logger.info("📝 กำลังสร้างไฟล์สินค้าที่ต้องตรวจสอบเพิ่มเติม...")
        pending_products = product_classification['pending_review']
        
        if pending_products:
            pending_df = pd.DataFrame(pending_products)
            pending_filename = f"pending_review_products_{timestamp}.csv"
            pending_filepath = os.path.join('output', pending_filename)
            pending_df.to_csv(pending_filepath, index=False, encoding='utf-8-sig')
            
            generated_files.append({
                'name': pending_filename,
                'type': 'action_required',
                'description': 'สินค้าที่ต้องตรวจสอบเพิ่มเติม',
                'count': len(pending_products)
            })
        
        # 4. Export รายงานสรุปการคัดแยก ✨
        logger.info("📝 กำลังสร้างรายงานสรุปการคัดแยก...")
        
        classification_stats = product_classification['classification_stats']
        summary_data = {
            'สินค้าเก่า_จำนวน': [len(app_state.old_products_data)],
            'สินค้าใหม่_จำนวน': [len(app_state.new_products_data)],
            'อนุมัติให้นำเข้า_จำนวน': [len(product_classification['approved_for_import'])],
            'ปฏิเสธเนื่องจากซ้ำ_จำนวน': [len(product_classification['rejected_duplicates'])],
            'ต้องตรวจสอบเพิ่ม_จำนวน': [len(product_classification['pending_review'])],
            'เปอร์เซ็นต์อนุมัติ': [f"{classification_stats['approval_rate']:.1f}%"],
            'เปอร์เซ็นต์ครอบคลุมการตรวจสอบ': [f"{classification_stats['human_review_coverage']:.1f}%"],
            'การตัดสินใจของมนุษย์_จำนวน': [len(all_feedback_data)],
            'วันที่วิเคราะห์': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'เวอร์ชันระบบ': ['Phase 4 Enhanced with Human Feedback & Smart Classification'],
            'สถานะการประมวลผล': ['สำเร็จสมบูรณ์ - พร้อมใช้งาน']
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = f"classification_summary_{timestamp}.csv"
        summary_filepath = os.path.join('output', summary_filename)
        summary_df.to_csv(summary_filepath, index=False, encoding='utf-8-sig')
        
        generated_files.append({
            'name': summary_filename,
            'type': 'summary',
            'description': 'สรุปการคัดแยกและวิเคราะห์',
            'count': 1
        })
        
        # 5. Export Human Feedback (ถ้ามี) ⚡
        if all_feedback_data:
            logger.info(f"📝 กำลังสร้างไฟล์ human feedback ({len(all_feedback_data)} รายการ)...")
            
            # Enhance results with Phase 4 features + Human Feedback
            enhanced_results = enhance_results_phase4(app_state.review_queue, all_feedback_data)
            feedback_df = pd.DataFrame(enhanced_results)
            
            feedback_filename = f"human_feedback_results_{timestamp}.csv"
            feedback_filepath = os.path.join('output', feedback_filename)
            feedback_df.to_csv(feedback_filepath, index=False, encoding='utf-8-sig')
            
            generated_files.append({
                'name': feedback_filename,
                'type': 'training',
                'description': 'ผลการตัดสินใจของมนุษย์ พร้อมข้อมูลสำหรับ AI',
                'count': len(enhanced_results)
            })
        
        # หาไฟล์หลัก (สินค้าที่อนุมัติให้นำเข้า)
        main_file = None
        main_file_count = 0
        
        for file_info in generated_files:
            if file_info['type'] == 'primary':
                main_file = file_info['name']
                main_file_count = file_info['count']
                break
        
        logger.info(f"✅ สร้างไฟล์เสร็จสิ้น: {len(generated_files)} ไฟล์")
        
        # สรุปการคัดแยก
        classification_summary = {
            'approved_for_import': len(product_classification['approved_for_import']),
            'rejected_duplicates': len(product_classification['rejected_duplicates']),
            'pending_review': len(product_classification['pending_review']),
            'total_processed': classification_stats['total_processed'],
            'approval_rate': classification_stats['approval_rate'],
            'human_review_coverage': classification_stats['human_review_coverage']
        }
        
        return jsonify({
            'success': True,
            'message': 'สร้างไฟล์ CSV สำเร็จ (คัดแยกตาม Human Feedback)',
            'generated_files': generated_files,
            'main_file': main_file,
            'classification_summary': classification_summary,
            'summary': {
                'approved_for_import': len(product_classification['approved_for_import']),
                'rejected_duplicates': len(product_classification['rejected_duplicates']),
                'pending_review': len(product_classification['pending_review']),
                'feedback_count': len(all_feedback_data),
                'total_files': len(generated_files)
            },
            'files_location': 'output/',
            'timestamp': timestamp,
            'main_purpose': f"ไฟล์หลัก: {main_file} - สินค้าที่อนุมัติให้นำเข้า {main_file_count} รายการ" if main_file else "ไฟล์สรุปการคัดแยก",
            'data_sources': f"Human feedback: {len(app_state.feedback_data)} in memory + backup data = {len(all_feedback_data)} total",
            'export_quality': {
                'data_completeness': 'สมบูรณ์',
                'human_feedback_included': len(all_feedback_data) > 0,
                'enhanced_processing': True,
                'backup_integration': True,
                'smart_classification': True
            },
            'business_value': {
                'ready_to_import': len(product_classification['approved_for_import']),
                'duplicate_prevention': len(product_classification['rejected_duplicates']),
                'quality_assurance': f"{classification_stats['approval_rate']:.1f}% approval rate"
            }
        })
        
    except Exception as e:
        logger.error(f"Export CSV error: {e}")
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาดในการสร้างไฟล์: {str(e)}'})

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
    """Enhanced API status with detailed /analyze capabilities"""
    # Check offline capability
    offline_status = "unknown"
    available_models = {}
    try:
        from advanced_models import check_offline_models, ensure_offline_capability
        available_models = check_offline_models()
        offline_ready = ensure_offline_capability()
        offline_status = "ready" if offline_ready else "not_ready"
    except Exception as e:
        offline_status = f"error: {str(e)}"
    
    # Get cache statistics
    cache_stats = {}
    if MODEL_CACHE_AVAILABLE:
        try:
            cache_stats = get_cache_stats()
        except Exception as e:
            cache_stats = {'error': str(e)}
    
    return jsonify({
        'status': 'running',
        'system': 'Human-in-the-Loop Product Deduplication v2.1 - Phase 4 Enhanced',
        'version': '2.1.0',
        'build_date': 'September 13, 2025',
        'api_endpoints': {
            '/analyze': {
                'method': 'POST',
                'description': 'AI-powered product deduplication analysis',
                'status': 'active',
                'features': [
                    'Multi-model AI analysis (TF-IDF, SentenceTransformer)',
                    'Offline capability (no internet required)', 
                    'Multilingual support (Thai-English)',
                    'Human-in-the-loop workflow',
                    'Confidence scoring and classification',
                    'Batch processing optimization',
                    'Memory-efficient processing'
                ],
                'supported_models': list(WEB_CONFIG.keys()) if isinstance(WEB_CONFIG, dict) else ['tfidf', 'sentence-bert', 'optimized-tfidf', 'mock'],
                'offline_models_available': available_models,
                'offline_status': offline_status,
                'performance_characteristics': {
                    'tfidf': {'speed': 'fast', 'accuracy': 'good', 'time': '<2s'},
                    'sentence-bert': {'speed': 'slower', 'accuracy': 'excellent', 'time': '5-15s'},
                    'optimized-tfidf': {'speed': 'medium', 'accuracy': 'very good', 'time': '2-5s'}
                }
            },
            '/api/review-summary': {
                'method': 'GET',
                'description': 'แสดงสรุปผลการตัดสินใจของมนุษย์ก่อนการส่งออก',
                'status': 'active',
                'features': [
                    'Statistics of human decisions',
                    'AI vs Human accuracy comparison', 
                    'Export readiness assessment',
                    'Decision details breakdown',
                    'Completion percentage tracking'
                ]
            },
            '/api/prepare-export': {
                'method': 'POST',
                'description': 'ประมวลผลและเตรียมข้อมูลก่อนการส่งออก',
                'status': 'active',
                'features': [
                    'Data processing and validation',
                    'Statistics calculation',
                    'Export preview generation',
                    'Human feedback integration',
                    'Quality assessment'
                ]
            },
            '/export-csv': {
                'method': 'GET', 
                'description': 'ส่งออกไฟล์ CSV หลังจากการประมวลผล',
                'status': 'enhanced',
                'features': [
                    'Enhanced CSV export with human feedback',
                    'Multiple file generation',
                    'Backup data integration', 
                    'Quality metadata inclusion',
                    'Training data for ML'
                ]
            }
        },
        'current_config': WEB_CONFIG,
        'system_capabilities': [
            'Interactive Human Review Interface',
            'Real-time Feedback Processing', 
            'AI-Human Collaboration',
            'Progress Tracking',
            'CSV Export with Human Feedback',
            'ML Training Data Generation',
            'Multi-Model Support',
            'Offline Operation',
            'Multilingual Processing',
            'Confidence Scoring',
            'Batch Processing',
            'Memory Optimization',
            '🆕 Pre-Export Review Summary',
            '🆕 Export Data Processing', 
            '🆕 Enhanced Statistics Display',
            '🆕 Quality Assessment Workflow'
        ],
        'state': {
            'old_products_loaded': app_state.old_products_data is not None,
            'new_products_loaded': app_state.new_products_data is not None,
            'review_queue_ready': len(app_state.review_queue) > 0,
            'feedback_collected': len(app_state.feedback_data),
            'analysis_completed': app_state.analysis_results is not None,
            'phase4_enhanced': True,
            'offline_ready': offline_status == "ready"
        },
        'performance_info': {
            'fresh_architecture_available': FRESH_ARCHITECTURE_AVAILABLE,
            'shared_utils_available': SHARED_UTILS_AVAILABLE,
            'model_cache_available': MODEL_CACHE_AVAILABLE,
            'model_cache_location': 'model_cache/',
            'supported_file_formats': ['.csv', '.xlsx', '.xls'],
            'max_file_size': '16MB',
            'recommended_hardware': {
                'ram': '4GB+ for TF-IDF, 8GB+ for SentenceTransformer',
                'cpu': 'Multi-core recommended',
                'gpu': 'Optional (CUDA-compatible for acceleration)'
            }
        },
        'cache_statistics': cache_stats if MODEL_CACHE_AVAILABLE else 'Cache not available',
        'memory_management': {
            'auto_cleanup': True,
            'background_cleanup_interval': '5 minutes',
            'max_unused_time': '30 minutes',
            'memory_limit': '2GB',
            'cleanup_apis': [
                'POST /api/cache/clear - Clear model cache',
                'POST /api/memory/cleanup - Full memory cleanup',
                'GET /api/cache/stats - View cache statistics'
            ]
        }
    })

if __name__ == '__main__':
    # โหลด feedback จาก backup file ตอน server เริ่มต้น
    try:
        backup_feedback = load_feedback_from_backup()
        if backup_feedback:
            app_state.feedback_data = backup_feedback
            logger.info(f"🔄 Restored {len(backup_feedback)} feedback items from backup on startup")
        else:
            logger.info("🆕 Starting with empty feedback data")
    except Exception as e:
        logger.warning(f"Failed to load backup feedback on startup: {e}")
    
    print("🎯 เริ่มต้นระบบหาสินค้าใหม่ที่ไม่ซ้ำ")
    print("🌐 เข้าใช้งานที่: http://localhost:5000")
    print("🤖 AI-Powered Product Uniqueness Detection")
    print("📋 จุดประสงค์: หาสินค้าใหม่ที่ไม่ซ้ำกับสินค้าเก่า")
    print("🔧 API Status: http://localhost:5000/api/status")
    print(f"💾 Human Feedback: {len(app_state.feedback_data)} items loaded from backup")
    print("🆕 New Features: Pre-export Review Summary & Data Processing")
    print("📊 Enhanced Workflow: /api/review-summary → /api/prepare-export → /export-csv")
    app.run(debug=True, host='0.0.0.0', port=5000)

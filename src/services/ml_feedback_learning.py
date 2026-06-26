#!/usr/bin/env python3
"""
ML Learning from Human Feedback System
=====================================

ระบบที่ให้ ML เรียนรู้จาก feedback ของมนุษย์เพื่อปรับปรุงการทำนาย

Features:
- เทรนโมเดลจาก human feedback
- ประเมินประสิทธิภาพโมเดล
- ปรับปรุงการทำนายแบบ continuous learning
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path

from src.services.human_feedback_system import (
    ProductComparison, FeedbackType
)
from src.core.fresh_implementations import ThaiTextProcessor
from supabase import Client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """สกัดฟีเจอร์สำหรับการเรียนรู้ของ ML"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def extract_features(self, comparisons: List[ProductComparison]) -> np.ndarray:
        """สกัดฟีเจอร์จากการเปรียบเทียบสินค้า"""
        features = []
        
        for comp in comparisons:
            # Basic similarity features
            feature_vector = [
                comp.similarity_score,
                comp.confidence_score,
                
                # Text length features
                len(comp.product1),
                len(comp.product2),
                abs(len(comp.product1) - len(comp.product2)),
                
                # Character overlap features
                self._calculate_character_overlap(comp.product1, comp.product2),
                self._calculate_word_overlap(comp.product1, comp.product2),
                
                # Language features
                self._has_thai_characters(comp.product1),
                self._has_thai_characters(comp.product2),
                self._has_english_characters(comp.product1),
                self._has_english_characters(comp.product2),
                
                # Number features
                self._has_numbers(comp.product1),
                self._has_numbers(comp.product2),
                self._extract_numbers_similarity(comp.product1, comp.product2),
                
                # Brand/model features
                self._brand_similarity(comp.product1, comp.product2),
                
                # Length ratio
                min(len(comp.product1), len(comp.product2)) / max(len(comp.product1), len(comp.product2)),
            ]
            
            features.append(feature_vector)
        
        features_array = np.array(features)
        
        # Scale features
        if not self.is_fitted:
            features_array = self.scaler.fit_transform(features_array)
            self.is_fitted = True
        else:
            features_array = self.scaler.transform(features_array)
        
        return features_array
    
    def _calculate_character_overlap(self, text1: str, text2: str) -> float:
        """คำนวณ character overlap"""
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def _calculate_word_overlap(self, text1: str, text2: str) -> float:
        """คำนวณ word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0
    
    def _has_thai_characters(self, text: str) -> float:
        """ตรวจสอบว่ามีอักษรไทยหรือไม่"""
        thai_chars = sum(1 for char in text if '\u0E00' <= char <= '\u0E7F')
        return thai_chars / len(text) if len(text) > 0 else 0.0
    
    def _has_english_characters(self, text: str) -> float:
        """ตรวจสอบว่ามีอักษรอังกฤษหรือไม่"""
        english_chars = sum(1 for char in text if char.isalpha() and char.isascii())
        return english_chars / len(text) if len(text) > 0 else 0.0
    
    def _has_numbers(self, text: str) -> float:
        """ตรวจสอบว่ามีตัวเลขหรือไม่"""
        numbers = sum(1 for char in text if char.isdigit())
        return numbers / len(text) if len(text) > 0 else 0.0
    
    def _extract_numbers_similarity(self, text1: str, text2: str) -> float:
        """เปรียบเทียบตัวเลขในข้อความ"""
        numbers1 = set([char for char in text1 if char.isdigit()])
        numbers2 = set([char for char in text2 if char.isdigit()])
        
        if not numbers1 and not numbers2:
            return 1.0
        elif not numbers1 or not numbers2:
            return 0.0
        
        intersection = len(numbers1.intersection(numbers2))
        union = len(numbers1.union(numbers2))
        return intersection / union
    
    def _brand_similarity(self, text1: str, text2: str) -> float:
        """เปรียบเทียบแบรนด์ที่พบในข้อความ"""
        common_brands = [
            'iphone', 'samsung', 'galaxy', 'macbook', 'ipad', 'airpods',
            'xiaomi', 'huawei', 'oppo', 'vivo', 'lenovo', 'asus', 'acer',
            'hp', 'dell', 'microsoft', 'surface', 'sony', 'lg', 'nokia'
        ]
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        brands1 = set([brand for brand in common_brands if brand in text1_lower])
        brands2 = set([brand for brand in common_brands if brand in text2_lower])
        
        if not brands1 and not brands2:
            return 0.5  # No brands detected
        elif brands1 == brands2:
            return 1.0  # Same brands
        elif brands1.intersection(brands2):
            return 0.7  # Some overlap
        else:
            return 0.0  # Different brands


class FeedbackLearningModel:
    """โมเดลสำหรับเรียนรู้จาก human feedback"""
    
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = self._create_model(model_type)
        self.feature_extractor = FeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.training_history = []
        
    def _create_model(self, model_type: str):
        """สร้างโมเดล ML"""
        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        elif model_type == "logistic_regression":
            return LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train_from_feedback(self, comparisons: List[ProductComparison]) -> Dict[str, Any]:
        """เทรนโมเดลจาก human feedback"""
        logger.info(f"🎓 เริ่มการเรียนรู้จาก {len(comparisons)} feedback")
        
        # Filter เฉพาะที่มี human feedback
        labeled_comparisons = [
            comp for comp in comparisons 
            if comp.human_feedback is not None
        ]
        
        if len(labeled_comparisons) < 10:
            raise ValueError("ต้องมี feedback อย่างน้อย 10 รายการสำหรับการเทรน")
        
        # สกัดฟีเจอร์
        X = self.feature_extractor.extract_features(labeled_comparisons)
        
        # เตรียม labels
        y_labels = [comp.human_feedback.value for comp in labeled_comparisons]
        y = self.label_encoder.fit_transform(y_labels)
        
        # แบ่งข้อมูล train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # เทรนโมเดล
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # ประเมินผล
        train_accuracy = accuracy_score(y_train, self.model.predict(X_train))
        test_accuracy = accuracy_score(y_test, self.model.predict(X_test))
        
        # Cross validation
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        
        # Classification report
        y_pred = self.model.predict(X_test)
        class_report = classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_.tolist()
        else:
            feature_importance = []
        
        training_result = {
            'training_date': datetime.now().isoformat(),
            'model_type': self.model_type,
            'total_samples': len(labeled_comparisons),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std(),
            'classification_report': class_report,
            'feature_importance': feature_importance,
            'classes': self.label_encoder.classes_.tolist()
        }
        
        self.training_history.append(training_result)
        
        logger.info(f"✅ การเทรนเสร็จสิ้น - ความแม่นยำ: {test_accuracy:.3f}")
        return training_result
    
    def predict_relationship(self, comparison: ProductComparison) -> Tuple[FeedbackType, float]:
        """ทำนายความสัมพันธ์ระหว่างสินค้า"""
        if not self.is_trained:
            raise ValueError("โมเดลยังไม่ได้รับการเทรน")
        
        # สกัดฟีเจอร์
        X = self.feature_extractor.extract_features([comparison])
        
        # ทำนาย
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # แปลงกลับเป็น label
        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities)
        
        return FeedbackType(predicted_label), confidence
    
    def evaluate_against_human(self, comparisons: List[ProductComparison]) -> Dict[str, Any]:
        """ประเมินประสิทธิภาพเทียบกับ human feedback"""
        if not self.is_trained:
            raise ValueError("โมเดลยังไม่ได้รับการเทรน")
        
        # Filter เฉพาะที่มี human feedback
        labeled_comparisons = [
            comp for comp in comparisons 
            if comp.human_feedback is not None
        ]
        
        if not labeled_comparisons:
            return {'error': 'ไม่มีข้อมูล human feedback สำหรับการประเมิน'}
        
        # ทำนายทั้งหมด
        predictions = []
        confidences = []
        true_labels = []
        
        for comp in labeled_comparisons:
            try:
                pred_type, confidence = self.predict_relationship(comp)
                predictions.append(pred_type.value)
                confidences.append(confidence)
                true_labels.append(comp.human_feedback.value)
            except Exception as e:
                logger.warning(f"ไม่สามารถทำนายได้: {e}")
                continue
        
        # คำนวณความแม่นยำ
        accuracy = sum(1 for p, t in zip(predictions, true_labels) if p == t) / len(predictions)
        
        # สร้าง confusion matrix
        unique_labels = list(set(true_labels + predictions))
        cm = confusion_matrix(true_labels, predictions, labels=unique_labels)
        
        # Classification report
        class_report = classification_report(
            true_labels, predictions, 
            labels=unique_labels,
            output_dict=True,
            zero_division=0
        )
        
        return {
            'evaluation_date': datetime.now().isoformat(),
            'total_samples': len(labeled_comparisons),
            'accuracy': accuracy,
            'average_confidence': np.mean(confidences),
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_labels': unique_labels,
            'classification_report': class_report,
            'predictions': list(zip(predictions, true_labels, confidences))
        }
    
    def save_model(self, filepath: str):
        """บันทึกโมเดล"""
        model_data = {
            'model': self.model,
            'feature_extractor': self.feature_extractor,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }
        joblib.dump(model_data, filepath)
        logger.info(f"💾 บันทึกโมเดลที่ {filepath}")
    
    def load_model(self, filepath: str):
        """โหลดโมเดล"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_extractor = model_data['feature_extractor']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data.get('training_history', [])
        logger.info(f"📂 โหลดโมเดลจาก {filepath}")


class ContinuousLearningSystem:
    """ระบบเรียนรู้แบบต่อเนื่อง"""
    
    def __init__(self, supabase_client: Client, model_path: str = "feedback_model.joblib"):
        self.supabase = supabase_client
        self.model_path = model_path
        self.model = FeedbackLearningModel()
        self.text_processor = ThaiTextProcessor(remove_all_spaces=True)
        
        # โหลดโมเดลเก่าถ้ามี
        if Path(model_path).exists():
            try:
                self.model.load_model(model_path)
                logger.info("โหลดโมเดลเดิมสำเร็จ")
            except Exception as e:
                logger.warning(f"ไม่สามารถโหลดโมเดลเดิม: {e}")
                
    def _fetch_training_data(self) -> List[ProductComparison]:
        """ดึงข้อมูล feedback จาก Supabase similarity_matches ที่รีวิวแล้ว"""
        response = self.supabase.table('similarity_matches') \
            .select('*, product_a:products!similarity_matches_product_a_id_fkey(name_th), product_b:products!similarity_matches_product_b_id_fkey(name_th)') \
            .eq('reviewed', True).execute()
        
        comparisons = []
        for row in response.data:
            name1 = row.get('product_a', {}).get('name_th', '') if isinstance(row.get('product_a'), dict) else ''
            name2 = row.get('product_b', {}).get('name_th', '') if isinstance(row.get('product_b'), dict) else ''
            
            if not name1 or not name2:
                continue

            # human_feedback interpretation
            feedback = FeedbackType.DUPLICATE if row.get('is_duplicate') else FeedbackType.DIFFERENT

            clean1 = self.text_processor.process(name1)
            clean2 = self.text_processor.process(name2)

            comp = ProductComparison(
                id=str(row['id']),
                product1=name1,
                product2=name2,
                product1_cleaned=clean1,
                product2_cleaned=clean2,
                similarity_score=float(row['similarity_score']),
                confidence_score=0.8,
                ml_prediction=feedback,
                human_feedback=feedback,
                is_training_data=True
            )
            comparisons.append(comp)
            
        return comparisons
    
    def retrain_model(self, min_new_feedback: int = 5) -> Dict[str, Any]:
        """เทรนโมเดลใหม่เมื่อมี feedback เพิ่ม"""
        # ดึงข้อมูล feedback ทั้งหมด
        training_data = self._fetch_training_data()
        
        if len(training_data) < 10:
            return {
                'status': 'insufficient_data',
                'message': f'ต้องมี feedback อย่างน้อย 10 รายการ (ปัจจุบันมี {len(training_data)})'
            }
        
        try:
            # เทรนโมเดลใหม่
            training_result = self.model.train_from_feedback(training_data)
            
            # บันทึกโมเดล
            self.model.save_model(self.model_path)
            
            # บันทึกประวัติการเทรน
            self._save_training_history(training_result)
            
            return {
                'status': 'success',
                'training_result': training_result
            }
            
        except Exception as e:
            logger.error(f"การเทรนโมเดลล้มเหลว: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _save_training_history(self, training_result: Dict[str, Any]):
        """บันทึกประวัติการเทรนเข้า Supabase ml_training_history table ถาวร"""
        try:
            feature_importance = training_result.get('feature_importance', [])
            # แปลง feature_importance list เป็น dict-list สำหรับ JSONB
            feature_names = [
                'similarity_score', 'confidence_score',
                'len_p1', 'len_p2', 'len_diff',
                'char_overlap', 'word_overlap',
                'thai_p1', 'thai_p2', 'eng_p1', 'eng_p2',
                'num_p1', 'num_p2', 'num_similarity',
                'brand_similarity', 'len_ratio'
            ]
            fi_dicts = [
                {"feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                 "importance": round(float(v), 6)}
                for i, v in enumerate(feature_importance)
            ] if isinstance(feature_importance, list) and len(feature_importance) > 0 else []

            record = {
                'training_date': training_result.get('training_date', datetime.now().isoformat()),
                'model_type':    training_result.get('model_type', 'random_forest'),
                'total_samples': int(training_result.get('total_samples', 0)),
                'train_samples': int(training_result.get('train_samples', 0)),
                'test_samples':  int(training_result.get('test_samples', 0)),
                'train_accuracy':    float(training_result.get('train_accuracy', 0)),
                'test_accuracy':     float(training_result.get('test_accuracy', 0)),
                'cv_mean_accuracy':  float(training_result.get('cv_mean_accuracy', 0)),
                'cv_std_accuracy':   float(training_result.get('cv_std_accuracy', 0)),
                'feature_importance': fi_dicts,
                'classes': training_result.get('classes', []),
                'classification_report': training_result.get('classification_report', {}),
            }
            self.supabase.table('ml_training_history').insert(record).execute()
            logger.info(f"✅ บันทึกประวัติการเทรนลง Supabase สำเร็จ (accuracy={record['test_accuracy']:.3f})")
        except Exception as e:
            logger.error(f"❌ ไม่สามารถบันทึกประวัติการเทรนลง Supabase: {e}")
            # Fallback: log เท่านั้น ไม่ให้ crash
            logger.info(f"Training History (log only): {json.dumps(training_result, ensure_ascii=False)[:200]}...")
    
    def get_model_performance(self) -> Dict[str, Any]:
        """ดึงข้อมูลประสิทธิภาพโมเดล"""
        if not self.model.is_trained:
            return {'status': 'not_trained', 'message': 'โมเดลยังไม่ได้รับการเทรน'}
        
        # ดึงข้อมูล feedback สำหรับประเมิน
        evaluation_data = self._fetch_training_data()
        
        if not evaluation_data:
            return {'status': 'no_data', 'message': 'ไม่มีข้อมูลสำหรับประเมิน'}
        
        # ประเมินประสิทธิภาพ
        evaluation_result = self.model.evaluate_against_human(evaluation_data)
        
        # นับจำนวน sessions จาก Supabase (ถ้าดึงได้)
        db_sessions = 0
        try:
            resp = self.supabase.table('ml_training_history').select('id', count='exact').execute()
            db_sessions = resp.count or 0
        except Exception:
            db_sessions = len(self.model.training_history)
        
        return {
            'status': 'success',
            'model_info': {
                'type': self.model.model_type,
                'is_trained': self.model.is_trained,
                'training_sessions': db_sessions
            },
            'evaluation': evaluation_result
        }

    def get_training_history(self, limit: int = 10) -> Dict[str, Any]:
        """ดึงประวัติการเทรนจาก Supabase (ล่าสุดก่อน)"""
        try:
            resp = self.supabase.table('ml_training_history') \
                .select('id,training_date,model_type,total_samples,train_accuracy,test_accuracy,cv_mean_accuracy,feature_importance,classes') \
                .order('training_date', desc=True) \
                .limit(limit) \
                .execute()
            return {
                'status': 'success',
                'history': resp.data or [],
                'total': len(resp.data or [])
            }
        except Exception as e:
            logger.error(f"ไม่สามารถดึงประวัติการเทรน: {e}")
            # Fallback จาก memory
            return {
                'status': 'success',
                'history': list(reversed(self.model.training_history[-limit:])),
                'total': len(self.model.training_history),
                'source': 'memory_fallback'
            }


def demo_learning_system():
    """ตัวอย่างการใช้งานระบบเรียนรู้"""
    print("🤖 Demo: ML Learning from Human Feedback")
    print("=" * 60)
    
    # สร้างข้อมูล feedback จำลอง
    demo_comparisons = [
        ProductComparison(
            id="demo_1",
            product1="iPhone 14 Pro Max 256GB สีดำ",
            product2="iPhone 14 Pro Max 256GB Black",
            similarity_score=0.92,
            confidence_score=0.85,
            ml_prediction=FeedbackType.DUPLICATE,
            human_feedback=FeedbackType.DUPLICATE,
            human_comments="สินค้าเดียวกัน แค่ภาษาต่างกัน"
        ),
        ProductComparison(
            id="demo_2",
            product1="Samsung Galaxy S23 Ultra",
            product2="Samsung Galaxy S23 Ultra 256GB",
            similarity_score=0.88,
            confidence_score=0.72,
            ml_prediction=FeedbackType.SIMILAR,
            human_feedback=FeedbackType.DUPLICATE,
            human_comments="ความจุต่างกัน แต่เป็นรุ่นเดียวกัน"
        ),
        ProductComparison(
            id="demo_3",
            product1="MacBook Pro 14 inch",
            product2="MacBook Air 13 inch",
            similarity_score=0.65,
            confidence_score=0.60,
            ml_prediction=FeedbackType.SIMILAR,
            human_feedback=FeedbackType.DIFFERENT,
            human_comments="แบรนด์เดียวกัน แต่รุ่นต่างกัน"
        )
    ]
    
    # สร้างระบบเรียนรู้
    learning_system = ContinuousLearningSystem()
    
    # บันทึก feedback ลงฐานข้อมูล
    for comp in demo_comparisons:
        learning_system.db.save_comparison(comp)
    
    print(f"\n📝 บันทึก {len(demo_comparisons)} feedback")
    
    # เทรนโมเดล
    print("\n🎓 เริ่มการเทรนโมเดล...")
    result = learning_system.retrain_model()
    
    if result['status'] == 'success':
        training_info = result['training_result']
        print(f"✅ การเทรนสำเร็จ!")
        print(f"   - ความแม่นยำ: {training_info['test_accuracy']:.3f}")
        print(f"   - Cross-validation: {training_info['cv_mean_accuracy']:.3f} ± {training_info['cv_std_accuracy']:.3f}")
    else:
        print(f"❌ การเทรนล้มเหลว: {result['message']}")
        return
    
    # ประเมินประสิทธิภาพ
    print("\n📊 ประเมินประสิทธิภาพโมเดล...")
    performance = learning_system.get_model_performance()
    
    if performance['status'] == 'success':
        eval_info = performance['evaluation']
        print(f"✅ ความแม่นยำ: {eval_info['accuracy']:.3f}")
        print(f"   - ความมั่นใจเฉลี่ย: {eval_info['average_confidence']:.3f}")
    
    # ทดสอบการทำนาย
    print("\n🔮 ทดสอบการทำนาย...")
    test_comparison = ProductComparison(
        id="test_1",
        product1="iPad Pro 11 inch",
        product2="iPad Pro 11\" M2",
        similarity_score=0.89,
        confidence_score=0.75,
        ml_prediction=FeedbackType.SIMILAR
    )
    
    try:
        prediction, confidence = learning_system.model.predict_relationship(test_comparison)
        print(f"✅ การทำนาย: {prediction.value} (ความมั่นใจ: {confidence:.3f})")
    except Exception as e:
        print(f"❌ ไม่สามารถทำนายได้: {e}")


if __name__ == "__main__":
    demo_learning_system()

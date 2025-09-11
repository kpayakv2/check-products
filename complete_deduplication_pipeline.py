#!/usr/bin/env python3
"""
Complete Human-in-the-Loop Product Deduplication Pipeline
========================================================

ระบบครบวงจรสำหรับหาสินค้าที่ไม่ซ้ำ และให้ ML เรียนรู้จาก human feedback

Usage:
    python complete_deduplication_pipeline.py --input products.csv --mode analyze
    python complete_deduplication_pipeline.py --input products.csv --mode review
    python complete_deduplication_pipeline.py --input products.csv --mode train
"""

import argparse
import pandas as pd
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Import our modules
from human_feedback_system import (
    ProductDeduplicationSystem, HumanReviewInterface, 
    HumanFeedbackDatabase, ProductComparison, FeedbackType
)
from ml_feedback_learning import ContinuousLearningSystem, FeedbackLearningModel
from api_server import app

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompletePipeline:
    """ไปป์ไลน์ครบวงจรสำหรับการหาสินค้าที่ไม่ซ้ำ"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.8,
                 db_path: str = "product_deduplication.db",
                 model_path: str = "deduplication_model.joblib"):
        
        self.similarity_threshold = similarity_threshold
        self.db_path = db_path
        self.model_path = model_path
        
        # Initialize components
        self.dedup_system = ProductDeduplicationSystem(similarity_threshold)
        self.learning_system = ContinuousLearningSystem(db_path, model_path)
        self.review_interface = HumanReviewInterface(self.learning_system.db)
        
        logger.info("🚀 เริ่มต้นระบบ Complete Deduplication Pipeline")
    
    def load_products_from_file(self, filepath: str, column_name: str = None) -> List[str]:
        """โหลดข้อมูลสินค้าจากไฟล์"""
        file_path = Path(filepath)
        
        if not file_path.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์: {filepath}")
        
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(filepath, encoding='utf-8-sig')
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"รองรับเฉพาะไฟล์ .csv และ .xlsx")
        
        # หาคอลัมน์ที่มีข้อมูลสินค้า
        if column_name:
            if column_name not in df.columns:
                raise ValueError(f"ไม่พบคอลัมน์: {column_name}")
            products = df[column_name].dropna().astype(str).tolist()
        else:
            # ลองหาคอลัมน์ที่น่าจะเป็นชื่อสินค้า
            possible_columns = [
                'product_name', 'name', 'ชื่อสินค้า', 'สินค้า',
                'product', 'item_name', 'item', 'title'
            ]
            
            found_column = None
            for col in possible_columns:
                if col in df.columns:
                    found_column = col
                    break
            
            if not found_column:
                # ใช้คอลัมน์แรกที่มีข้อมูลแบบ string
                string_columns = df.select_dtypes(include=['object']).columns
                if len(string_columns) > 0:
                    found_column = string_columns[0]
                    logger.warning(f"ใช้คอลัมน์ '{found_column}' เป็นชื่อสินค้า")
                else:
                    raise ValueError("ไม่สามารถระบุคอลัมน์ที่มีชื่อสินค้าได้")
            
            products = df[found_column].dropna().astype(str).tolist()
        
        logger.info(f"📦 โหลดสินค้า {len(products)} รายการจากไฟล์ {filepath}")
        return products
    
    def analyze_products(self, products: List[str]) -> Dict[str, Any]:
        """วิเคราะห์และหาสินค้าที่ซ้ำ"""
        logger.info("🔍 เริ่มวิเคราะห์สินค้าที่อาจซ้ำกัน...")
        
        # หาการเปรียบเทียบที่น่าสนใจ
        comparisons = self.dedup_system.find_potential_duplicates(products)
        
        # จัดกลุ่มตามระดับความคล้าย
        analysis_result = {
            'total_products': len(products),
            'total_comparisons': len(comparisons),
            'high_similarity': [],  # >= 0.9
            'medium_similarity': [], # 0.7-0.9
            'low_similarity': [],   # 0.5-0.7
            'ml_predictions': {
                'duplicate': 0,
                'similar': 0,
                'different': 0,
                'uncertain': 0
            },
            'requires_human_review': []
        }
        
        for comp in comparisons:
            # จัดตามระดับความคล้าย
            if comp.similarity_score >= 0.9:
                analysis_result['high_similarity'].append({
                    'product1': comp.product1,
                    'product2': comp.product2,
                    'similarity': comp.similarity_score,
                    'ml_prediction': comp.ml_prediction.value
                })
            elif comp.similarity_score >= 0.7:
                analysis_result['medium_similarity'].append({
                    'product1': comp.product1,
                    'product2': comp.product2,
                    'similarity': comp.similarity_score,
                    'ml_prediction': comp.ml_prediction.value
                })
            else:
                analysis_result['low_similarity'].append({
                    'product1': comp.product1,
                    'product2': comp.product2,
                    'similarity': comp.similarity_score,
                    'ml_prediction': comp.ml_prediction.value
                })
            
            # นับการทำนายของ ML
            analysis_result['ml_predictions'][comp.ml_prediction.value] += 1
            
            # รายการที่ต้องให้มนุษย์ตรวจ (confidence ต่ำ)
            if comp.confidence_score < 0.7:
                analysis_result['requires_human_review'].append({
                    'id': comp.id,
                    'product1': comp.product1,
                    'product2': comp.product2,
                    'similarity': comp.similarity_score,
                    'confidence': comp.confidence_score,
                    'ml_prediction': comp.ml_prediction.value
                })
        
        logger.info(f"✅ วิเคราะห์เสร็จสิ้น - พบการเปรียบเทียบ {len(comparisons)} คู่")
        return analysis_result
    
    def start_human_review(self, reviewer_name: str, batch_size: int = 10):
        """เริ่มเซสชันให้มนุษย์ตรวจสอบ"""
        logger.info(f"👤 เริ่มเซสชันตรวจสอบโดย {reviewer_name}")
        self.review_interface.start_review_session(reviewer_name, batch_size)
    
    def train_model(self) -> Dict[str, Any]:
        """เทรนโมเดลจาก human feedback"""
        logger.info("🎓 เริ่มการเทรนโมเดล...")
        return self.learning_system.retrain_model()
    
    def extract_unique_products(self, products: List[str]) -> Dict[str, Any]:
        """สกัดสินค้าที่ไม่ซ้ำ"""
        logger.info("📦 สกัดสินค้าที่ไม่ซ้ำ...")
        
        unique_products = self.dedup_system.extract_unique_products(products)
        
        # จัดกลุ่มผลลัพธ์
        clusters = {}
        representatives = []
        
        for product in unique_products:
            if product.cluster_id not in clusters:
                clusters[product.cluster_id] = []
            clusters[product.cluster_id].append({
                'name': product.name,
                'is_representative': product.is_representative,
                'confidence': product.confidence
            })
            
            if product.is_representative:
                representatives.append(product.name)
        
        result = {
            'total_input_products': len(products),
            'total_unique_products': len(representatives),
            'deduplication_ratio': 1 - (len(representatives) / len(products)),
            'clusters': clusters,
            'representative_products': representatives,
            'cluster_summary': {
                'total_clusters': len(clusters),
                'single_product_clusters': sum(1 for c in clusters.values() if len(c) == 1),
                'multi_product_clusters': sum(1 for c in clusters.values() if len(c) > 1),
                'largest_cluster_size': max(len(c) for c in clusters.values()) if clusters else 0
            }
        }
        
        logger.info(f"✅ พบสินค้าที่ไม่ซ้ำ {len(representatives)} รายการ จาก {len(products)} รายการ")
        return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """ดูสถานะของระบบ"""
        # ข้อมูลฐานข้อมูล
        pending_reviews = self.learning_system.db.get_pending_reviews(limit=1000)
        training_data = self.learning_system.db.get_training_data()
        
        # สถานะโมเดล
        model_performance = self.learning_system.get_model_performance()
        
        status = {
            'database': {
                'pending_reviews': len(pending_reviews),
                'completed_reviews': len(training_data),
                'total_comparisons': len(pending_reviews) + len(training_data)
            },
            'model': {
                'is_trained': self.learning_system.model.is_trained,
                'model_type': self.learning_system.model.model_type,
                'training_sessions': len(self.learning_system.model.training_history)
            },
            'performance': model_performance
        }
        
        return status
    
    def export_results(self, results: Dict[str, Any], output_file: str):
        """ส่งออกผลลัพธ์"""
        output_path = Path(output_file)
        
        if output_path.suffix.lower() == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        elif output_path.suffix.lower() == '.csv':
            # ส่งออกเป็น CSV สำหรับสินค้าที่ไม่ซ้ำ
            if 'representative_products' in results:
                df = pd.DataFrame({
                    'unique_product': results['representative_products']
                })
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
            else:
                # ส่งออกการวิเคราะห์
                data_for_export = []
                for category in ['high_similarity', 'medium_similarity', 'low_similarity']:
                    if category in results:
                        for item in results[category]:
                            data_for_export.append({
                                'category': category,
                                'product1': item['product1'],
                                'product2': item['product2'],
                                'similarity': item['similarity'],
                                'ml_prediction': item['ml_prediction']
                            })
                
                df = pd.DataFrame(data_for_export)
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"💾 ส่งออกผลลัพธ์ที่ {output_path}")


def main():
    """ฟังก์ชันหลัก"""
    parser = argparse.ArgumentParser(description='Complete Product Deduplication Pipeline')
    parser.add_argument('--input', '-i', required=True, help='Input file (CSV/Excel)')
    parser.add_argument('--column', '-c', help='Column name for product names')
    parser.add_argument('--mode', '-m', choices=['analyze', 'review', 'train', 'extract', 'status', 'api'], 
                        default='analyze', help='Operation mode')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--reviewer', '-r', help='Reviewer name for review mode')
    parser.add_argument('--batch-size', '-b', type=int, default=10, help='Review batch size')
    parser.add_argument('--threshold', '-t', type=float, default=0.8, help='Similarity threshold')
    parser.add_argument('--host', default='127.0.0.1', help='API host')
    parser.add_argument('--port', type=int, default=8000, help='API port')
    
    args = parser.parse_args()
    
    # สร้างไปป์ไลน์
    pipeline = CompletePipeline(similarity_threshold=args.threshold)
    
    try:
        if args.mode == 'api':
            # เริ่ม API server
            import uvicorn
            print(f"🌐 เริ่ม API Server ที่ http://{args.host}:{args.port}")
            print(f"📖 เอกสาร API: http://{args.host}:{args.port}/docs")
            print(f"🔗 Web Interface: http://{args.host}:{args.port}/web")
            uvicorn.run(app, host=args.host, port=args.port)
            return
        
        # โหลดข้อมูลสินค้า
        products = pipeline.load_products_from_file(args.input, args.column)
        
        if args.mode == 'analyze':
            # วิเคราะห์สินค้า
            results = pipeline.analyze_products(products)
            
            # แสดงผลสรุป
            print("\n📊 ผลการวิเคราะห์:")
            print(f"   สินค้าทั้งหมด: {results['total_products']:,} รายการ")
            print(f"   การเปรียบเทียบ: {results['total_comparisons']:,} คู่")
            print(f"   ความคล้ายสูง (≥0.9): {len(results['high_similarity'])} คู่")
            print(f"   ความคล้ายปานกลาง (0.7-0.9): {len(results['medium_similarity'])} คู่")
            print(f"   ความคล้ายต่ำ (0.5-0.7): {len(results['low_similarity'])} คู่")
            print(f"   ต้องตรวจสอบโดยมนุษย์: {len(results['requires_human_review'])} คู่")
            
            print("\n🤖 การทำนายของ ML:")
            for pred_type, count in results['ml_predictions'].items():
                print(f"   {pred_type}: {count} คู่")
            
            if args.output:
                pipeline.export_results(results, args.output)
        
        elif args.mode == 'review':
            # เซสชันตรวจสอบโดยมนุษย์
            if not args.reviewer:
                reviewer_name = input("ชื่อผู้ตรวจสอบ: ").strip()
                if not reviewer_name:
                    print("❌ ต้องระบุชื่อผู้ตรวจสอบ")
                    sys.exit(1)
            else:
                reviewer_name = args.reviewer
            
            # วิเคราะห์ก่อนเพื่อสร้างข้อมูลสำหรับตรวจสอบ
            pipeline.analyze_products(products)
            
            # เริ่มเซสชันตรวจสอบ
            pipeline.start_human_review(reviewer_name, args.batch_size)
        
        elif args.mode == 'train':
            # เทรนโมเดล
            result = pipeline.train_model()
            
            if result['status'] == 'success':
                training_info = result['training_result']
                print(f"✅ การเทรนโมเดลสำเร็จ!")
                print(f"   ความแม่นยำ: {training_info['test_accuracy']:.3f}")
                print(f"   Cross-validation: {training_info['cv_mean_accuracy']:.3f} ± {training_info['cv_std_accuracy']:.3f}")
                print(f"   ข้อมูลการเทรน: {training_info['total_samples']} รายการ")
            else:
                print(f"❌ การเทรนล้มเหลว: {result['message']}")
        
        elif args.mode == 'extract':
            # สกัดสินค้าที่ไม่ซ้ำ
            results = pipeline.extract_unique_products(products)
            
            print("\n📦 ผลการสกัดสินค้าที่ไม่ซ้ำ:")
            print(f"   สินค้าเข้า: {results['total_input_products']:,} รายการ")
            print(f"   สินค้าที่ไม่ซ้ำ: {results['total_unique_products']:,} รายการ")
            print(f"   อัตราการลดซ้ำ: {results['deduplication_ratio']:.1%}")
            print(f"   กลุ่มทั้งหมด: {results['cluster_summary']['total_clusters']} กลุ่ม")
            print(f"   กลุ่มที่มีสินค้าเดียว: {results['cluster_summary']['single_product_clusters']} กลุ่ม")
            print(f"   กลุ่มที่มีหลายสินค้า: {results['cluster_summary']['multi_product_clusters']} กลุ่ม")
            
            if args.output:
                pipeline.export_results(results, args.output)
        
        elif args.mode == 'status':
            # แสดงสถานะระบบ
            status = pipeline.get_system_status()
            
            print("\n📊 สถานะระบบ:")
            print(f"   รอตรวจสอบ: {status['database']['pending_reviews']} รายการ")
            print(f"   ตรวจสอบแล้ว: {status['database']['completed_reviews']} รายการ")
            print(f"   โมเดลเทรนแล้ว: {'✅' if status['model']['is_trained'] else '❌'}")
            print(f"   ประเภทโมเดล: {status['model']['model_type']}")
            print(f"   เซสชันการเทรน: {status['model']['training_sessions']} ครั้ง")
            
            if status['performance']['status'] == 'success':
                perf = status['performance']['evaluation']
                print(f"   ความแม่นยำโมเดล: {perf['accuracy']:.3f}")
    
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาด: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

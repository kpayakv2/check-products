#!/usr/bin/env python3
"""
Human-in-the-Loop Product Deduplication System
==============================================

ระบบหาสินค้าที่ไม่ซ้ำ และให้มนุษย์ตรวจสอบเพื่อให้ ML เรียนรู้

Features:
- หาสินค้าที่ไม่ซ้ำและใกล้เคียง
- ระบบให้มนุษย์ตรวจสอบและยืนยัน
- เก็บ feedback เพื่อให้ ML เรียนรู้
- ปรับปรุงโมเดลจาก human feedback
"""

import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from enum import Enum

# Import existing components
from fresh_architecture import ProductSimilarityPipeline
from fresh_implementations import ComponentFactory
from main_phase4 import Phase4Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """ประเภทของการให้ feedback"""
    DUPLICATE = "duplicate"      # สินค้าซ้ำ
    SIMILAR = "similar"          # สินค้าคล้าย แต่ไม่ซ้ำ
    DIFFERENT = "different"      # สินค้าต่างกัน
    UNCERTAIN = "uncertain"      # ไม่แน่ใจ ต้องตรวจสอบเพิ่ม


@dataclass
class ProductComparison:
    """ข้อมูลการเปรียบเทียบสินค้า"""
    id: str
    product1: str
    product2: str
    similarity_score: float
    confidence_score: float
    ml_prediction: FeedbackType
    human_feedback: Optional[FeedbackType] = None
    human_comments: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    is_training_data: bool = False


@dataclass
class UniqueProduct:
    """ข้อมูลสินค้าที่ไม่ซ้ำ"""
    id: str
    name: str
    cluster_id: Optional[str] = None
    is_representative: bool = False
    similar_products: List[str] = None
    confidence: float = 0.0


class HumanFeedbackDatabase:
    """ฐานข้อมูลสำหรับเก็บ feedback จากมนุษย์"""
    
    def __init__(self, db_path: str = "human_feedback.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """สร้างตารางฐานข้อมูล"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # ตารางการเปรียบเทียบสินค้า
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS product_comparisons (
                id TEXT PRIMARY KEY,
                product1 TEXT NOT NULL,
                product2 TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                confidence_score REAL NOT NULL,
                ml_prediction TEXT NOT NULL,
                human_feedback TEXT,
                human_comments TEXT,
                reviewed_by TEXT,
                reviewed_at TIMESTAMP,
                is_training_data BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # ตารางสินค้าที่ไม่ซ้ำ
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS unique_products (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                cluster_id TEXT,
                is_representative BOOLEAN DEFAULT FALSE,
                confidence REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # ตารางประวัติการเรียนรู้ของโมเดล
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                feedback_count INTEGER,
                accuracy_before REAL,
                accuracy_after REAL,
                model_version TEXT,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_comparison(self, comparison: ProductComparison):
        """บันทึกการเปรียบเทียบสินค้า"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO product_comparisons 
            (id, product1, product2, similarity_score, confidence_score, 
             ml_prediction, human_feedback, human_comments, reviewed_by, 
             reviewed_at, is_training_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            comparison.id, comparison.product1, comparison.product2,
            comparison.similarity_score, comparison.confidence_score,
            comparison.ml_prediction.value,
            comparison.human_feedback.value if comparison.human_feedback else None,
            comparison.human_comments, comparison.reviewed_by,
            comparison.reviewed_at, comparison.is_training_data
        ))
        
        conn.commit()
        conn.close()
    
    def get_pending_reviews(self, limit: int = 50) -> List[ProductComparison]:
        """ดึงรายการที่รอการตรวจสอบ"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM product_comparisons 
            WHERE human_feedback IS NULL 
            ORDER BY confidence_score ASC, similarity_score DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        comparisons = []
        for row in rows:
            comp = ProductComparison(
                id=row[0], product1=row[1], product2=row[2],
                similarity_score=row[3], confidence_score=row[4],
                ml_prediction=FeedbackType(row[5]),
                human_feedback=FeedbackType(row[6]) if row[6] else None,
                human_comments=row[7], reviewed_by=row[8],
                reviewed_at=row[9], is_training_data=row[10]
            )
            comparisons.append(comp)
        
        return comparisons
    
    def get_training_data(self) -> List[ProductComparison]:
        """ดึงข้อมูลที่มี human feedback สำหรับการเทรน"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM product_comparisons 
            WHERE human_feedback IS NOT NULL
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        comparisons = []
        for row in rows:
            comp = ProductComparison(
                id=row[0], product1=row[1], product2=row[2],
                similarity_score=row[3], confidence_score=row[4],
                ml_prediction=FeedbackType(row[5]),
                human_feedback=FeedbackType(row[6]),
                human_comments=row[7], reviewed_by=row[8],
                reviewed_at=row[9], is_training_data=row[10]
            )
            comparisons.append(comp)
        
        return comparisons


class ProductDeduplicationSystem:
    """ระบบหาสินค้าที่ไม่ซ้ำ"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.pipeline = self._init_pipeline()
        self.db = HumanFeedbackDatabase()
        
    def _init_pipeline(self) -> ProductSimilarityPipeline:
        """สร้าง pipeline สำหรับคำนวณความคล้าย"""
        config = Phase4Config()
        config.similarity_threshold = self.similarity_threshold
        
        # สร้างคอมโพเนนต์
        data_source = ComponentFactory.create_data_source("csv")
        data_sink = ComponentFactory.create_data_sink("csv")
        text_processor = ComponentFactory.create_text_processor("thai")
        embedding_model = ComponentFactory.create_embedding_model("sentence_transformer")
        similarity_calculator = ComponentFactory.create_similarity_calculator("cosine")
        
        return ProductSimilarityPipeline(
            data_source=data_source,
            data_sink=data_sink,
            text_processor=text_processor,
            embedding_model=embedding_model,
            similarity_calculator=similarity_calculator,
            config=config
        )
    
    def find_potential_duplicates(self, products: List[str]) -> List[ProductComparison]:
        """หาสินค้าที่อาจซ้ำกัน"""
        logger.info(f"🔍 วิเคราะห์ {len(products)} สินค้าเพื่อหาสินค้าที่ซ้ำ...")
        
        # สร้าง embeddings
        embeddings = self.pipeline.embedding_model.create_embeddings(products)
        
        # หาคู่ที่ใกล้เคียง
        similarity_matrix = self.pipeline.similarity_calculator.compute_similarity(
            embeddings, embeddings
        )
        
        comparisons = []
        
        for i in range(len(products)):
            for j in range(i + 1, len(products)):
                similarity = float(similarity_matrix[i, j])
                
                # สร้างการเปรียบเทียบสำหรับสินค้าที่มีความคล้ายสูง
                if similarity > 0.5:  # ความคล้ายขั้นต่ำ
                    comparison_id = f"comp_{i}_{j}_{int(similarity * 10000)}"
                    
                    # ทำนายโดย ML
                    ml_prediction = self._predict_relationship(similarity)
                    confidence = self._calculate_confidence(similarity)
                    
                    comparison = ProductComparison(
                        id=comparison_id,
                        product1=products[i],
                        product2=products[j],
                        similarity_score=similarity,
                        confidence_score=confidence,
                        ml_prediction=ml_prediction
                    )
                    
                    comparisons.append(comparison)
                    self.db.save_comparison(comparison)
        
        logger.info(f"✅ พบการเปรียบเทียบ {len(comparisons)} คู่")
        return comparisons
    
    def _predict_relationship(self, similarity: float) -> FeedbackType:
        """ทำนายความสัมพันธ์ระหว่างสินค้าโดย ML"""
        if similarity >= 0.9:
            return FeedbackType.DUPLICATE
        elif similarity >= 0.7:
            return FeedbackType.SIMILAR
        elif similarity >= 0.5:
            return FeedbackType.DIFFERENT
        else:
            return FeedbackType.DIFFERENT
    
    def _calculate_confidence(self, similarity: float) -> float:
        """คำนวณความมั่นใจในการทำนาย"""
        if similarity >= 0.95 or similarity <= 0.3:
            return 0.9  # มั่นใจสูง
        elif similarity >= 0.85 or similarity <= 0.5:
            return 0.7  # มั่นใจปานกลาง
        else:
            return 0.4  # มั่นใจต่ำ (ต้องให้มนุษย์ตรวจ)
    
    def extract_unique_products(self, products: List[str]) -> List[UniqueProduct]:
        """สกัดสินค้าที่ไม่ซ้ำ"""
        comparisons = self.find_potential_duplicates(products)
        
        # สร้างกลุ่มสินค้าที่ซ้ำ
        clusters = self._create_clusters(products, comparisons)
        
        unique_products = []
        for cluster_id, cluster_products in clusters.items():
            # เลือกสินค้าที่เป็นตัวแทนของกลุ่ม
            representative = self._select_representative(cluster_products)
            
            for i, product in enumerate(cluster_products):
                unique_product = UniqueProduct(
                    id=f"unique_{cluster_id}_{i}",
                    name=product,
                    cluster_id=cluster_id,
                    is_representative=(product == representative),
                    similar_products=[p for p in cluster_products if p != product],
                    confidence=0.8 if product == representative else 0.6
                )
                unique_products.append(unique_product)
        
        return unique_products
    
    def _create_clusters(self, products: List[str], comparisons: List[ProductComparison]) -> Dict[str, List[str]]:
        """สร้างกลุ่มสินค้าที่คล้ายกัน"""
        clusters = {}
        product_to_cluster = {}
        cluster_counter = 0
        
        for comparison in comparisons:
            if comparison.ml_prediction == FeedbackType.DUPLICATE:
                p1, p2 = comparison.product1, comparison.product2
                
                if p1 in product_to_cluster and p2 in product_to_cluster:
                    # ทั้งคู่อยู่ในกลุ่มแล้ว - รวมกลุ่ม
                    c1, c2 = product_to_cluster[p1], product_to_cluster[p2]
                    if c1 != c2:
                        # รวมกลุ่ม c2 เข้า c1
                        clusters[c1].extend(clusters[c2])
                        for p in clusters[c2]:
                            product_to_cluster[p] = c1
                        del clusters[c2]
                
                elif p1 in product_to_cluster:
                    # เพิ่ม p2 เข้ากลุ่มของ p1
                    cluster_id = product_to_cluster[p1]
                    clusters[cluster_id].append(p2)
                    product_to_cluster[p2] = cluster_id
                
                elif p2 in product_to_cluster:
                    # เพิ่ม p1 เข้ากลุ่มของ p2
                    cluster_id = product_to_cluster[p2]
                    clusters[cluster_id].append(p1)
                    product_to_cluster[p1] = cluster_id
                
                else:
                    # สร้างกลุ่มใหม่
                    cluster_id = f"cluster_{cluster_counter}"
                    clusters[cluster_id] = [p1, p2]
                    product_to_cluster[p1] = cluster_id
                    product_to_cluster[p2] = cluster_id
                    cluster_counter += 1
        
        # สินค้าที่ไม่อยู่ในกลุ่มใด ให้สร้างกลุ่มเดี่ยว
        for product in products:
            if product not in product_to_cluster:
                cluster_id = f"cluster_{cluster_counter}"
                clusters[cluster_id] = [product]
                product_to_cluster[product] = cluster_id
                cluster_counter += 1
        
        return clusters
    
    def _select_representative(self, cluster_products: List[str]) -> str:
        """เลือกสินค้าที่เป็นตัวแทนของกลุ่ม"""
        if len(cluster_products) == 1:
            return cluster_products[0]
        
        # เลือกสินค้าที่มีชื่อสั้นที่สุด (โดยทั่วไปจะเป็นชื่อหลัก)
        return min(cluster_products, key=len)


class HumanReviewInterface:
    """อินเทอร์เฟซสำหรับให้มนุษย์ตรวจสอบ"""
    
    def __init__(self, db: HumanFeedbackDatabase):
        self.db = db
    
    def start_review_session(self, reviewer_name: str, batch_size: int = 10):
        """เริ่มเซสชันการตรวจสอบ"""
        print(f"\n🔍 เริ่มเซสชันตรวจสอบโดย: {reviewer_name}")
        print("=" * 60)
        
        pending_reviews = self.db.get_pending_reviews(batch_size)
        
        if not pending_reviews:
            print("✅ ไม่มีรายการที่ต้องตรวจสอบ")
            return
        
        reviewed_count = 0
        for comparison in pending_reviews:
            print(f"\n📝 รายการที่ {reviewed_count + 1}/{len(pending_reviews)}")
            print("-" * 40)
            print(f"สินค้า 1: {comparison.product1}")
            print(f"สินค้า 2: {comparison.product2}")
            print(f"ความคล้าย: {comparison.similarity_score:.3f}")
            print(f"ความมั่นใจ: {comparison.confidence_score:.3f}")
            print(f"ML ทำนาย: {comparison.ml_prediction.value}")
            
            # รับ feedback จากมนุษย์
            feedback = self._get_human_feedback()
            if feedback is None:
                break  # ออกจากเซสชัน
            
            comments = input("ความคิดเห็นเพิ่มเติม (ไม่บังคับ): ").strip()
            
            # บันทึก feedback
            comparison.human_feedback = feedback
            comparison.human_comments = comments if comments else None
            comparison.reviewed_by = reviewer_name
            comparison.reviewed_at = datetime.now()
            comparison.is_training_data = True
            
            self.db.save_comparison(comparison)
            reviewed_count += 1
            
            print(f"✅ บันทึก feedback แล้ว")
        
        print(f"\n🎯 ตรวจสอบเสร็จแล้ว {reviewed_count} รายการ")
    
    def _get_human_feedback(self) -> Optional[FeedbackType]:
        """รับ feedback จากมนุษย์"""
        print("\nโปรดเลือก:")
        print("1. สินค้าซ้ำ (duplicate)")
        print("2. สินค้าคล้าย แต่ไม่ซ้ำ (similar)")
        print("3. สินค้าต่างกัน (different)")
        print("4. ไม่แน่ใจ (uncertain)")
        print("0. ออกจากเซสชัน")
        
        while True:
            choice = input("เลือก (1-4, 0=ออก): ").strip()
            
            if choice == "0":
                return None
            elif choice == "1":
                return FeedbackType.DUPLICATE
            elif choice == "2":
                return FeedbackType.SIMILAR
            elif choice == "3":
                return FeedbackType.DIFFERENT
            elif choice == "4":
                return FeedbackType.UNCERTAIN
            else:
                print("❌ กรุณาเลือก 1-4 หรือ 0")


def main():
    """ฟังก์ชันหลักสำหรับทดสอบระบบ"""
    print("🤖 Human-in-the-Loop Product Deduplication System")
    print("=" * 60)
    
    # ตัวอย่างข้อมูลสินค้า
    sample_products = [
        "iPhone 14 Pro Max 256GB สีดำ",
        "iPhone 14 Pro Max 256GB Black",
        "iPhone 14 Pro Max 256 GB ดำ",
        "Samsung Galaxy S23 Ultra",
        "Samsung Galaxy S23 Ultra 256GB",
        "MacBook Pro 14 inch M2",
        "MacBook Pro 14\" M2 Chip",
        "iPad Air 5th Generation",
        "iPad Air 5 64GB WiFi",
        "AirPods Pro 2nd Generation"
    ]
    
    # สร้างระบบ
    system = ProductDeduplicationSystem(similarity_threshold=0.8)
    
    # หาสินค้าที่อาจซ้ำ
    print("\n🔍 ขั้นตอนที่ 1: หาสินค้าที่อาจซ้ำกัน")
    comparisons = system.find_potential_duplicates(sample_products)
    
    print(f"พบการเปรียบเทียบ {len(comparisons)} คู่")
    
    # แสดงผลการทำนายของ ML
    print("\n🤖 การทำนายของ ML:")
    for comp in comparisons[:5]:  # แสดง 5 อันแรก
        print(f"• {comp.product1}")
        print(f"  vs {comp.product2}")
        print(f"  ความคล้าย: {comp.similarity_score:.3f}")
        print(f"  ทำนาย: {comp.ml_prediction.value}")
        print(f"  ความมั่นใจ: {comp.confidence_score:.3f}")
        print()
    
    # เริ่ม session การตรวจสอบ (แบบจำลอง)
    print("\n👤 ขั้นตอนที่ 2: การตรวจสอบโดยมนุษย์")
    review_interface = HumanReviewInterface(system.db)
    
    # ในการใช้งานจริง ให้เรียก:
    # review_interface.start_review_session("reviewer_name")
    
    # สร้างสินค้าที่ไม่ซ้ำ
    print("\n📦 ขั้นตอนที่ 3: สกัดสินค้าที่ไม่ซ้ำ")
    unique_products = system.extract_unique_products(sample_products)
    
    print(f"พบสินค้าที่ไม่ซ้ำ {len(unique_products)} รายการ")
    
    # แสดงผลลัพธ์
    clusters = {}
    for product in unique_products:
        if product.cluster_id not in clusters:
            clusters[product.cluster_id] = []
        clusters[product.cluster_id].append(product)
    
    print(f"\nจัดกลุ่มเป็น {len(clusters)} กลุ่ม:")
    for cluster_id, products in clusters.items():
        print(f"\n🏷️ {cluster_id}:")
        for product in products:
            marker = "👑" if product.is_representative else "  "
            print(f"{marker} {product.name}")


if __name__ == "__main__":
    main()

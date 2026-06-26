#!/usr/bin/env python3
"""
Human-in-the-Loop Product Deduplication System (Refactored)
==========================================================

โมเดลและประเภทข้อมูลสำหรับการตรวจสินค้าซ้ำ (Deduplication)
- เก็บรักษาคลาส FeedbackType, ProductComparison, UniqueProduct ที่ระบบ ML และ API ต้องใช้
- ลบโค้ดตระกูล SQLite, CLI และสถาปัตยกรรม Batch ที่ตกค้างออกทั้งหมด
"""

from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

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
    product1_cleaned: str  # เพิ่มข้อความที่ทำความสะอาดแล้ว
    product2_cleaned: str  # เพิ่มข้อความที่ทำความสะอาดแล้ว
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

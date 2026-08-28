#!/usr/bin/env python3
"""
Taxonomy Service
================
Centralized service for managing taxonomy nodes, keyword rules, and reference products.
Consolidates Supabase interactions to eliminate code duplication across scripts.
"""

import json
import logging
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Import ThaiTextProcessor
from src.core.fresh_implementations import (
    ThaiTextProcessor,
    merge_short_token_runs,
    tokenize_thai,
)

logger = logging.getLogger(__name__)

class TaxonomyService:
    """Service to handle all taxonomy and rule-related operations with Supabase."""

    # keyword ที่สั้นกว่านี้จะ match กลางคำจนจัดหมวดผิด (เช่น "สี" ใน "ยาสีฟัน")
    MIN_KEYWORD_LENGTH = 3

    # หน่วยวัดที่หลุดมาเป็น token แยกหลังตัดตัวเลขออก เช่น "1500g" -> "1500" + "g"
    # ถ้าไม่กรองจะไปเกาะกับคำถัดไปกลายเป็น keyword เพี้ยน เช่น "gออริจิ"
    MEASUREMENT_UNITS = {
        'g', 'kg', 'mg', 'ml', 'l', 'cc', 'oz', 'lb', 'pcs', 'pc',
        'ก', 'กก', 'กรัม', 'มล', 'ลิตร', 'ซม', 'มม', 'นิ้ว', 'ห่อ', 'ชิ้น',
    }

    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.category_names = {}
        self.category_embeddings = {}
        self.keyword_rules = []
        self.reference_lessons = []
        self.processor = ThaiTextProcessor()
        
        # Stopwords for Thai products
        self.product_stopwords = {
            'ขนาด', 'ราคาพิเศษ', 'ตรา', 'รส', 'สูตร', 'แพ็ค', 'กล่อง', 
            'แถม', 'ฟรี', 'ราคา', 'บาท', 'จำนวน', 'จำกัด', 'จัดชุด',
            'ประหยัด', 'สุดคุ้ม', 'ของแท้', 'ส่งฟรี', 'โปรโมชั่น'
        }
        
    def extract_auto_keywords(self, product_name: str) -> List[str]:
        """
        Extract meaningful keywords from a product name for auto-learning.
        Removes measurements (g, ml, kg), numbers, and common Thai product stopwords.
        """
        if not product_name:
            return []
            
        # 1. Clean with ThaiTextProcessor
        cleaned = self.processor.process(product_name)

        # 2. แบ่งตามช่องว่างที่คนพิมพ์ไว้ก่อน — เป็นขอบเขตคำที่เชื่อถือได้ที่สุด
        #    ถ้าไม่แบ่ง เศษของ "บรีส" จะไปเกาะกับ "เอกเซล" กลายเป็น "บรีสเอก"
        segments: List[List[str]] = []
        for chunk in cleaned.split():
            # ตัดคำจริง ไม่ใช่ split() เพราะภาษาไทยไม่เว้นวรรคภายในคำ
            # split() เดิมให้ token ก้อนเดียวติดขนาดมาด้วย เช่น "ยาสีฟันดอกบัวคู่150g"
            # ซึ่งไม่มีวันไป match สินค้าตัวอื่น ทำให้ระบบเรียนจาก UI แล้วไม่ได้อะไรเลย
            segments.append([])
            for token in tokenize_thai(chunk):
                # ตัวเลขและหน่วยวัดเป็นเส้นแบ่งด้วย ไม่งั้น "150g ออริจิ" -> "gออริจิ"
                # และ "โอโม่2100g" จะเสีย "โอ" ไปจนเหลือแค่ "โม่"
                if re.search(r'\d', token) or token in self.MEASUREMENT_UNITS:
                    segments.append([])
                    continue
                segments[-1].append(token)

        extracted = []
        for segment in segments:
            # รวมเศษแบรนด์ที่พจนานุกรมไม่รู้จักกลับเป็นคำเดียว เช่น บ|รี|ส -> บรีส
            for token in merge_short_token_runs(segment, self.MIN_KEYWORD_LENGTH):
                if token in self.product_stopwords:
                    continue

                # เศษคำสั้นเกินไปจะ match มั่ว (บทเรียนจาก "สี" ที่ไป match ใน "ยาสีฟัน")
                if len(token) < self.MIN_KEYWORD_LENGTH:
                    continue

                # "รสต้มยำกุ้ง" -> เก็บเฉพาะส่วนที่บอกรสชาติ
                if token.startswith('รส') and len(token) > 2:
                    extracted.append(token[2:])
                    continue

                extracted.append(token)

        return extracted

    def learn_from_verified_product(self, product_name: str, category_id: str) -> Dict[str, Any]:
        """
        Auto-learn new keyword rules from a verified product.
        Optimized for ARRAY based keywords column.
        """
        new_keywords = self.extract_auto_keywords(product_name)
        if not new_keywords:
            return {"status": "skipped", "message": "No meaningful keywords found"}
            
        try:
            # 1. Look for an existing 'auto_learned' rule for this category
            res = self.supabase.table("keyword_rules")\
                .select("*")\
                .eq("category_id", category_id)\
                .eq("match_type", "auto_learned")\
                .execute()
            
            from datetime import datetime
            
            if res.data:
                # Update existing rule
                existing_rule = res.data[0]
                existing_keywords = set(existing_rule.get("keywords") or [])
                
                # Filter for truly new keywords
                to_add = [kw for kw in new_keywords if kw not in existing_keywords and len(kw) >= 2]
                
                if not to_add:
                    return {"status": "no_change", "message": "Keywords already exist"}
                    
                updated_keywords = list(existing_keywords.union(to_add))
                self.supabase.table("keyword_rules")\
                    .update({"keywords": updated_keywords, "updated_at": datetime.now().isoformat()})\
                    .eq("id", existing_rule["id"])\
                    .execute()
                    
                return {
                    "status": "updated",
                    "new_keywords": to_add,
                    "total_keywords": len(updated_keywords)
                }
            else:
                # Create new rule
                to_add = [kw for kw in new_keywords if len(kw) >= 2]
                if not to_add:
                    return {"status": "skipped", "message": "Keywords too short"}
                    
                import uuid
                auto_code = f"auto_kw_{uuid.uuid4().hex[:8]}"
                
                rule_data = {
                    "code": auto_code,
                    "name": f"Auto-learned for Category {category_id[:8]}",
                    "keywords": to_add,
                    "category_id": category_id,
                    "match_type": "auto_learned",
                    "priority": 5, # Low priority
                    "is_active": True,
                    "created_at": datetime.now().isoformat()
                }
                
                self.supabase.table("keyword_rules").insert(rule_data).execute()
                
                return {
                    "status": "created",
                    "code": auto_code,
                    "new_keywords": to_add
                }
                
        except Exception as e:
            logger.error(f"Auto-learning failed: {e}")
            return {"status": "error", "message": str(e)}

    def load_taxonomy_nodes(self) -> bool:
        """Load category nodes and their embeddings."""
        try:
            print("📚 Loading Taxonomy Nodes...")
            res = self.supabase.table("taxonomy_nodes").select("id, name_th, embedding, keywords").execute()
            
            self.category_names = {node['id']: node['name_th'] for node in res.data}
            
            for node in res.data:
                if node.get('embedding'):
                    emb = node['embedding']
                    if isinstance(emb, str):
                        emb = json.loads(emb)
                    self.category_embeddings[node['id']] = np.array(emb)
            
            print(f"✅ Loaded {len(self.category_names)} categories.")
            return True
        except Exception as e:
            logger.error(f"Failed to load taxonomy nodes: {e}")
            return False
            
    def load_keyword_rules(self) -> bool:
        """Load keyword classification rules."""
        try:
            print("📚 Loading Keyword Rules...")
            res = self.supabase.table("keyword_rules").select("*").execute()
            self.keyword_rules = res.data
            print(f"✅ Loaded {len(self.keyword_rules)} keyword rules.")
            return True
        except Exception as e:
            logger.error(f"Failed to load keyword rules: {e}")
            return False
            
    def load_reference_lessons(self) -> bool:
        """Load approved products (lessons) as reference data."""
        try:
            print("📖 Loading reference lessons (approved products)...")
            res = self.supabase.table("products").select("id, name_th, category_id, embedding").eq("status", "approved").execute()
            
            processed_lessons = []
            for lesson in res.data:
                if lesson.get('embedding'):
                    emb = lesson['embedding']
                    if isinstance(emb, str):
                        emb = json.loads(emb)
                    lesson['embedding_arr'] = np.array(emb)
                    processed_lessons.append(lesson)
            
            self.reference_lessons = processed_lessons
            print(f"✅ Loaded {len(self.reference_lessons)} reference lessons.")
            return True
        except Exception as e:
            logger.error(f"Failed to load reference lessons: {e}")
            return False
            
    def load_all_metadata(self) -> bool:
        """Load all metadata required for classification."""
        success = self.load_taxonomy_nodes()
        success &= self.load_keyword_rules()
        success &= self.load_reference_lessons()
        return success

    def get_category_name(self, category_id: str) -> str:
        """Get Thai name for a category ID."""
        return self.category_names.get(category_id, "Unknown")
        
    def get_all_categories(self) -> Dict[str, str]:
        """Get mapping of all category IDs to names."""
        return self.category_names

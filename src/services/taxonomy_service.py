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
from src.core.fresh_implementations import ThaiTextProcessor

logger = logging.getLogger(__name__)

class TaxonomyService:
    """Service to handle all taxonomy and rule-related operations with Supabase."""
    
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
        
        # 2. Tokenize (Simple split by space since ThaiTextProcessor already added spaces)
        tokens = cleaned.split()
        
        extracted = []
        for token in tokens:
            # Skip if token is purely numeric or looks like measurement (e.g., 600มล., 1.5kg)
            if re.match(r'^[\d\.]+$', token):
                continue
            if re.search(r'\d+', token) and any(unit in token for unit in ['ก.', 'กรัม', 'มล.', 'ลิตร', 'kg', 'g', 'ml', 'l']):
                continue
            if re.match(r'^\d+[กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮa-zA-Z\.]+$', token):
                # Likely measurement like 60ก., 500ml
                continue
                
            # Skip if it's a stopword
            if token in self.product_stopwords:
                continue
                
            # Clean "รส" prefix if it exists as a standalone token or part of token
            # But "รสต้มยำกุ้ง" should stay as one if possible or split
            if token.startswith('รส') and len(token) > 2:
                # Keep the flavor part
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

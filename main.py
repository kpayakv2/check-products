import pandas as pd
from sentence_transformers import SentenceTransformer, util

# โหลดชื่อสินค้าเดิมและใหม่จากไฟล์แยกกัน
old_products = pd.read_csv('D:/product_checker/cleaned_products.csv')
new_products = pd.read_csv('D:/bill26668/_ocr_output/merged_receipts.csv')
old_product_names = old_products['name'].tolist()

# ตรวจสอบและลบรายการสินค้าซ้ำในข้อมูลสินค้าใหม่
if new_products.duplicated(subset=['รายการ']).any():
    duplicate_rows = new_products[new_products.duplicated(subset=['รายการ'], keep=False)]
    duplicate_rows.to_csv('d:/product_checker/duplicate_new_products.csv', index=False, encoding='utf-8-sig')
    print(f"พบสินค้าซ้ำ {len(duplicate_rows)} รายการ บันทึกที่ duplicate_new_products.csv")

new_products = new_products.drop_duplicates(subset=['รายการ'])
new_product_names = new_products['รายการ'].tolist()

# โหลดโมเดล pre-trained
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# สร้าง embeddings ของชื่อสินค้าเดิม
old_embeddings = model.encode(old_product_names, convert_to_tensor=True)

def check_product_similarity(new_product, old_product_names, old_embeddings, model, top_k=3):
    new_embedding = model.encode([new_product], convert_to_tensor=True)
    cos_scores = util.cos_sim(new_embedding, old_embeddings)[0]
    top_results = cos_scores.topk(k=top_k)
    result = []
    for score, idx in zip(top_results[0], top_results[1]):
        result.append((old_product_names[idx], float(score)))
    return result

if __name__ == "__main__":
    output_rows = []
    for new_product in new_product_names:
        results = check_product_similarity(new_product, old_product_names, old_embeddings, model)
        for name, score in results:
            output_rows.append({
                'new_product': new_product,
                'matched_old_product': name,
                'score': score
            })
    # ส่งออกผลลัพธ์เป็นไฟล์ CSV
    output_df = pd.DataFrame(output_rows)
output_df.to_csv('d:/product_checker/matched_products.csv', index=False, encoding='utf-8-sig')
print('บันทึกผลลัพธ์ที่ d:/product_checker/matched_products.csv เรียบร้อยแล้ว (encoding utf-8-sig สำหรับ Excel)')

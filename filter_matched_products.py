import pandas as pd

df = pd.read_csv("d:/product_checker/matched_products.csv")

# สินค้าที่ต้องตรวจสอบ (score >= 0.90)
check_df = df[df["score"] >= 0.90].sort_values(by="score", ascending=False)
check_df.to_csv("d:/product_checker/matched_products_check.csv", index=False, encoding="utf-8-sig")

# สินค้าที่ไม่ซ้ำ (score < 0.90)
unique_df = df[df["score"] < 0.90].sort_values(by="score", ascending=False)
unique_df.to_csv(
    "d:/product_checker/matched_products_unique.csv", index=False, encoding="utf-8-sig"
)

print("บันทึกไฟล์ matched_products_check.csv และ matched_products_unique.csv เรียบร้อยแล้ว")

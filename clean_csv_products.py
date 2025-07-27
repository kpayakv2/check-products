import pandas as pd

# ข้ามแถวหัวรายงาน 4 แถวแรก (header=3 เพราะนับจาก 0)
df = pd.read_csv("d:/product_checker/สินค้าในpos.csv", header=3)

# เปลี่ยนชื่อคอลัมน์ 'รายการ' เป็น 'name'
df = df.rename(columns={"รายการ": "name"})

if "name" not in df.columns:
    print("Columns:", df.columns.tolist())
    print("กรุณาตรวจสอบชื่อคอลัมน์สินค้าในไฟล์ CSV แล้วแก้ไขบรรทัด df.rename(...) ให้ถูกต้อง")
else:
    df["name"] = df["name"].astype(str).str.strip()
    # กรองเฉพาะชื่อสินค้าที่ไม่ว่าง, ไม่ใช่ NaN, ไม่ใช่หัวตาราง
    df = df[~df["name"].str.lower().str.contains("nan|^$|วันที่ เวลา สร้างสินค้า", na=True)]
    df = df.drop_duplicates(subset=["name"])
    # ส่งออกเฉพาะคอลัมน์ name เท่านั้น
    df[["name"]].to_csv(
        "d:/product_checker/cleaned_products.csv", index=False, encoding="utf-8-sig"
    )
    print("บันทึกไฟล์ cleaned_products.csv เฉพาะคอลัมน์ชื่อสินค้าแล้ว")

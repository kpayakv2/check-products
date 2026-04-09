import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

file_path = 'input/รายการสินค้าพร้อมหมวดหมู่_AI.txt'

try:
    with open(file_path, 'r', encoding='utf-16') as f:
        content = f.read()
    
    # Show first 20 lines
    lines = content.split('\n')
    for line in lines[:20]:
        print(line)
except Exception as e:
    print(f"Error: {e}")

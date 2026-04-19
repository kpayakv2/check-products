import os
import re
from pathlib import Path

def audit_frontend(base_path="taxonomy-app"):
    report = ["# Frontend Logic & Integrity Audit Report\n"]
    
    app_path = Path(base_path) / "app"
    components_path = Path(base_path) / "components"
    
    # 1. Route Map
    report.append("## 🗺️ Route & Page Inventory")
    report.append("| Route | File Path | Type | Thai Optimized? |")
    report.append("|-------|-----------|------|-----------------|")
    
    for root, dirs, files in os.walk(app_path):
        if "page.tsx" in files:
            rel_path = os.path.relpath(root, app_path)
            route = "/" if rel_path == "." else "/" + rel_path.replace("\\", "/")
            file_path = os.path.join(root, "page.tsx")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                is_client = "'use client'" in content
                has_thai_class = "thai-text" in content or "notoSansThai" in content
                type_str = "Client" if is_client else "Server"
                thai_str = "✅ Yes" if has_thai_class else "⚠️ No"
                report.append(f"| {route} | {rel_path}/page.tsx | {type_str} | {thai_str} |")

    # 2. Component Inventory
    report.append("\n## 🧩 Component Analysis")
    report.append("| Category | Component Name | API Calls? | Thai Support? |")
    report.append("|----------|----------------|------------|---------------|")
    
    for root, dirs, files in os.walk(components_path):
        for file in files:
            if file.endswith((".tsx", ".js")):
                comp_name = file
                category = os.path.basename(root)
                file_path = os.path.join(root, file)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    has_api = "fetch" in content or "supabase" in content
                    has_thai = "thai-text" in content or "font-" in content
                    api_str = "🔗 Yes" if has_api else "No"
                    thai_str = "🇹🇭 Yes" if has_thai else "⚠️ No"
                    report.append(f"| {category} | {comp_name} | {api_str} | {thai_str} |")

    # 3. Global Rules Check
    report.append("\n## ⚖️ Global Rules Compliance (GEMINI.md)")
    
    # Check for Sarabun or Thai fonts in globals.css
    globals_css = Path(base_path) / "app" / "globals.css"
    if globals_css.exists():
        with open(globals_css, 'r', encoding='utf-8') as f:
            css_content = f.read()
            if "IBM Plex Sans Thai" in css_content or "Noto Sans Thai" in css_content:
                report.append("- ✅ Thai Typography: Found IBM Plex/Noto Sans Thai in globals.css")
            else:
                report.append("- ❌ Thai Typography: Thai fonts NOT found in globals.css")
            
            if "line-height" in css_content and "1.6" in css_content:
                report.append("- ✅ Antigravity: Line-height optimization found for Thai text")
            else:
                report.append("- ⚠️ Antigravity: Line-height might be too tight for Thai characters")

    return "\n".join(report)

if __name__ == "__main__":
    audit_results = audit_frontend()
    with open("FRONTEND_AUDIT_REPORT.md", "w", encoding="utf-8") as f:
        f.write(audit_results)
    print("Audit complete! Results saved to FRONTEND_AUDIT_REPORT.md")

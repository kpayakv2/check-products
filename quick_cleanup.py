"""Quick cleanup for duplicate files."""
import os
from pathlib import Path

# Files to delete (legacy and duplicates)
files_to_delete = [
    "main_legacy.py",
    "main_legacy_legacy.py", 
    "main_new.py",
    "main_new_legacy.py",
    "main_production.py",  # We have main_phase4.py and api_server.py now
    "clean_csv_products_legacy.py",
    "filter_matched_products_legacy.py",
    "filter_matched_products_legacy_legacy.py",
    "debug_sklearn.py",
    "debug_step.py",
    "test_minimal.py",
    "test_basic.py", 
    "test_imports.py",
    "test_architecture.py",
    "test_core_components.py",
    "test_data_loaders.py",
    "test_end_to_end.py",
    "test_fresh_pipeline.py",
    "phase4_advanced.py",  # We have main_phase4.py now
    "advanced_models.py",  # Not used in current implementation
    "benchmark_phase4.py",
    "benchmark_comparison.py"
]

# Directories with duplicates
duplicate_dirs = [
    "src/config/settings_simple.py",
    "src/config/settings_backup.py"
]

project_root = Path("D:/product_checker/check-products")

print("🧹 Quick Cleanup of Duplicate Files")
print("=" * 40)

deleted_count = 0
errors = []

# Delete individual files
for filename in files_to_delete:
    file_path = project_root / filename
    if file_path.exists():
        try:
            os.remove(file_path)
            print(f"✅ Deleted: {filename}")
            deleted_count += 1
        except Exception as e:
            print(f"❌ Error deleting {filename}: {e}")
            errors.append(f"{filename}: {e}")
    else:
        print(f"⚠️  Not found: {filename}")

# Delete duplicate config files
for filepath in duplicate_dirs:
    file_path = project_root / filepath
    if file_path.exists():
        try:
            os.remove(file_path)
            print(f"✅ Deleted: {filepath}")
            deleted_count += 1
        except Exception as e:
            print(f"❌ Error deleting {filepath}: {e}")
            errors.append(f"{filepath}: {e}")

print(f"\n📊 Cleanup Summary:")
print(f"   Deleted: {deleted_count} files")
print(f"   Errors: {len(errors)}")

if errors:
    print(f"\n❌ Errors encountered:")
    for error in errors:
        print(f"   - {error}")

print(f"\n✅ Cleanup complete!")

#!/usr/bin/env python3
"""
Code Cleanup Script - Remove Duplicate and Legacy Files
======================================================

Identify and remove redundant code files to clean up the project.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict
import fnmatch


def find_duplicate_files() -> Dict[str, List[str]]:
    """Find files that appear to be duplicates or legacy versions."""
    
    project_root = Path(__file__).parent
    duplicates = {
        "legacy_files": [],
        "backup_files": [],
        "duplicate_configs": [],
        "old_main_files": [],
        "test_duplicates": [],
        "debug_files": []
    }
    
    # Patterns for different types of duplicates
    legacy_patterns = ["*legacy*", "*_old*", "*_backup*"]
    debug_patterns = ["debug_*", "*_debug*", "test_debug*"]
    old_main_patterns = ["main_new*", "main_production*"]  # Keep main.py and main_phase4.py
    
    for file_path in project_root.rglob("*.py"):
        file_name = file_path.name
        relative_path = str(file_path.relative_to(project_root))
        
        # Check for legacy files
        for pattern in legacy_patterns:
            if fnmatch.fnmatch(file_name, pattern):
                duplicates["legacy_files"].append(relative_path)
                break
        
        # Check for debug files
        for pattern in debug_patterns:
            if fnmatch.fnmatch(file_name, pattern):
                duplicates["debug_files"].append(relative_path)
                break
        
        # Check for old main files (keep main.py, main_phase4.py, api_server.py)
        if file_name.startswith("main_") and file_name not in ["main.py", "main_phase4.py"]:
            duplicates["old_main_files"].append(relative_path)
        
        # Check for config duplicates
        if "config" in file_name and ("simple" in file_name or "backup" in file_name):
            duplicates["duplicate_configs"].append(relative_path)
        
        # Check for test duplicates
        if file_name.startswith("test_") and any(x in file_name for x in ["minimal", "basic", "imports"]):
            duplicates["test_duplicates"].append(relative_path)
    
    return duplicates


def analyze_file_content(file_path: Path) -> Dict[str, any]:
    """Analyze file content to determine if it's redundant."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "size": len(content),
            "lines": len(content.split('\n')),
            "has_main": '__name__ == "__main__"' in content,
            "has_class": 'class ' in content,
            "has_imports": 'import ' in content,
            "is_empty": len(content.strip()) < 50
        }
    except Exception as e:
        return {"error": str(e)}


def create_cleanup_plan() -> Dict[str, List[str]]:
    """Create a plan for cleaning up redundant files."""
    
    duplicates = find_duplicate_files()
    cleanup_plan = {
        "safe_to_delete": [],
        "review_needed": [],
        "keep": []
    }
    
    # Files that are safe to delete
    safe_patterns = [
        "main_legacy.py",
        "main_legacy_legacy.py", 
        "main_new.py",
        "main_new_legacy.py",
        "clean_csv_products_legacy.py",
        "filter_matched_products_legacy.py",
        "filter_matched_products_legacy_legacy.py",
        "debug_sklearn.py",
        "debug_step.py",
        "test_minimal.py",
        "test_basic.py",
        "test_imports.py"
    ]
    
    project_root = Path(__file__).parent
    
    for category, files in duplicates.items():
        for file_path in files:
            file_name = Path(file_path).name
            full_path = project_root / file_path
            
            if file_name in safe_patterns:
                cleanup_plan["safe_to_delete"].append(file_path)
            elif any(pattern in file_name for pattern in ["legacy", "backup", "_old"]):
                cleanup_plan["safe_to_delete"].append(file_path)
            elif "debug" in file_name.lower():
                cleanup_plan["safe_to_delete"].append(file_path)
            else:
                cleanup_plan["review_needed"].append(file_path)
    
    # Files to definitely keep
    keep_files = [
        "main.py",
        "main_phase4.py", 
        "api_server.py",
        "fresh_architecture.py",
        "fresh_implementations.py",
        "phase5_demo.py",
        "test_api_client.py"
    ]
    
    cleanup_plan["keep"] = keep_files
    
    return cleanup_plan


def perform_cleanup(cleanup_plan: Dict[str, List[str]], dry_run: bool = True) -> Dict[str, int]:
    """Perform the actual cleanup."""
    
    results = {
        "deleted": 0,
        "errors": 0,
        "skipped": 0
    }
    
    project_root = Path(__file__).parent
    
    print(f"🧹 Starting cleanup (dry_run={dry_run})...")
    
    for file_path in cleanup_plan["safe_to_delete"]:
        full_path = project_root / file_path
        
        if full_path.exists():
            print(f"{'[DRY RUN] Would delete' if dry_run else 'Deleting'}: {file_path}")
            
            if not dry_run:
                try:
                    full_path.unlink()
                    results["deleted"] += 1
                except Exception as e:
                    print(f"❌ Error deleting {file_path}: {e}")
                    results["errors"] += 1
            else:
                results["deleted"] += 1
        else:
            results["skipped"] += 1
    
    return results


def main():
    """Main cleanup function."""
    print("🔍 Analyzing Project for Duplicate and Legacy Files")
    print("=" * 60)
    
    # Find duplicates
    duplicates = find_duplicate_files()
    
    print("\n📋 Duplicate File Analysis:")
    total_duplicates = 0
    for category, files in duplicates.items():
        if files:
            print(f"\n{category.replace('_', ' ').title()}:")
            for file_path in files[:5]:  # Show first 5
                print(f"   - {file_path}")
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more")
            total_duplicates += len(files)
    
    print(f"\n📊 Total potentially redundant files: {total_duplicates}")
    
    # Create cleanup plan
    cleanup_plan = create_cleanup_plan()
    
    print(f"\n🧹 Cleanup Plan:")
    print(f"   Safe to delete: {len(cleanup_plan['safe_to_delete'])} files")
    print(f"   Need review: {len(cleanup_plan['review_needed'])} files")
    print(f"   Keep: {len(cleanup_plan['keep'])} files")
    
    print(f"\n📝 Files marked for deletion:")
    for file_path in cleanup_plan["safe_to_delete"]:
        print(f"   ❌ {file_path}")
    
    if cleanup_plan["review_needed"]:
        print(f"\n⚠️  Files needing manual review:")
        for file_path in cleanup_plan["review_needed"]:
            print(f"   ⚠️  {file_path}")
    
    # Perform dry run
    print(f"\n🔄 Performing dry run...")
    dry_results = perform_cleanup(cleanup_plan, dry_run=True)
    
    print(f"\n📊 Dry Run Results:")
    print(f"   Would delete: {dry_results['deleted']} files")
    print(f"   Errors: {dry_results['errors']}")
    print(f"   Skipped: {dry_results['skipped']}")
    
    # Ask for confirmation for actual cleanup
    print(f"\n❓ Would you like to perform the actual cleanup? (y/n)")
    
    # For script execution, let's save the cleanup plan
    cleanup_file = Path(__file__).parent / "cleanup_plan.json"
    import json
    with open(cleanup_file, 'w') as f:
        json.dump(cleanup_plan, f, indent=2)
    
    print(f"💾 Cleanup plan saved to: {cleanup_file}")
    print(f"🎯 To execute cleanup, run: python -c \"import json; from pathlib import Path; exec(open('cleanup_script.py').read().replace('dry_run=True', 'dry_run=False'))\"")
    
    return cleanup_plan


if __name__ == "__main__":
    cleanup_plan = main()
    print(f"\n✅ Analysis complete!")

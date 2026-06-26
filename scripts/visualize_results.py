#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Category Classification Results
==========================================

สร้างกราฟและ visualization จากผลการทดสอบ
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Set Thai font for matplotlib
matplotlib.rc('font', family='TH Sarabun New')

def load_latest_results():
    """โหลดผลลัพธ์ล่าสุด"""
    results_dir = Path("evaluation_results")
    
    if not results_dir.exists():
        print("❌ evaluation_results directory not found")
        return None
    
    # Find latest files
    files = {
        'keyword': sorted(results_dir.glob("category_eval_keyword_*.json"))[-1],
        'embedding': sorted(results_dir.glob("category_eval_embedding_*.json"))[-1],
        'hybrid': sorted(results_dir.glob("category_eval_hybrid_*.json"))[-1],
    }
    
    results = {}
    for method, filepath in files.items():
        with open(filepath, 'r', encoding='utf-8') as f:
            results[method] = json.load(f)
    
    return results


def plot_confidence_distribution(results, output_file='confidence_distribution.png'):
    """Plot confidence score distribution"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    methods = ['keyword', 'embedding', 'hybrid']
    colors = ['#3b82f6', '#10b981', '#f59e0b']
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        # Extract confidences
        confidences = [
            r['top_category']['confidence']
            for r in results[method]
            if r.get('top_category')
        ]
        
        # Plot histogram
        ax.hist(confidences, bins=20, color=colors[idx], alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(confidences), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
        
        ax.set_title(f'{method.capitalize()} Method', fontsize=14, fontweight='bold')
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_confidence_comparison(results, output_file='confidence_comparison.png'):
    """Compare confidence scores across methods"""
    
    methods = ['keyword', 'embedding', 'hybrid']
    colors = ['#3b82f6', '#10b981', '#f59e0b']
    
    # Extract statistics
    stats = {}
    for method in methods:
        confidences = [
            r['top_category']['confidence']
            for r in results[method]
            if r.get('top_category')
        ]
        stats[method] = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'median': np.median(confidences)
        }
    
    # Create box plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    data = [
        [r['top_category']['confidence'] for r in results[method] if r.get('top_category')]
        for method in methods
    ]
    
    bp = ax1.boxplot(data, labels=[m.capitalize() for m in methods], 
                     patch_artist=True, notch=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Confidence Score', fontsize=12)
    ax1.grid(alpha=0.3)
    
    # Bar plot for statistics
    x = np.arange(len(methods))
    width = 0.2
    
    ax2.bar(x - width, [stats[m]['mean'] for m in methods], width, 
            label='Mean', color=colors, alpha=0.8)
    ax2.bar(x, [stats[m]['median'] for m in methods], width, 
            label='Median', color=colors, alpha=0.6)
    ax2.bar(x + width, [stats[m]['max'] for m in methods], width, 
            label='Max', color=colors, alpha=0.4)
    
    ax2.set_title('Statistical Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Confidence Score', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.capitalize() for m in methods])
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_category_distribution(results, output_file='category_distribution.png'):
    """Plot category distribution for each method"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    methods = ['keyword', 'embedding', 'hybrid']
    colors = ['#3b82f6', '#10b981', '#f59e0b']
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        # Count categories
        categories = [
            r['top_category'].get('category_name', 'Unknown')
            for r in results[method]
            if r.get('top_category')
        ]
        
        category_counts = Counter(categories)
        top_categories = dict(category_counts.most_common(8))
        
        # Create bar plot
        y_pos = np.arange(len(top_categories))
        counts = list(top_categories.values())
        labels = [name[:20] + '...' if len(name) > 20 else name 
                  for name in top_categories.keys()]
        
        ax.barh(y_pos, counts, color=colors[idx], alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Product Count', fontsize=11)
        ax.set_title(f'{method.capitalize()} Method\nTop 8 Categories', 
                     fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        
        # Add count labels
        for i, count in enumerate(counts):
            ax.text(count + 0.1, i, str(count), va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_method_comparison_per_product(results, output_file='method_comparison.png'):
    """Compare all methods for each product"""
    
    # Get product names
    products = [r['product'] for r in results['keyword'][:10]]
    
    # Extract confidences
    keyword_conf = [r['top_category']['confidence'] if r.get('top_category') else 0 
                    for r in results['keyword'][:10]]
    embedding_conf = [r['top_category']['confidence'] if r.get('top_category') else 0 
                      for r in results['embedding'][:10]]
    hybrid_conf = [r['top_category']['confidence'] if r.get('top_category') else 0 
                   for r in results['hybrid'][:10]]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(products))
    width = 0.25
    
    ax.bar(x - width, keyword_conf, width, label='Keyword', color='#3b82f6', alpha=0.8)
    ax.bar(x, embedding_conf, width, label='Embedding', color='#10b981', alpha=0.8)
    ax.bar(x + width, hybrid_conf, width, label='Hybrid', color='#f59e0b', alpha=0.8)
    
    ax.set_xlabel('Products', fontsize=12, fontweight='bold')
    ax.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
    ax.set_title('Confidence Comparison per Product (First 10 Products)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p[:15] + '...' for p in products], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_performance_metrics(results, output_file='performance_metrics.png'):
    """Plot comprehensive performance metrics"""
    
    methods = ['keyword', 'embedding', 'hybrid']
    
    # Calculate metrics
    metrics = {}
    for method in methods:
        confidences = [
            r['top_category']['confidence']
            for r in results[method]
            if r.get('top_category')
        ]
        
        metrics[method] = {
            'Coverage': len([c for c in confidences]) / len(results[method]) * 100,
            'Avg Confidence': np.mean(confidences) * 100,
            'Consistency': (1 - np.std(confidences)) * 100,
            'Reliability': np.median(confidences) * 100
        }
    
    # Create radar chart
    categories = list(metrics['keyword'].keys())
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#3b82f6', '#10b981', '#f59e0b']
    
    for idx, method in enumerate(methods):
        values = list(metrics[method].values())
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=method.capitalize(), 
                color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 100)
    ax.set_title('Performance Metrics Comparison', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def generate_summary_table(results):
    """Generate summary statistics table"""
    
    methods = ['keyword', 'embedding', 'hybrid']
    
    print("\n" + "=" * 70)
    print("📊 SUMMARY STATISTICS")
    print("=" * 70)
    
    print(f"\n{'Method':<12} {'Coverage':<10} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 70)
    
    for method in methods:
        confidences = [
            r['top_category']['confidence']
            for r in results[method]
            if r.get('top_category')
        ]
        
        coverage = len(confidences) / len(results[method]) * 100
        
        print(f"{method.capitalize():<12} "
              f"{coverage:>8.1f}% "
              f"{np.mean(confidences):>9.4f} "
              f"{np.median(confidences):>9.4f} "
              f"{np.std(confidences):>9.4f} "
              f"{np.min(confidences):>9.4f} "
              f"{np.max(confidences):>9.4f}")
    
    print("=" * 70)


def main():
    """Main visualization function"""
    
    print("=" * 70)
    print("📊 CATEGORY CLASSIFICATION RESULTS VISUALIZATION")
    print("=" * 70)
    
    # Load results
    print("\n📁 Loading evaluation results...")
    results = load_latest_results()
    
    if not results:
        print("❌ No results found")
        return
    
    print(f"✅ Loaded results for {len(results['keyword'])} products")
    
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}/")
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    
    plot_confidence_distribution(results, 
                                  output_dir / "1_confidence_distribution.png")
    
    plot_confidence_comparison(results, 
                               output_dir / "2_confidence_comparison.png")
    
    plot_category_distribution(results, 
                               output_dir / "3_category_distribution.png")
    
    plot_method_comparison_per_product(results, 
                                       output_dir / "4_method_comparison.png")
    
    plot_performance_metrics(results, 
                            output_dir / "5_performance_metrics.png")
    
    # Print summary
    generate_summary_table(results)
    
    print(f"\n✅ All visualizations saved to: {output_dir}/")
    print("\n📊 Generated files:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"   - {file.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

"""
Visualize comparison between heuristic and embedding-based scorers
"""

import json
import matplotlib.pyplot as plt
import numpy as np


def load_scores(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return {item['id']: item for item in json.load(f)}


def main():
    print("Creating comparison visualizations...")
    
    # Load both results
    heuristic = load_scores('data/final_scores_heuristic.json')
    embedding = load_scores('data/final_scores_embedding.json')
    
    # Extract scores for matching functions
    h_scores = []
    e_scores = []
    labels = []
    
    for func_id in sorted(heuristic.keys()):
        h_scores.append(heuristic[func_id]['final_score'])
        e_scores.append(embedding[func_id]['final_score'])
        labels.append(heuristic[func_id]['label'])
    
    h_scores = np.array(h_scores)
    e_scores = np.array(e_scores)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Scatter plot of scores
    axes[0].scatter(h_scores, e_scores, alpha=0.5, s=30)
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect agreement')
    axes[0].set_xlabel('Heuristic Score', fontsize=12)
    axes[0].set_ylabel('Embedding Score', fontsize=12)
    axes[0].set_title('Score Correlation', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(h_scores, e_scores)[0, 1]
    axes[0].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=axes[0].transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Distribution comparison
    bins = np.linspace(0, 1, 30)
    axes[1].hist(h_scores, bins=bins, alpha=0.6, label='Heuristic', color='blue', edgecolor='black')
    axes[1].hist(e_scores, bins=bins, alpha=0.6, label='Embedding', color='orange', edgecolor='black')
    axes[1].axvline(0.45, color='red', linestyle='--', linewidth=1, alpha=0.7, label='UTILITY/MIXED')
    axes[1].axvline(0.60, color='green', linestyle='--', linewidth=1, alpha=0.7, label='MIXED/CORE')
    axes[1].set_xlabel('Score', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Score Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Classification comparison (stacked bar)
    h_counts = {'CORE': 0, 'MIXED': 0, 'UTILITY': 0}
    e_counts = {'CORE': 0, 'MIXED': 0, 'UTILITY': 0}
    
    for item in heuristic.values():
        h_counts[item['classification']] += 1
    for item in embedding.values():
        e_counts[item['classification']] += 1
    
    categories = ['CORE', 'MIXED', 'UTILITY']
    h_values = [h_counts[c] for c in categories]
    e_values = [e_counts[c] for c in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = axes[2].bar(x - width/2, h_values, width, label='Heuristic', color='skyblue', edgecolor='black')
    bars2 = axes[2].bar(x + width/2, e_values, width, label='Embedding', color='lightcoral', edgecolor='black')
    
    axes[2].set_xlabel('Classification', fontsize=12)
    axes[2].set_ylabel('Count', fontsize=12)
    axes[2].set_title('Classification Comparison', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(categories)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path = 'data/scorer_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Print statistics
    print("\nStatistics:")
    print(f"  Correlation: {corr:.3f}")
    print(f"  Mean absolute difference: {np.mean(np.abs(h_scores - e_scores)):.3f}")
    print(f"  Max difference: {np.max(np.abs(h_scores - e_scores)):.3f}")
    print(f"  Agreement rate: {np.mean(np.sign(h_scores - 0.5) == np.sign(e_scores - 0.5))*100:.1f}%")


if __name__ == '__main__':
    main()


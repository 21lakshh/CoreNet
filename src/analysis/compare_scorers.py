"""
Compare heuristic scorer vs embedding-based scorer
"""

import json
from collections import Counter


def load_scores(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return {item['id']: item for item in json.load(f)}


def main():
    print("="*70)
    print("SCORER COMPARISON: Heuristic vs Embedding-Based")
    print("="*70)
    
    # Load both results
    heuristic = load_scores('data/final_scores_heuristic.json')
    embedding = load_scores('data/final_scores_embedding.json')
    
    # Basic stats
    print("\n1. CLASSIFICATION DISTRIBUTION")
    print("-" * 70)
    
    h_counts = Counter(item['classification'] for item in heuristic.values())
    e_counts = Counter(item['classification'] for item in embedding.values())
    
    print(f"\n{'Classification':<15} {'Heuristic':<15} {'Embedding':<15} {'Difference'}")
    print("-" * 70)
    for cls in ['CORE', 'MIXED', 'UTILITY']:
        h_pct = h_counts[cls] / len(heuristic) * 100
        e_pct = e_counts[cls] / len(embedding) * 100
        diff = e_counts[cls] - h_counts[cls]
        print(f"{cls:<15} {h_counts[cls]:3d} ({h_pct:5.1f}%)   {e_counts[cls]:3d} ({e_pct:5.1f}%)   {diff:+3d}")
    
    # Agreement analysis
    print("\n\n2. AGREEMENT ANALYSIS")
    print("-" * 70)
    
    agreements = 0
    disagreements = []
    
    for func_id in heuristic:
        h_class = heuristic[func_id]['classification']
        e_class = embedding[func_id]['classification']
        
        if h_class == e_class:
            agreements += 1
        else:
            disagreements.append({
                'id': func_id,
                'label': heuristic[func_id]['label'],
                'heuristic': h_class,
                'embedding': e_class,
                'h_score': heuristic[func_id]['final_score'],
                'e_score': embedding[func_id]['final_score'],
                'cluster': embedding[func_id]['breakdown']['cluster']
            })
    
    agreement_rate = agreements / len(heuristic) * 100
    print(f"\nTotal functions: {len(heuristic)}")
    print(f"Agreements: {agreements} ({agreement_rate:.1f}%)")
    print(f"Disagreements: {len(disagreements)} ({100-agreement_rate:.1f}%)")
    
    # Breakdown by transition type
    print("\n\nDisagreement breakdown:")
    transitions = Counter((d['heuristic'], d['embedding']) for d in disagreements)
    for (h_cls, e_cls), count in transitions.most_common():
        print(f"  {h_cls} -> {e_cls}: {count}")
    
    # Interesting disagreements
    print("\n\n3. NOTABLE DISAGREEMENTS")
    print("-" * 70)
    
    # Heuristic says CORE, Embedding says UTILITY/MIXED
    print("\nHeuristic=CORE, Embedding=UTILITY/MIXED (potential false positives in heuristic):")
    h_core_e_other = [d for d in disagreements if d['heuristic'] == 'CORE' and d['embedding'] in ['UTILITY', 'MIXED']]
    for d in h_core_e_other[:10]:
        print(f"  {d['label']:30s} H:{d['h_score']:.3f} E:{d['e_score']:.3f} cluster={d['cluster']}")
    
    # Heuristic says UTILITY, Embedding says CORE
    print("\nHeuristic=UTILITY, Embedding=CORE (potential missed important functions in heuristic):")
    h_util_e_core = [d for d in disagreements if d['heuristic'] == 'UTILITY' and d['embedding'] == 'CORE']
    for d in h_util_e_core[:10]:
        print(f"  {d['label']:30s} H:{d['h_score']:.3f} E:{d['e_score']:.3f} cluster={d['cluster']}")
    
    # Score differences
    print("\n\n4. LARGEST SCORE DIFFERENCES")
    print("-" * 70)
    
    score_diffs = []
    for func_id in heuristic:
        h_score = heuristic[func_id]['final_score']
        e_score = embedding[func_id]['final_score']
        diff = abs(h_score - e_score)
        score_diffs.append({
            'label': heuristic[func_id]['label'],
            'diff': diff,
            'h_score': h_score,
            'e_score': e_score,
            'h_class': heuristic[func_id]['classification'],
            'e_class': embedding[func_id]['classification'],
        })
    
    score_diffs.sort(key=lambda x: x['diff'], reverse=True)
    
    print("\nTop 10 functions with biggest score differences:")
    for i, d in enumerate(score_diffs[:10], 1):
        print(f"{i:2d}. {d['label']:30s}")
        print(f"    Heuristic: {d['h_score']:.3f} ({d['h_class']})")
        print(f"    Embedding: {d['e_score']:.3f} ({d['e_class']})")
        print(f"    Difference: {d['diff']:.3f}")
    
    # Mixed category analysis
    print("\n\n5. MIXED CATEGORY EFFICIENCY")
    print("-" * 70)
    print(f"\nHeuristic MIXED: {h_counts['MIXED']} ({h_counts['MIXED']/len(heuristic)*100:.1f}%)")
    print(f"Embedding MIXED: {e_counts['MIXED']} ({e_counts['MIXED']/len(embedding)*100:.1f}%)")
    
    if e_counts['MIXED'] < h_counts['MIXED']:
        reduction = (h_counts['MIXED'] - e_counts['MIXED']) / h_counts['MIXED'] * 100
        print(f"\nEmbedding approach reduced MIXED by {reduction:.1f}%")
    elif e_counts['MIXED'] > h_counts['MIXED']:
        increase = (e_counts['MIXED'] - h_counts['MIXED']) / h_counts['MIXED'] * 100
        print(f"\nEmbedding approach increased MIXED by {increase:.1f}%")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print("\nKey insights:")
    print("  - Embedding approach uses semantic understanding + structural metrics")
    print("  - Heuristic approach uses hand-crafted rules + complexity")
    print("  - High agreement suggests both approaches capture similar patterns")
    print("  - Disagreements highlight edge cases worth manual review")


if __name__ == '__main__':
    main()


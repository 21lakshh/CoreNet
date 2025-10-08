"""
Embedding-based scorer using cluster assignments and augmented features
Maps clusters to CORE/UTILITY classifications with fine-grained scoring
"""

import json
from typing import Dict, List, Any


def map_cluster_to_base_score(cluster_id: int, metrics: Dict[str, Any]) -> float:
    """
    Map cluster ID to base score using cluster characteristics
    
    Cluster analysis:
    - Cluster 0: Low CC (2.61), low fan-in (10.46), few decorators → Data models/Exceptions
    - Cluster 1: High CC (74.87), high fan-in (18.01), many decorators → Core logic
    - Cluster 2: Medium CC (5.04), medium fan-in (10.11), few decorators → Helpers
    """
    cc = metrics['cyclomatic_complexity']
    fan_in = metrics['fan_in']
    fan_out = metrics['fan_out']
    has_decorator = metrics['has_decorator']
    
    if cluster_id == 0:
        # Data models and exceptions - generally UTILITY
        # But check for important exceptions or security models
        base = 0.30
        
        # Boost for exceptions (might be important)
        if cc > 5:
            base += 0.05
        
        # Penalty for very simple models
        if cc <= 2 and fan_in < 5:
            base -= 0.10
            
    elif cluster_id == 1:
        # Core business logic - generally CORE
        base = 0.75
        
        # Strong boost for decorators (framework entry points)
        if has_decorator:
            base += 0.15
        
        # Boost for high complexity
        if cc > 50:
            base += 0.10
        elif cc > 20:
            base += 0.05
        
        # Boost for high connectivity
        if fan_in > 20:
            base += 0.05
            
    else:  # cluster_id == 2
        # Simple helpers - MIXED/UTILITY depending on usage
        base = 0.45
        
        # Check if it's a widely-used helper (high fan-in)
        if fan_in > 15:
            base -= 0.15  # Widely reused → utility
        
        # Check if it has moderate complexity (might be important)
        if cc > 10:
            base += 0.10
        
        # Boost for decorators
        if has_decorator:
            base += 0.15
    
    return base


def refine_score_with_metadata(
    base_score: float,
    func_label: str,
    func_type: str,
    decorators: List[str]
) -> float:
    """
    Refine score using function metadata (name, type, decorators)
    """
    score = base_score
    
    # Security-related functions (CRITICAL - must be CORE)
    if any(x in func_label for x in ['OAuth', 'APIKey', 'HTTP', 'Bearer', 'Security', 'Credentials']):
        score = max(score, 0.80)
    
    # HTTP verbs (CORE routing functions)
    http_verbs = ['get', 'post', 'put', 'patch', 'delete', 'options', 'head']
    if func_label in http_verbs:
        score += 0.20
    
    # Routing keywords
    if any(x in func_label.lower() for x in ['route', 'websocket', 'mount', 'include_router']):
        score += 0.15
    
    # API entry points
    if func_label.startswith(('add_api', 'api_route', 'on_event', 'middleware')):
        score += 0.15
    
    # Decorator analysis
    for dec in decorators:
        dec_lower = dec.lower()
        # HTTP route decorators
        if any(verb in dec_lower for verb in http_verbs):
            score += 0.20
        # Framework decorators
        if any(x in dec_lower for x in ['route', 'depends', 'middleware', 'event']):
            score += 0.10
        # Deprecation decorator (might be less important)
        if 'deprecat' in dec_lower:
            score -= 0.05
    
    # Utility patterns (reduce score)
    if any(func_label.startswith(x) for x in ['get_', 'set_', 'to_', 'from_', 'format_', 'parse_']):
        score -= 0.15
    
    if func_label.endswith(('_handler', '_encoder', '_decoder', '_parser', '_validator')):
        score -= 0.15
    
    # Exception classes (generally utility)
    if func_label.endswith(('Exception', 'Error')) or 'exception' in func_label.lower():
        score -= 0.15
    
    # Dunder methods (generally utility)
    if func_label.startswith('__') and func_label.endswith('__'):
        score -= 0.20
        # But __init__ might be important for core classes
        if func_label == '__init__':
            score += 0.10
    
    # Class vs Function/Method
    if func_type == 'Class':
        # Classes are structural - slightly reduce unless they're security
        if not any(x in func_label for x in ['OAuth', 'APIKey', 'HTTP', 'Security']):
            score -= 0.05
    
    return max(0.0, min(1.0, score))


def classify_by_score(score: float) -> str:
    """
    Classify function based on final score
    Using hardened thresholds to minimize MIXED
    """
    if score > 0.55:
        return 'CORE'
    elif score < 0.45:
        return 'UTILITY'
    else:
        return 'MIXED'


def compute_embedding_score(cluster_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute final score for a function using cluster + metadata
    """
    # Extract data
    func_id = cluster_data['id']
    func_label = cluster_data['label']
    func_type = cluster_data['type']
    cluster_id = cluster_data['cluster']
    metrics = cluster_data['metrics']
    
    # Compute base score from cluster
    base_score = map_cluster_to_base_score(cluster_id, metrics)
    
    # Refine with metadata
    final_score = refine_score_with_metadata(
        base_score,
        func_label,
        func_type,
        metrics['decorators']
    )
    
    # Classify
    classification = classify_by_score(final_score)
    
    return {
        'id': func_id,
        'label': func_label,
        'type': func_type,
        'final_score': round(final_score, 4),
        'classification': classification,
        'breakdown': {
            'cluster': cluster_id,
            'base_score': round(base_score, 4),
            'cyclomatic_complexity': metrics['cyclomatic_complexity'],
            'fan_in': metrics['fan_in'],
            'fan_out': metrics['fan_out'],
            'has_decorator': metrics['has_decorator'],
            'decorators': metrics['decorators'],
        }
    }


def main():
    print("="*70)
    print("EMBEDDING-BASED SCORING & CLASSIFICATION")
    print("="*70)
    
    # Load cluster results
    print("\n[1/3] Loading cluster assignments...")
    with open('data/embedding_clusters.json', 'r', encoding='utf-8') as f:
        cluster_data = json.load(f)
    print(f"  Loaded {len(cluster_data)} functions with cluster assignments")
    
    # Compute scores
    print("\n[2/3] Computing scores...")
    results = []
    for func in cluster_data:
        scored = compute_embedding_score(func)
        results.append(scored)
    
    # Save results
    print("\n[3/3] Saving results...")
    output_file = 'data/final_scores_embedding.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("CLASSIFICATION SUMMARY")
    print("="*70)
    
    core = [r for r in results if r['classification'] == 'CORE']
    mixed = [r for r in results if r['classification'] == 'MIXED']
    utility = [r for r in results if r['classification'] == 'UTILITY']
    
    print(f"\nTotal functions: {len(results)}")
    print(f"Score range: [{min(r['final_score'] for r in results):.3f}, {max(r['final_score'] for r in results):.3f}]")
    print(f"Mean score: {sum(r['final_score'] for r in results)/len(results):.3f}")
    
    print(f"\nClassification:")
    print(f"  CORE (>0.55):       {len(core):3d} ({len(core)/len(results)*100:5.1f}%)")
    print(f"  MIXED [0.45-0.55]:  {len(mixed):3d} ({len(mixed)/len(results)*100:5.1f}%)")
    print(f"  UTILITY (<0.45):    {len(utility):3d} ({len(utility)/len(results)*100:5.1f}%)")
    
    print(f"\nBy cluster:")
    for cluster_id in range(3):
        cluster_funcs = [r for r in results if r['breakdown']['cluster'] == cluster_id]
        cluster_core = sum(1 for r in cluster_funcs if r['classification'] == 'CORE')
        cluster_utility = sum(1 for r in cluster_funcs if r['classification'] == 'UTILITY')
        cluster_mixed = sum(1 for r in cluster_funcs if r['classification'] == 'MIXED')
        print(f"  Cluster {cluster_id}: CORE={cluster_core}, MIXED={cluster_mixed}, UTILITY={cluster_utility}")
    
    # Top CORE examples
    print("\n" + "="*70)
    print("TOP 10 CORE FUNCTIONS")
    print("="*70)
    core_sorted = sorted(core, key=lambda x: x['final_score'], reverse=True)
    for i, func in enumerate(core_sorted[:10], 1):
        decorators_str = ', '.join(func['breakdown']['decorators'][:2]) if func['breakdown']['decorators'] else 'none'
        print(f"{i:2d}. [{func['final_score']:.3f}] {func['label']:30s} (cluster {func['breakdown']['cluster']}, @{decorators_str})")
    
    # Top UTILITY examples
    print("\n" + "="*70)
    print("TOP 10 UTILITY FUNCTIONS (lowest scores)")
    print("="*70)
    utility_sorted = sorted(utility, key=lambda x: x['final_score'])
    for i, func in enumerate(utility_sorted[:10], 1):
        print(f"{i:2d}. [{func['final_score']:.3f}] {func['label']:30s} (cluster {func['breakdown']['cluster']}, fan-in={func['breakdown']['fan_in']})")
    
    # Security check
    print("\n" + "="*70)
    print("SECURITY FUNCTION VALIDATION")
    print("="*70)
    security_keywords = ['OAuth', 'APIKey', 'HTTP', 'Bearer', 'Security', 'Credentials']
    security_funcs = [r for r in results if any(kw in r['label'] for kw in security_keywords)]
    
    if security_funcs:
        print(f"\nFound {len(security_funcs)} security-related functions:")
        for func in security_funcs:
            status = "OK" if func['classification'] == 'CORE' else "WARNING"
            print(f"  [{status}] {func['label']:30s} -> {func['classification']} (score={func['final_score']:.3f})")
    else:
        print("\nNo security-related functions found")
    
    print("\n" + "="*70)
    print("SCORING COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()


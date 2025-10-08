import json
import re
from typing import Dict, Any


def score_by_heuristics(func: Dict[str, Any]) -> float:
    """
    Rule-based scoring using name patterns, file paths, and types
    Returns: score in [0, 1] where 0=utility, 1=core
    """
    score = 0.5  # neutral baseline
    label = func['label']
    node_type = func['type']
    filepath = func.get('filepath', '')
    filename = filepath.split('/')[-1] if filepath else ''
    
    # === SECURITY OVERRIDE (check first!) ===
    # Security/Auth are ALWAYS core, regardless of file location
    is_security = any(x in label for x in ['OAuth', 'APIKey', 'HTTP', 'Bearer', 'Security', 'Credentials'])
    
    # === FILE PATH SIGNALS ===
    # Strong utility indicators (but not if security-related)
    utility_files = ['models.py', 'params.py', 'utils.py', 'encoders.py', 
                     'exceptions.py', 'exception_handlers.py', 'docs.py', 
                     'responses.py']
    if any(f in filepath for f in utility_files) and not is_security:
        score -= 0.35
    
    # Strong core indicators
    core_files = ['applications.py', 'routing.py']
    if any(f in filepath for f in core_files):
        score += 0.35
    
    # Security/Auth files are CORE business logic
    security_files = ['oauth2.py', 'api_key.py', 'http.py', 'security/', 'base.py']
    if any(f in filepath for f in security_files):
        score += 0.30
    
    # === LABEL PATTERNS ===
    # HTTP verb methods are CORE business logic
    http_verbs = ['get', 'post', 'put', 'patch', 'delete', 'options', 'head', 'trace']
    if label in http_verbs:
        score += 0.4
    
    # Routing/WebSocket patterns are CORE
    if any(x in label.lower() for x in ['route', 'websocket', 'mount', 'include_router']):
        score += 0.3
    
    # API/Endpoint patterns are CORE
    if label.startswith(('add_api', 'api_route', 'on_event', 'middleware')):
        score += 0.25
    
    # Utility naming patterns
    if re.match(r'^(get|set|to|from|format|parse)_', label):
        score -= 0.3
    
    if label.endswith(('_handler', '_encoder', '_decoder', '_parser', '_validator')):
        score -= 0.3
    
    if label.startswith('generate_'):
        score -= 0.2
    
    # Exception/Error indicators
    if label.endswith(('Exception', 'Error')) or 'exception' in label.lower():
        score -= 0.2
    
    # Dunder methods (boilerplate, push down harder)
    if re.match(r'^__.*__$', label):
        score -= 0.25  # Increased from 0.1
        # __init__ in params/models are pure data structure constructors
        if label == '__init__' and filename in ['params.py', 'models.py']:
            score -= 0.15  # Extra penalty
    
    # Decorator pattern
    if label == 'decorator':
        score -= 0.2
    
    # === TYPE-BASED ADJUSTMENTS ===
    if node_type == 'Class':
        # Security/Auth classes are CORE (apply BEFORE file penalties)
        if is_security:
            score += 0.65  # Very strong boost to override file penalties + ensure above tight threshold
        # Exception/Error classes
        elif label.endswith(('Exception', 'Error')):
            score -= 0.2
        # Model/Schema/Param classes (data structures) - but not security
        elif filename in ['models.py', 'params.py']:
            score -= 0.3
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


def score_by_complexity(metrics: Dict[str, Any]) -> float:
    """
    Score based on code complexity metrics
    More complex code is more likely to be core logic
    Returns: score in [0, 1]
    """
    line_count = metrics['line_count']
    cyclomatic = metrics['cyclomatic_complexity']
    param_count = metrics['param_count']
    
    # Normalize each metric
    # Line count: 5-100 lines is typical, <10 is simple, >50 is complex
    line_score = min(line_count / 50, 1.0)
    
    # Cyclomatic complexity: 1-3 simple, 4-10 moderate, >10 complex
    complexity_score = min(cyclomatic / 10, 1.0)
    
    # Param count: 0-2 simple, 3-5 moderate, >5 complex
    param_score = min(param_count / 5, 1.0)
    
    # Combined complexity score
    complexity = (line_score * 0.4 + complexity_score * 0.4 + param_score * 0.2)
    
    return complexity


def compute_final_score(func: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine heuristic and complexity signals into final score
    Weights: 40% heuristic + 60% complexity (no embeddings)
    TIGHT THRESHOLDS: Narrower MIXED zone for more decisive classification
    SECURITY OVERRIDE: Security classes use heuristic-dominant weighting
    """
    heuristic = score_by_heuristics(func)
    complexity = score_by_complexity(func['static_metrics'])
    
    # Security override: prioritize heuristics over complexity for security classes
    is_security = any(x in func['label'] for x in ['OAuth', 'APIKey', 'HTTP', 'Bearer', 'Security', 'Credentials'])
    
    if is_security:
        # Security classes: 80% heuristic, 20% complexity (prevent short classes from being penalized)
        final = 0.8 * heuristic + 0.2 * complexity
    else:
        # Normal: 40% heuristic, 60% complexity
        final = 0.4 * heuristic + 0.6 * complexity
    
    # TIGHTER THRESHOLDS - narrower MIXED zone
    if final > 0.55:  # Was 0.60
        classification = 'CORE'
    elif final < 0.45:  # Was 0.40
        classification = 'UTILITY'
    else:
        classification = 'MIXED'
    
    return {
        'id': func['id'],
        'label': func['label'],
        'type': func['type'],
        'final_score': round(final, 4),
        'classification': classification,
        'breakdown': {
            'heuristic': round(heuristic, 4),
            'complexity': round(complexity, 4),
            'line_count': func['static_metrics']['line_count'],
            'param_count': func['static_metrics']['param_count'],
            'cyclomatic_complexity': func['static_metrics']['cyclomatic_complexity'],
            'has_docstring': func['static_metrics']['has_docstring']
        }
    }


def main():
    FUNCTIONS_FILE = 'data/extracted_functions.json'
    OUTPUT_FILE = 'data/final_scores_heuristic.json'
    
    # Load data
    print("Loading extracted functions...")
    with open(FUNCTIONS_FILE, 'r', encoding='utf-8') as f:
        functions = json.load(f)
    
    # Compute final scores (static analysis only)
    print(f"\nComputing final scores for {len(functions)} functions...")
    print("Using: 40% heuristics + 60% complexity (no embeddings)")
    print("TIGHT MODE: CORE > 0.55, UTILITY < 0.45, MIXED [0.45-0.55]")
    results = []
    
    for func in functions:
        result = compute_final_score(func)
        results.append(result)
    
    # Save
    print(f"Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Statistics
    print("\n" + "="*70)
    print("FINAL SCORING RESULTS (TIGHT THRESHOLDS)")
    print("="*70)
    
    scores = [r['final_score'] for r in results]
    print(f"Total functions: {len(results)}")
    print(f"Score range: [{min(scores):.3f}, {max(scores):.3f}]")
    print(f"Mean score: {sum(scores)/len(scores):.3f}")
    
    # Classification breakdown
    core = [r for r in results if r['classification'] == 'CORE']
    utility = [r for r in results if r['classification'] == 'UTILITY']
    mixed = [r for r in results if r['classification'] == 'MIXED']
    
    print(f"\nClassification:")
    print(f"  CORE (>0.55):       {len(core):3d} ({len(core)/len(results)*100:5.1f}%)")
    print(f"  MIXED [0.45-0.55]:  {len(mixed):3d} ({len(mixed)/len(results)*100:5.1f}%)")
    print(f"  UTILITY (<0.45):    {len(utility):3d} ({len(utility)/len(results)*100:5.1f}%)")
    
    # Comparison with standard thresholds
    standard_core = sum(1 for s in scores if s > 0.60)
    standard_mixed = sum(1 for s in scores if 0.40 <= s <= 0.60)
    standard_utility = sum(1 for s in scores if s < 0.40)
    
    print(f"\nComparison with standard thresholds (0.40-0.60):")
    print(f"  Standard CORE:    {standard_core:3d} ({standard_core/len(results)*100:5.1f}%)")
    print(f"  Standard MIXED:   {standard_mixed:3d} ({standard_mixed/len(results)*100:5.1f}%)")
    print(f"  Standard UTILITY: {standard_utility:3d} ({standard_utility/len(results)*100:5.1f}%)")
    
    print(f"\nMIXED reduction: {standard_mixed} -> {len(mixed)} ({(standard_mixed-len(mixed))/standard_mixed*100:.1f}% decrease)")
    
    # Top examples
    print("\n" + "="*70)
    print("TOP 10 CORE FUNCTIONS (highest scores)")
    print("="*70)
    top_core = sorted(results, key=lambda x: x['final_score'], reverse=True)[:10]
    for r in top_core:
        filepath = r['id'].split(':')[1].split('/')[-1]
        print(f"{r['final_score']:.3f} | {r['label']:30s} | {r['type']:8s} | {filepath}")
    
    print("\n" + "="*70)
    print("TOP 10 UTILITY FUNCTIONS (lowest scores)")
    print("="*70)
    top_utility = sorted(results, key=lambda x: x['final_score'])[:10]
    for r in top_utility:
        filepath = r['id'].split(':')[1].split('/')[-1]
        print(f"{r['final_score']:.3f} | {r['label']:30s} | {r['type']:8s} | {filepath}")
    
    # Show MIXED examples
    print("\n" + "="*70)
    print("MIXED ZONE EXAMPLES (0.45-0.55)")
    print("="*70)
    mixed_examples = sorted(mixed, key=lambda x: x['final_score'])[:10]
    for r in mixed_examples:
        filepath = r['id'].split(':')[1].split('/')[-1]
        print(f"{r['final_score']:.3f} | {r['label']:30s} | {r['type']:8s} | {filepath}")
    
    print("\nDone! Results saved to " + OUTPUT_FILE)


if __name__ == '__main__':
    main()


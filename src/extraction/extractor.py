import json
import re
from typing import Dict, List, Any


def parse_id(node_id: str) -> Dict[str, str]:
    parts = node_id.split(':')
    if len(parts) >= 3:
        return {'filepath': parts[1]}
    return {'filepath': ''}


def count_lines(code: str) -> int:
    lines = [line.strip() for line in code.split('\n')]
    return len([l for l in lines if l and not l.startswith('#')])


def count_params(code: str) -> int:
    match = re.search(r'def\s+\w+\s*\((.*?)\):', code, re.DOTALL)
    if not match:
        return 0
    
    params = match.group(1)
    if not params.strip():
        return 0
    
    depth = 0
    param_count = 1
    for char in params:
        if char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
        elif char == ',' and depth == 0:
            param_count += 1
    
    return param_count


def has_docstring(code: str) -> bool:
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if 'def ' in line or 'class ' in line:
            for j in range(i+1, min(i+5, len(lines))):
                if '"""' in lines[j] or "'''" in lines[j]:
                    return True
            break
    return False


def estimate_cyclomatic_complexity(code: str) -> int:
    keywords = [
        "if",
        "elif",
        "else if",
        "for",
        "while",
        "try",
        "except",
        "finally",
        "with",
        "and",
        "or",
        "assert",
        "match",
        "case",
        "lambda",  
        "return",  
        "yield",   
        "continue",
        "break",
        "comprehension_if",  
    ]
    complexity = 1  
    
    for keyword in keywords:
        complexity += code.count(keyword)
    
    return complexity


def compute_static_metrics(code: str) -> Dict[str, Any]:
    return {
        'line_count': count_lines(code),
        'param_count': count_params(code),
        'has_docstring': has_docstring(code),
        'cyclomatic_complexity': estimate_cyclomatic_complexity(code)
    }


def extract_functions(input_file: str, output_file: str):
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    nodes = data['analysisData']['graphNodes']
    print(f"Total nodes: {len(nodes)}")
    
    code_types = {'Method', 'Function', 'Class'}
    filtered = [n for n in nodes if n.get('type') in code_types]
    print(f"Filtered to code nodes: {len(filtered)}")
    
    extracted = []
    for node in filtered:
        filepath_data = parse_id(node['id'])
        code = node.get('code', '')
        
        entry = {
            'id': node['id'],
            'label': node['label'],
            'code': code,
            'type': node['type'],
            'filepath': filepath_data['filepath'],  
            'static_metrics': compute_static_metrics(code)
        }
        extracted.append(entry)
    
    print(f"\nSaving {len(extracted)} functions to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total extracted: {len(extracted)}")
    
    type_counts = {}
    for entry in extracted:
        t = entry['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"By type: {type_counts}")
    
    line_counts = [e['static_metrics']['line_count'] for e in extracted]
    param_counts = [e['static_metrics']['param_count'] for e in extracted]
    has_docs = sum(1 for e in extracted if e['static_metrics']['has_docstring'])
    
    print(f"\nStatic Metrics:")
    print(f"  Avg line count: {sum(line_counts)/len(line_counts):.1f}")
    print(f"  Avg param count: {sum(param_counts)/len(param_counts):.1f}")
    print(f"  With docstrings: {has_docs}/{len(extracted)} ({has_docs/len(extracted)*100:.1f}%)")
    print(f"\nDone!")


if __name__ == '__main__':
    extract_functions(
        input_file='data/analysis-with-code.json',
        output_file='data/extracted_functions.json'
    )


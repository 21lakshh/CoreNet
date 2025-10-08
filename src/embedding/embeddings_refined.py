"""
Refined embedding-based function classifier
Combines CodeBERT embeddings with structural metrics and clustering
"""

import json
import re
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Lazy imports for heavy dependencies
def get_model_and_tokenizer():
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model, torch


def extract_decorators(code: str) -> List[str]:
    """Extract decorator names from function code"""
    decorators = []
    lines = code.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('@'):
            # Extract decorator name (strip @ and arguments)
            decorator = line[1:].split('(')[0].strip()
            decorators.append(decorator)
    return decorators


def compute_decoration_features(decorators: List[str]) -> Dict[str, float]:
    """
    Compute decoration status features
    Returns dict with binary/count features
    """
    features = {
        'has_decorator': 1.0 if decorators else 0.0,
        'decorator_count': float(len(decorators)),
        'is_http_route': 0.0,
        'is_classmethod': 0.0,
        'is_staticmethod': 0.0,
        'is_property': 0.0,
        'is_framework_entry': 0.0,
    }
    
    for dec in decorators:
        dec_lower = dec.lower()
        # HTTP route decorators
        if any(verb in dec_lower for verb in ['get', 'post', 'put', 'patch', 'delete', 'route']):
            features['is_http_route'] = 1.0
            features['is_framework_entry'] = 1.0
        # Python built-ins
        if dec in ['classmethod', 'class_method']:
            features['is_classmethod'] = 1.0
        if dec in ['staticmethod', 'static_method']:
            features['is_staticmethod'] = 1.0
        if dec in ['property']:
            features['is_property'] = 1.0
        # Framework decorators
        if any(x in dec_lower for x in ['validator', 'middleware', 'event', 'depends']):
            features['is_framework_entry'] = 1.0
    
    return features


def build_call_graph(functions: List[Dict]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Build call graph and compute fan-in/fan-out for each function
    Returns: (fan_in_dict, fan_out_dict)
    """
    fan_in = {f['id']: 0 for f in functions}
    fan_out = {f['id']: 0 for f in functions}
    
    # Build name->id mapping for faster lookup
    label_to_ids = {}
    for f in functions:
        label = f['label']
        if label not in label_to_ids:
            label_to_ids[label] = []
        label_to_ids[label].append(f['id'])
    
    # Count function calls
    for func in functions:
        code = func['code']
        # Find function calls (simple regex approach)
        # Matches: function_name( or self.method( or obj.method(
        call_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        calls = re.findall(call_pattern, code)
        
        called_functions = set()
        for call_name in calls:
            # Skip control flow keywords and common built-ins
            if call_name in ['if', 'for', 'while', 'def', 'class', 'return', 
                            'print', 'len', 'str', 'int', 'float', 'dict', 
                            'list', 'set', 'tuple', 'range', 'enumerate']:
                continue
            
            if call_name in label_to_ids:
                called_functions.add(call_name)
        
        # Update fan-out for this function
        fan_out[func['id']] = len(called_functions)
        
        # Update fan-in for called functions
        for called_name in called_functions:
            for called_id in label_to_ids[called_name]:
                if called_id != func['id']:  # Don't count self-calls
                    fan_in[called_id] += 1
    
    return fan_in, fan_out


def generate_code_embedding(code: str, tokenizer, model, torch) -> np.ndarray:
    """Generate CodeBERT embedding for a code snippet"""
    # Truncate code if too long
    if len(code) > 2000:
        code = code[:2000]
    
    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Use mean pooling over sequence
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.squeeze().numpy()


def create_augmented_features(
    embedding: np.ndarray,
    cc: int,
    fan_in: int,
    fan_out: int,
    decoration_features: Dict[str, float]
) -> np.ndarray:
    """
    Combine embedding with structural metrics
    Returns augmented feature vector
    """
    # Normalize metrics to similar scale as embeddings
    cc_norm = min(cc / 20.0, 1.0)  # Cap at 20
    fan_in_norm = min(fan_in / 10.0, 1.0)  # Cap at 10
    fan_out_norm = min(fan_out / 10.0, 1.0)  # Cap at 10
    
    # Weight structural features more heavily by repeating them
    structural_features = np.array([
        cc_norm,
        cc_norm,  # Duplicate for emphasis
        fan_in_norm,
        fan_in_norm,
        fan_in_norm,  # Triple fan-in (very important for utility detection)
        fan_out_norm,
        fan_out_norm,
        decoration_features['has_decorator'],
        decoration_features['is_http_route'],
        decoration_features['is_http_route'],  # Emphasize route decorators
        decoration_features['is_framework_entry'],
        decoration_features['is_framework_entry'],
        decoration_features['is_classmethod'],
        decoration_features['is_staticmethod'],
        decoration_features['is_property'],
    ])
    
    # Concatenate embedding with structural features
    return np.concatenate([embedding, structural_features])


def apply_clustering(features: np.ndarray, n_clusters: int = 3) -> Tuple[np.ndarray, Any]:
    """
    Apply k-means clustering to augmented features
    Returns: (cluster_labels, kmeans_model)
    """
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(features_scaled)
    
    return labels, kmeans


def visualize_clusters(
    features: np.ndarray,
    labels: np.ndarray,
    function_data: List[Dict],
    output_path: str = 'data/cluster_visualization.png'
):
    """
    Visualize clusters using PCA dimensionality reduction
    """
    # Reduce to 2D using PCA
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(features)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Colored by cluster
    scatter1 = axes[0].scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=labels,
        cmap='viridis',
        alpha=0.6,
        s=50
    )
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[0].set_title('Functions Clustered by Augmented Embeddings')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    
    # Plot 2: Colored by cyclomatic complexity
    cc_values = [f['static_metrics']['cyclomatic_complexity'] for f in function_data]
    scatter2 = axes[1].scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=cc_values,
        cmap='plasma',
        alpha=0.6,
        s=50
    )
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[1].set_title('Functions Colored by Cyclomatic Complexity')
    plt.colorbar(scatter2, ax=axes[1], label='CC')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    return features_2d


def analyze_clusters(
    cluster_labels: np.ndarray,
    function_data: List[Dict],
    fan_in_dict: Dict[str, int],
    fan_out_dict: Dict[str, int]
):
    """Print detailed cluster analysis"""
    n_clusters = len(set(cluster_labels))
    
    print("\n" + "="*70)
    print("CLUSTER ANALYSIS")
    print("="*70)
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_funcs = [f for i, f in enumerate(function_data) if cluster_mask[i]]
        
        print(f"\n{'='*70}")
        print(f"CLUSTER {cluster_id}: {len(cluster_funcs)} functions ({len(cluster_funcs)/len(function_data)*100:.1f}%)")
        print(f"{'='*70}")
        
        # Compute cluster statistics
        cc_values = [f['static_metrics']['cyclomatic_complexity'] for f in cluster_funcs]
        fan_in_values = [fan_in_dict[f['id']] for f in cluster_funcs]
        fan_out_values = [fan_out_dict[f['id']] for f in cluster_funcs]
        
        has_decorators = sum(1 for f in cluster_funcs if '@' in f['code'])
        
        print(f"\nMetrics:")
        print(f"  Avg Cyclomatic Complexity: {np.mean(cc_values):.2f}")
        print(f"  Avg Fan-in:  {np.mean(fan_in_values):.2f} (how many call this)")
        print(f"  Avg Fan-out: {np.mean(fan_out_values):.2f} (how many this calls)")
        print(f"  With decorators: {has_decorators} ({has_decorators/len(cluster_funcs)*100:.1f}%)")
        
        # Common file paths
        filepaths = [f['id'].split(':')[1].split('/')[-1] for f in cluster_funcs]
        from collections import Counter
        top_files = Counter(filepaths).most_common(5)
        print(f"\nTop files:")
        for filename, count in top_files:
            print(f"  {filename}: {count}")
        
        # Sample functions
        print(f"\nSample functions:")
        for func in cluster_funcs[:5]:
            decorators = extract_decorators(func['code'])
            dec_str = f"@{decorators[0]}" if decorators else "no decorator"
            print(f"  - {func['label']:25s} ({dec_str:20s}) CC={func['static_metrics']['cyclomatic_complexity']}")


def main():
    print("="*70)
    print("REFINED EMBEDDING-BASED FUNCTION CLASSIFIER")
    print("="*70)
    
    # Load extracted functions
    print("\n[1/7] Loading extracted functions...")
    with open('data/extracted_functions.json', 'r', encoding='utf-8') as f:
        functions = json.load(f)
    print(f"  Loaded {len(functions)} functions")
    
    # Build call graph
    print("\n[2/7] Building call graph (fan-in/fan-out)...")
    fan_in_dict, fan_out_dict = build_call_graph(functions)
    avg_fan_in = np.mean(list(fan_in_dict.values()))
    avg_fan_out = np.mean(list(fan_out_dict.values()))
    print(f"  Avg fan-in: {avg_fan_in:.2f}, Avg fan-out: {avg_fan_out:.2f}")
    
    # Extract decorators
    print("\n[3/7] Extracting decorators...")
    for func in functions:
        func['decorators'] = extract_decorators(func['code'])
        func['decoration_features'] = compute_decoration_features(func['decorators'])
    decorated_count = sum(1 for f in functions if f['decorators'])
    print(f"  Found {decorated_count} functions with decorators")
    
    # Load model
    print("\n[4/7] Loading CodeBERT model...")
    tokenizer, model, torch = get_model_and_tokenizer()
    print("  Model loaded")
    
    # Generate embeddings and augmented features
    print("\n[5/7] Generating augmented embeddings...")
    augmented_features = []
    for i, func in enumerate(functions):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(functions)}")
        
        # Generate base embedding
        embedding = generate_code_embedding(func['code'], tokenizer, model, torch)
        
        # Augment with structural metrics
        aug_features = create_augmented_features(
            embedding,
            func['static_metrics']['cyclomatic_complexity'],
            fan_in_dict[func['id']],
            fan_out_dict[func['id']],
            func['decoration_features']
        )
        augmented_features.append(aug_features)
    
    augmented_features = np.array(augmented_features)
    print(f"  Generated augmented features: shape {augmented_features.shape}")
    print(f"    - Embedding dimensions: 768")
    print(f"    - Structural features: {augmented_features.shape[1] - 768}")
    
    # Apply clustering
    print("\n[6/7] Applying k-means clustering (k=3)...")
    cluster_labels, kmeans = apply_clustering(augmented_features, n_clusters=3)
    print(f"  Clustering complete")
    
    # Analyze clusters
    analyze_clusters(cluster_labels, functions, fan_in_dict, fan_out_dict)
    
    # Visualize
    print("\n[7/7] Generating visualizations...")
    features_2d = visualize_clusters(augmented_features, cluster_labels, functions)
    
    # Save results
    print("\nSaving results...")
    results = []
    for i, func in enumerate(functions):
        results.append({
            'id': func['id'],
            'label': func['label'],
            'type': func['type'],
            'cluster': int(cluster_labels[i]),
            'metrics': {
                'cyclomatic_complexity': func['static_metrics']['cyclomatic_complexity'],
                'fan_in': fan_in_dict[func['id']],
                'fan_out': fan_out_dict[func['id']],
                'has_decorator': bool(func['decorators']),
                'decorators': func['decorators'],
            },
            'pca_coords': [float(features_2d[i, 0]), float(features_2d[i, 1])]
        })
    
    output_file = 'data/embedding_clusters.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {output_file}")
    
    print("\n" + "="*70)
    print("CLUSTERING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review cluster_visualization.png to see function groupings")
    print("  2. Examine embedding_clusters.json for cluster assignments")
    print("  3. Interpret clusters:")
    print("     - High fan-in + low complexity -> utility functions")
    print("     - High CC + decorators -> core business logic")
    print("     - Low fan-in/out -> isolated helpers")


if __name__ == '__main__':
    main()


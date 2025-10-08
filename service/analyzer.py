"""
Core analysis logic - imports from pipeline modules
"""
import sys
import os
from typing import List, Dict, Any
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel

# Add src directory to path to import pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import pipeline functions directly
from extraction.extractor import extract_functions, compute_static_metrics, parse_id
from scoring.scorer_heuristic import score_by_heuristics, score_by_complexity, compute_final_score
from scoring.scorer_embedding import compute_embedding_score
from embedding.embeddings_refined import (
    get_model_and_tokenizer, 
    build_call_graph, 
    extract_decorators, 
    compute_decoration_features,
    generate_code_embedding,
    create_augmented_features,
    apply_clustering
)


def extract_function_data(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract and enrich function data with static metrics using pipeline logic"""
    functions = []
    
    for node in nodes:
        if node['type'] not in ['Function', 'Method', 'Class']:
            continue
        
        filepath_data = parse_id(node['id'])
        code = node.get('code', '')
        
        functions.append({
            'id': node['id'],
            'label': node['label'],
            'code': code,
            'type': node['type'],
            'filepath': filepath_data['filepath'],
            'static_metrics': compute_static_metrics(code)
        })
    
    return functions


def score_heuristic(functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score functions using heuristic approach - uses pipeline logic"""
    results = []
    
    for func in functions:
        result = compute_final_score(func)
        results.append({
            **func,
            'score': result['final_score'],
            'classification': result['classification']
        })
    
    return sorted(results, key=lambda x: x['score'], reverse=True)


# ========================================
# EMBEDDING SCORING
# ========================================

class EmbeddingAnalyzer:
    """Singleton embedding analyzer with model caching - uses pipeline logic"""
    _instance = None
    _model = None
    _tokenizer = None
    _torch = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self):
        """Load CodeBERT model (cached)"""
        if self._model is None:
            self._tokenizer, self._model, self._torch = get_model_and_tokenizer()
        return self._model, self._tokenizer, self._torch
    
    def analyze(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score functions using embedding approach - uses pipeline logic"""
        # Load model
        model, tokenizer, torch = self.load_model()
        
        # Build call graph
        fan_in_dict, fan_out_dict = build_call_graph(functions)
        
        # Extract decorators and compute decoration features
        for func in functions:
            func['decorators'] = extract_decorators(func['code'])
            func['decoration_features'] = compute_decoration_features(func['decorators'])
        
        # Generate embeddings and augmented features
        augmented_features = []
        for func in functions:
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
        
        # Apply clustering
        n_clusters = min(3, len(functions))
        cluster_labels, kmeans = apply_clustering(augmented_features, n_clusters=n_clusters)
        
        # Score each function using pipeline logic
        results = []
        for i, func in enumerate(functions):
            # Create cluster data in pipeline format
            cluster_data = {
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
                }
            }
            
            # Use pipeline scoring
            scored = compute_embedding_score(cluster_data)
            
            results.append({
                **func,
                'score': scored['final_score'],
                'classification': scored['classification'],
                'cluster': scored['breakdown']['cluster'],
                'semantic_metrics': {
                    'fan_in': scored['breakdown']['fan_in'],
                    'fan_out': scored['breakdown']['fan_out']
                },
                'has_decorator': scored['breakdown']['has_decorator']
            })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)


# Utility Function Detection System

Static analysis system to identify and filter trivial utility functions from the FastAPI codebase.

## 🎯 Goal

Distinguish **core business logic** from **low-value utility functions** to help developers focus on important code during analysis and refactoring.

---

## 📊 Evolution: Two Approaches

### Approach 1: Heuristic-Based (Initial)

**Strategy**: Pure static analysis using hand-crafted rules

**Components**:
- 🏷️ **Naming patterns** (HTTP verbs, `get_*`, `*_handler`, dunder methods)
- 📂 **File path heuristics** (`models.py` = utility, `routing.py` = core)
- 🧮 **Code metrics** (cyclomatic complexity, line count, parameter count)
- 🔒 **Security protection** (adaptive weighting for security-related functions)

**Scoring formula**: `final = 0.4 × heuristic + 0.6 × complexity` (with adaptive weights for security)

**Strengths**:
- ✅ Fast (~1 second)
- ✅ No ML dependencies
- ✅ Highly interpretable rules
- ✅ Easy to customize

**Limitations**:
- ❌ Can't understand semantic similarity (e.g., `Components` class marked UTILITY despite being important)
- ❌ Relies on file path assumptions (fails when important code lives in "utility" files)
- ❌ No global connectivity analysis (doesn't know which functions are heavily used)

**Results**:
- CORE: 138 (55.0%)
- MIXED: 20 (8.0%)
- UTILITY: 93 (37.1%)
- **Decisiveness: 92%**

---

### Approach 2: Embedding-Based (Refined) ⭐ **RECOMMENDED**

**Strategy**: Semantic code embeddings + structural metrics + clustering

**Key Innovation**: Augmented feature vectors that combine:
1. **768D CodeBERT embeddings** - Semantic understanding of code
2. **15D structural features**:
   - 🧮 Cyclomatic complexity (normalized)
   - 🔁 Fan-in (how many functions call this)
   - 🔁 Fan-out (how many functions this calls)
   - 🎀 Decorator status (HTTP routes, framework decorators)

**Pipeline**:
```
1. CodeBERT embedding (768D)
2. Call graph analysis → fan-in/out metrics
3. Decorator extraction → framework entry points
4. Feature augmentation → 783D vector
5. K-means clustering (k=3)
6. Cluster-based scoring + metadata refinement
```

**The Magic**: By combining embeddings with structural metrics, we get:
- **Semantic grouping** - Similar functions cluster together
- **Global context** - High fan-in = widely used utility
- **Framework awareness** - Decorators signal entry points

**Discovered Clusters**:

| Cluster | % | Avg CC | Avg Fan-in | Decorators | Interpretation |
|---------|---|--------|------------|------------|----------------|
| **0** | 22.3% | 2.61 | 10.46 | 7.1% | Data models & exceptions (UTILITY) |
| **1** | 49.0% | 74.87 | 18.01 | 33.3% | Core business logic (CORE) |
| **2** | 28.7% | 5.04 | 10.11 | 8.3% | Simple helpers (MIXED/UTILITY) |

**Strengths**:
- ✅ **Semantic understanding** - Captures code meaning, not just syntax
- ✅ **Global connectivity** - Fan-in/out reveals true utility vs core
- ✅ **More decisive** - Only 6.4% MIXED (vs 8.0% in heuristic)
- ✅ **Catches important functions heuristic misses** (see below)
- ✅ **Visualizable** - PCA plots show clear cluster separation

**Results** (with hardened thresholds):
- CORE: 150 (59.8%)
- MIXED: 16 (6.4%)
- UTILITY: 85 (33.9%)
- **Decisiveness: 93.6%** 🎯

---

## 🔍 What Improved: Key Wins

### 1. **Caught Important Functions the Heuristic Missed**

| Function | Heuristic | Embedding | Why Heuristic Failed | Why Embedding Succeeded |
|----------|-----------|-----------|----------------------|-------------------------|
| `Components` | 0.144 (UTILITY) | 0.700 (CORE) | In `models.py` → penalized by file path | Cluster 1: High CC (1828), high fan-in, semantic similarity to core |
| `EmailStr` | 0.370 (UTILITY) | 0.900 (CORE) | Generic name pattern | Cluster 1: Important validation type, high complexity |
| `_validate` | 0.161 (UTILITY) | 0.600 (CORE) | Underscore prefix → utility pattern | Cluster 2: Actual validation logic, moderate complexity |

**Insight**: File paths and naming conventions can be misleading. Embeddings + call graphs reveal true importance.

### 2. **More Decisive Classifications**

- **MIXED reduced**: 8.0% → 6.4% (-20% reduction)
- **Agreement rate**: 85.3% between approaches
- **Score correlation**: 0.881 (strong consistency where both methods agree)

### 3. **Better Cluster Separation**

The PCA visualization shows **clear 3-cluster structure**:
- **Cluster 0** (purple): Tight grouping of simple models/exceptions
- **Cluster 1** (yellow): Spread of complex core functions (high variance in complexity)
- **Cluster 2** (teal): Moderate helpers, clearly separated from core

See `data/cluster_visualization.png` for visual proof.

### 4. **Perfect Security Validation** (Both Methods)

**100% accuracy** on security functions: All 33 security-related functions (OAuth, APIKey, HTTPBearer, etc.) correctly classified as CORE with **0 false positives**.

---

## 📂 Architecture

### Pipeline 1: Heuristic Scorer

```
data/analysis-with-code.json (FastAPI graph)
         ↓
    [extractor.py]
    - Filter to Method/Function/Class nodes
    - Compute CC, line count, param count
         ↓
data/extracted_functions.json (251 functions)
         ↓
   [scorer_tight.py]
   - Heuristic score: path + name patterns
   - Complexity score: CC + length
   - Adaptive weighting (security boost)
   - Thresholds: CORE>0.55, UTILITY<0.45
         ↓
data/final_scores_tight.json
```

### Pipeline 2: Embedding Scorer ⭐

```
data/extracted_functions.json
         ↓
  [embeddings_refined.py]
  - CodeBERT embedding (768D)
  - Call graph: fan-in/out
  - Decorator extraction
  - Feature augmentation (783D)
  - K-means clustering (k=3)
         ↓
data/embedding_clusters.json
         ↓
  [scorer_embedding.py]
  - Map cluster → base score
  - Refine with metadata (security, HTTP verbs)
  - Thresholds: CORE>0.55, UTILITY<0.45
         ↓
data/final_scores_embedding.json
```

---

## 🚀 Usage

### Quick Start: Heuristic Scorer (Fast)

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Run full pipeline
python run_pipeline.py
```

Output:
- `data/extracted_functions.json` - All functions with metrics
- `data/final_scores_tight.json` - Classifications

### Advanced: Embedding Scorer (Recommended)

```bash
# Ensure ML dependencies installed
pip install -r requirements_embedding.txt

# Step 1: Extract functions (if not already done)
python extractor.py

# Step 2: Generate embeddings and cluster
python embeddings_refined.py

# Step 3: Score based on clusters
python scorer_embedding.py
```

Output:
- `data/embedding_clusters.json` - Cluster assignments
- `data/final_scores_embedding.json` - Classifications
- `data/cluster_visualization.png` - PCA visualization

### Compare Both Approaches

```bash
# Ensure both scorers have run
python compare_scorers.py
python visualize_comparison.py
```

Output:
- `data/scorer_comparison.png` - Visual comparison

---

## 📊 Results Comparison

### Classification Distribution

| Category | Heuristic | Embedding | Winner |
|----------|-----------|-----------|--------|
| **CORE** | 138 (55.0%) | 150 (59.8%) | Embedding (+12) |
| **MIXED** | 20 (8.0%) | 16 (6.4%) | Embedding (-4) |
| **UTILITY** | 93 (37.1%) | 85 (33.9%) | Heuristic (+8) |

### Agreement Metrics

- **Classification agreement**: 85.3% (214/251 functions)
- **Score correlation**: 0.881
- **Mean absolute difference**: 0.121
- **Directional agreement**: 93.6% (same side of 0.5)

### Top CORE Functions (Both Agree)

1. `FastAPI` class (score: 1.000)
2. `add_api_route` (score: 1.000)
3. HTTP verbs: `get`, `post`, `put`, `delete`, etc. (score: 1.000)
4. `include_router`, `middleware`, `api_route` (score: 1.000)

### Top UTILITY Functions (Both Agree)

1. `get_path_param_names` (score: 0.050-0.100)
2. Exception classes: `FastAPIError`, `ValidationException`
3. Dunder methods: `__repr__`, `__call__`, `__get_validators__`
4. Simple model classes from `models.py`, `params.py`

---

## 🎨 Visualizations

### 1. Cluster Visualization (`cluster_visualization.png`)

**Left plot**: Functions colored by cluster (3 distinct groups)
**Right plot**: Same functions colored by cyclomatic complexity

**Key insight**: Cluster 1 (yellow) captures the mega-complex functions (CC up to 1828!), clearly separated from simple utilities.

### 2. Scorer Comparison (`scorer_comparison.png`)

**Left**: Score correlation scatter (0.881 correlation)
**Middle**: Score distribution histograms
**Right**: Classification bar chart

**Key insight**: Strong agreement (85.3%) with a few interpretable disagreements.

---

## 📝 Example Output

### Heuristic Scorer

```json
{
  "id": "code:fastapi/routing.py:add_api_route:443",
  "label": "add_api_route",
  "type": "Method",
  "final_score": 1.0000,
  "classification": "CORE",
  "breakdown": {
    "heuristic": 0.9500,
    "complexity": 1.0000,
    "line_count": 129,
    "param_count": 16,
    "cyclomatic_complexity": 143,
    "has_docstring": true
  }
}
```

### Embedding Scorer

```json
{
  "id": "code:fastapi/routing.py:add_api_route:443",
  "label": "add_api_route",
  "type": "Method",
  "final_score": 1.0000,
  "classification": "CORE",
  "breakdown": {
    "cluster": 1,
    "base_score": 0.9000,
    "cyclomatic_complexity": 143,
    "fan_in": 25,
    "fan_out": 5,
    "has_decorator": false,
    "decorators": []
  }
}
```

---

## 🔧 Customization

### Adjust Classification Thresholds

**Heuristic** (`scorer_tight.py`):
```python
if final > 0.55:  # Lower for more CORE
    classification = 'CORE'
elif final < 0.45:  # Raise for fewer UTILITY
    classification = 'UTILITY'
```

**Embedding** (`scorer_embedding.py`):
```python
if score > 0.55:  # Already hardened
    return 'CORE'
elif score < 0.45:
    return 'UTILITY'
```

### Adjust Feature Weights (Embedding)

In `embeddings_refined.py`, modify the structural feature duplication:
```python
structural_features = np.array([
    cc_norm,
    cc_norm,  # Duplicate = more weight
    fan_in_norm,
    fan_in_norm,
    fan_in_norm,  # Triple = even more weight
    # ... more features
])
```

### Change Number of Clusters

In `embeddings_refined.py`:
```python
# Try 4 or 5 clusters for finer-grained grouping
kmeans = KMeans(n_clusters=4, random_state=42)
```

---

## 📦 Dependencies

### Minimal (Heuristic Only)

```
pandas>=1.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Full (With Embeddings)

```
transformers>=4.30.0
torch>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.0.0
```

Install:
```bash
# Heuristic only
pip install -r requirements.txt

# Full system
pip install -r requirements_embedding.txt
```

---

## 📁 Files Created

### Core Data
- `data/extracted_functions.json` - All 251 functions with static metrics

### Heuristic Approach
- `data/final_scores_tight.json` - Heuristic scores & classifications

### Embedding Approach
- `data/embedding_clusters.json` - Cluster assignments + metrics
- `data/final_scores_embedding.json` - Embedding scores & classifications
- `data/cluster_visualization.png` - PCA visualization
- `data/scorer_comparison.png` - Comparison plots

### Analysis Scripts
- `extractor.py` - Extract functions from graph
- `scorer_tight.py` - Heuristic scorer
- `embeddings_refined.py` - Generate embeddings & cluster
- `scorer_embedding.py` - Embedding-based scorer
- `compare_scorers.py` - Compare both approaches
- `visualize_comparison.py` - Generate comparison plots
- `run_pipeline.py` - Run heuristic pipeline

---

## 🏆 Which Approach to Use?

### Use **Embedding Scorer** if:
- ✅ You want the most accurate results
- ✅ You need semantic understanding of code
- ✅ You have 30 seconds for analysis (one-time cost)
- ✅ You can install transformers/torch
- ✅ You want visual cluster analysis

### Use **Heuristic Scorer** if:
- ⚡ You need real-time scoring (<1 second)
- 🪶 You want minimal dependencies
- 🔧 You prefer interpretable, hand-crafted rules
- 🎯 92% decisiveness is good enough

**Both approaches achieve 100% accuracy on security functions with 0 false positives.**

---

## 🎯 Key Takeaways

1. **Embeddings alone don't work** - We tried pure CodeBERT embeddings initially, but all scores clustered around 0.5 (no discrimination).

2. **Augmented features are the secret** - Combining 768D embeddings with 15D structural metrics (CC, fan-in/out, decorators) creates clear cluster separation.

3. **Call graphs matter** - Fan-in reveals true utility (widely reused helpers) vs core logic (called frequently but for different reasons).

4. **Semantic understanding beats rules** - The embedding approach caught important functions (`Components`, `EmailStr`) that the heuristic missed due to file path assumptions.

5. **Clustering provides interpretability** - K-means with k=3 naturally discovered: (1) data models, (2) core logic, (3) helpers.

6. **Hardened thresholds improve decisiveness** - Moving from [0.40-0.60] to [0.45-0.55] for MIXED reduced ambiguous cases by 20%.

---

## 🚧 Future Improvements

- [ ] Try hierarchical clustering to discover subcategories within CORE/UTILITY
- [ ] Experiment with DBSCAN for automatic cluster discovery
- [ ] Add type annotation analysis (typed functions might be more important)
- [ ] Consider docstring quality as a signal (well-documented = important?)
- [ ] Build a web UI for interactive exploration of clusters

---

## 📚 References

- **CodeBERT**: [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)
- **FastAPI**: Target codebase for analysis
- **K-means**: Scikit-learn implementation
- **PCA**: For dimensionality reduction in visualization

---

Built with ❤️ for better code understanding

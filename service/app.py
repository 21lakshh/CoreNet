"""
FastAPI service for utility function detection
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np

from .models import (
    AnalysisRequest, 
    AnalysisResponse, 
    HealthResponse,
    ScoredFunction,
    StaticMetrics,
    SemanticMetrics
)
from .analyzer import (
    extract_function_data,
    score_heuristic,
    EmbeddingAnalyzer
)

# Global analyzer instance (for model caching)
embedding_analyzer = EmbeddingAnalyzer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events"""
    # Preload embedding model on startup
    embedding_analyzer.load_model()
    yield
    # Cleanup on shutdown (if needed)


app = FastAPI(
    title="Utility Function Detection Service",
    description="Detect and filter trivial utility functions from codebases using heuristic or ML-based approaches",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "approaches": ["heuristic", "embedding"]
    }


@app.post("/analyze/heuristic", response_model=AnalysisResponse)
async def analyze_heuristic(request: AnalysisRequest):
    """
    Analyze functions using heuristic approach (fast, deterministic)
    
    Uses:
    - File path patterns
    - Function naming conventions
    - Static complexity metrics (cyclomatic complexity, line count, params)
    - Adaptive weighting for security functions
    
    Thresholds:
    - CORE: score > 0.55
    - UTILITY: score < 0.45
    - MIXED: 0.45 <= score <= 0.55
    """
    try:
        # Extract nodes from analysisData
        nodes = request.get_nodes()
        
        # Extract and score
        functions = extract_function_data(nodes)
        scored = score_heuristic(functions)
        
        # Build response
        response_functions = []
        for func in scored:
            response_functions.append(ScoredFunction(
                id=func['id'],
                label=func['label'],
                type=func['type'],
                code=func['code'],
                filepath=func['filepath'],
                classification=func['classification'],
                score=func['score'],
                static_metrics=StaticMetrics(**func['static_metrics'])
            ))
        
        # Stats
        core_count = sum(1 for f in scored if f['classification'] == 'CORE')
        mixed_count = sum(1 for f in scored if f['classification'] == 'MIXED')
        utility_count = sum(1 for f in scored if f['classification'] == 'UTILITY')
        mean_score = float(np.mean([f['score'] for f in scored])) if scored else 0.0
        
        return AnalysisResponse(
            total_functions=len(scored),
            core_count=core_count,
            mixed_count=mixed_count,
            utility_count=utility_count,
            mean_score=round(mean_score, 3),
            functions=response_functions
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/embedding", response_model=AnalysisResponse)
async def analyze_embedding(request: AnalysisRequest):
    """
    Analyze functions using embedding approach (semantic, ML-based)
    
    Uses:
    - CodeBERT embeddings (semantic code understanding)
    - Augmented with structural metrics (cyclomatic complexity, fan-in/out, decorators)
    - K-Means clustering to identify function archetypes
    - Metadata refinement (complexity, connectivity, framework patterns)
    
    Thresholds:
    - CORE: score > 0.55
    - UTILITY: score < 0.45
    - MIXED: 0.45 <= score <= 0.55
    """
    try:
        # Extract nodes from analysisData
        nodes = request.get_nodes()
        
        # Extract
        functions = extract_function_data(nodes)
        
        # Score with embeddings
        scored = embedding_analyzer.analyze(functions)
        
        # Build response
        response_functions = []
        for func in scored:
            response_functions.append(ScoredFunction(
                id=func['id'],
                label=func['label'],
                type=func['type'],
                code=func['code'],
                filepath=func['filepath'],
                classification=func['classification'],
                score=func['score'],
                static_metrics=StaticMetrics(**func['static_metrics']),
                semantic_metrics=SemanticMetrics(**func['semantic_metrics']),
                cluster=func['cluster'],
                has_decorator=func['has_decorator']
            ))
        
        # Stats
        core_count = sum(1 for f in scored if f['classification'] == 'CORE')
        mixed_count = sum(1 for f in scored if f['classification'] == 'MIXED')
        utility_count = sum(1 for f in scored if f['classification'] == 'UTILITY')
        mean_score = float(np.mean([f['score'] for f in scored])) if scored else 0.0
        
        return AnalysisResponse(
            total_functions=len(scored),
            core_count=core_count,
            mixed_count=mixed_count,
            utility_count=utility_count,
            mean_score=round(mean_score, 3),
            functions=response_functions
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


"""
Pydantic models for FastAPI request/response validation
"""
from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """Input node from analysis graph"""
    id: str
    label: str
    code: str
    type: str


class AnalysisRequest(BaseModel):
    """Request body for analysis endpoints"""
    analysisData: Dict[str, Any] = Field(..., description="Analysis data containing graphNodes")
    
    def get_nodes(self) -> List[Dict[str, Any]]:
        """Extract nodes from analysisData"""
        return self.analysisData.get('graphNodes', [])


class StaticMetrics(BaseModel):
    """Static code metrics"""
    line_count: int
    param_count: int
    has_docstring: bool
    cyclomatic_complexity: int


class SemanticMetrics(BaseModel):
    """Semantic analysis metrics (embedding approach only)"""
    fan_in: int = Field(0, description="Number of incoming calls")
    fan_out: int = Field(0, description="Number of outgoing calls")


class ScoredFunction(BaseModel):
    """Analyzed and scored function"""
    id: str
    label: str
    type: str
    code: str
    filepath: str
    classification: Literal["CORE", "MIXED", "UTILITY"]
    score: float
    static_metrics: StaticMetrics
    semantic_metrics: Optional[SemanticMetrics] = None
    cluster: Optional[int] = None
    has_decorator: Optional[bool] = None


class AnalysisResponse(BaseModel):
    """Response body for analysis endpoints"""
    total_functions: int
    core_count: int
    mixed_count: int
    utility_count: int
    mean_score: float
    functions: List[ScoredFunction] = Field(..., description="All analyzed functions, sorted by score (descending)")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "total_functions": 251,
                "core_count": 138,
                "mixed_count": 20,
                "utility_count": 93,
                "mean_score": 0.540,
                "functions": [
                    {
                        "id": "code:fastapi/applications.py:get:123",
                        "label": "get",
                        "type": "Method",
                        "code": "def get(self, path: str): ...",
                        "filepath": "applications.py",
                        "classification": "CORE",
                        "score": 1.0,
                        "static_metrics": {
                            "line_count": 15,
                            "param_count": 2,
                            "has_docstring": True,
                            "cyclomatic_complexity": 3
                        }
                    }
                ]
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    approaches: List[str]


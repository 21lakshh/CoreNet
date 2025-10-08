"""
Start the FastAPI service
"""
import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("Starting Utility Function Detection Service")
    print("=" * 70)
    print("Server: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("Endpoints:")
    print("  - GET  /               (health check)")
    print("  - POST /analyze/heuristic  (fast, deterministic)")
    print("  - POST /analyze/embedding  (semantic, ML-based)")
    print("=" * 70)
    print()
    
    uvicorn.run(
        "service.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


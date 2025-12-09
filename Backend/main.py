from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from Backend.config import settings
from Backend.models import QueryRequest, QueryResponse, SourceDocument
from Backend.services.rag_pipeline import get_langchain_rag

app = FastAPI(
    title="Indian Law Chatbot API",
    description="RAG-based chatbot with auto-ingestion and query preprocessing",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize services and auto-ingest documents on startup"""
    print("="*70)
    print("🚀 Starting Indian Law Chatbot API")
    print("="*70)
    
    # Initialize RAG system
    rag = get_langchain_rag()
    
    # Auto-ingest documents
    print("\n📚 Starting auto-ingestion of documents...")
    try:
        message, new, updated, skipped = rag.auto_ingest_documents()
        print(f"\n✅ Auto-ingestion Summary:")
        print(f"   • New files: {new}")
        print(f"   • Updated files: {updated}")
        print(f"   • Skipped files: {skipped}")
        print(f"   • Status: {message}")
    except Exception as e:
        print(f"\n❌ Auto-ingestion failed: {e}")
    
    print("\n" + "="*70)
    print("✅ Server Ready!")
    print("="*70 + "\n")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Indian Law Chatbot API",
        "status": "running",
        "version": "3.0.0",
        "features": [
            "Auto-ingestion on startup",
            "Duplicate detection",
            "Query preprocessing",
            "Typo correction"
        ],
        "endpoints": {
            "query": "/query",
            "stats": "/stats",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        rag = get_langchain_rag()
        stats = rag.get_stats()
        return {
            "status": "healthy",
            "stats": stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG pipeline
    
    Automatically:
    - Preprocesses query (fixes typos, removes filler)
    - Searches relevant documents
    - Generates answer
    """
    try:
        rag = get_langchain_rag()
        answer, sources, is_relevant = rag.query(request.query)
        
        # Convert sources to SourceDocument models
        source_docs = [
            SourceDocument(
                content=src["content"],
                source=src["source"],
                page=src.get("page")
            )
            for src in sources
        ]
        
        return QueryResponse(
            answer=answer,
            sources=source_docs
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get statistics about the vector store"""
    try:
        rag = get_langchain_rag()
        return rag.get_stats()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh")
async def refresh_documents():
    """
    Manually trigger document refresh
    (Useful if you add new PDFs after server startup)
    """
    try:
        rag = get_langchain_rag()
        message, new, updated, skipped = rag.auto_ingest_documents()
        
        return {
            "status": "success",
            "message": message,
            "new_files": new,
            "updated_files": updated,
            "skipped_files": skipped
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "Backend.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=True
    )
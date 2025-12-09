import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Server Config
    BACKEND_HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
    
    # Paths
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    DATA_PATH: str = os.getenv("DATA_PATH", "./data/raw")
    
    # RAG Parameters
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5
    
    # Model Names
    LLM_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    COLLECTION_NAME: str = "indian_laws"

settings = Settings()
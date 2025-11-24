"""FastAPI main application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import ALLOWED_ORIGINS
from core.logging import setup_logger
from models.database import initialize_supabase
from api.routes import documents, qa

logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Management RAG Backend",
    description="Backend for RAG-based document Q&A with web search fallback",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
initialize_supabase()

# Include routers
app.include_router(documents.router, tags=["documents"])
app.include_router(qa.router, tags=["qa"])

logger.info("Application started successfully")


if __name__ == "__main__":
    import uvicorn
    from core.config import FASTAPI_HOST, FASTAPI_PORT

    uvicorn.run(
        "main:app",
        host=FASTAPI_HOST,
        port=FASTAPI_PORT,
        reload=True,
    )

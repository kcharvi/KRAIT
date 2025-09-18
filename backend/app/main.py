from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config import settings
from .routers import chat, health, kernel, critic

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="KRAIT - Advanced GPU kernel analysis and optimization platform with AI-powered code review",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(kernel.router, prefix="/api/v1/kernel", tags=["kernel"])
app.include_router(critic.router, prefix="/api/v1/critic", tags=["critic"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "KRAIT Backend API - Kernel Review, Analysis, and Intelligent Tuning",
        "version": settings.version,
        "docs": "/docs",
        "health": "/api/v1/health"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )

from fastapi import APIRouter
from ..models import HealthResponse
from ..config import settings

router = APIRouter(prefix="/api/v1", tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    available_providers = []
    
    # Check Gemini availability
    if settings.gemini_api_key:
        available_providers.append("gemini")
    
    # Check Hugging Face availability
    if settings.huggingface_api_key:
        available_providers.append("huggingface")
    
    return HealthResponse(
        status="healthy",
        version=settings.version,
        available_providers=available_providers
    )

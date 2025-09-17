from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
import asyncio
import json

from ..models import ChatRequest, ChatResponse, ErrorResponse, ModelInfo
from ..llm.gemini_client import GeminiClient
from ..llm.huggingface_client import HuggingFaceClient
from ..config import settings

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# Initialize clients
gemini_client = None
huggingface_client = None

def get_gemini_client() -> GeminiClient:
    global gemini_client
    if gemini_client is None:
        try:
            gemini_client = GeminiClient()
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize Gemini client: {str(e)}"
            )
    return gemini_client

def get_huggingface_client() -> HuggingFaceClient:
    global huggingface_client
    if huggingface_client is None:
        try:
            huggingface_client = HuggingFaceClient()
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize Hugging Face client: {str(e)}"
            )
    return huggingface_client

@router.post("/completions", response_model=ChatResponse)
async def create_completion(request: ChatRequest):
    """
    Create a chat completion using the specified model and provider
    """
    try:
        if request.provider == "gemini":
            client = get_gemini_client()
        elif request.provider == "huggingface":
            client = get_huggingface_client()
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported provider: {request.provider}"
            )
        
        response = await client.generate_response(
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating completion: {str(e)}"
        )

@router.post("/completions/stream")
async def create_streaming_completion(request: ChatRequest):
    """
    Create a streaming chat completion using the specified model and provider
    """
    try:
        if request.provider == "gemini":
            client = get_gemini_client()
        elif request.provider == "huggingface":
            client = get_huggingface_client()
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported provider: {request.provider}"
            )
        
        async def generate_stream():
            try:
                async for chunk in client.generate_stream_response(
                    messages=request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                ):
                    # Format as Server-Sent Events
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                
                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating streaming completion: {str(e)}"
        )

@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """
    List all available models from all providers
    """
    try:
        models = []
        
        # Get Gemini models
        try:
            gemini_client = get_gemini_client()
            gemini_models = gemini_client.get_available_models()
            models.extend([ModelInfo(**model) for model in gemini_models])
        except Exception as e:
            print(f"Warning: Could not load Gemini models: {e}")
        
        # Get Hugging Face models
        try:
            hf_client = get_huggingface_client()
            hf_models = hf_client.get_available_models()
            models.extend([ModelInfo(**model) for model in hf_models])
        except Exception as e:
            print(f"Warning: Could not load Hugging Face models: {e}")
        
        return models
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing models: {str(e)}"
        )

@router.get("/models/{provider}", response_model=List[ModelInfo])
async def list_models_by_provider(provider: str):
    """
    List models for a specific provider
    """
    try:
        if provider == "gemini":
            client = get_gemini_client()
        elif provider == "huggingface":
            client = get_huggingface_client()
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported provider: {provider}"
            )
        
        models = client.get_available_models()
        return [ModelInfo(**model) for model in models]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing {provider} models: {str(e)}"
        )

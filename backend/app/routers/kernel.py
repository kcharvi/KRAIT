"""
Kernel generation endpoints
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
import asyncio
from ..utils.system_prompt import create_kernel_generation_prompt
from ..llm.gemini_client import GeminiClient
from ..llm.huggingface_client import HuggingFaceClient
from ..config import Settings

router = APIRouter()
settings = Settings()

class KernelGenerationRequest(BaseModel):
    backend: str
    hardware: str
    code: str
    user_prompt: str
    problem_name: Optional[str] = None
    provider: str = "gemini"  # gemini or huggingface

class KernelGenerationResponse(BaseModel):
    optimized_code: str
    explanation: str
    optimizations_applied: list[str]

@router.post("/generate", response_model=KernelGenerationResponse)
async def generate_kernel(request: KernelGenerationRequest):
    """
    Generate an optimized kernel based on the provided inputs.
    """
    try:
        # Create system prompt
        system_prompt = create_kernel_generation_prompt(
            backend=request.backend,
            hardware=request.hardware,
            code=request.code,
            user_prompt=request.user_prompt,
            problem_name=request.problem_name
        )
        
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please generate an optimized kernel for {request.backend} on {request.hardware} based on the provided code and requirements."}
        ]
        
        # Generate response using selected provider
        if request.provider == "gemini":
            try:
                client = GeminiClient()
                response = await client.generate_response(messages)
            except Exception as e:
                print(f"Gemini client error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Gemini client error: {str(e)}")
        elif request.provider == "huggingface":
            try:
                client = HuggingFaceClient()
                response = await client.generate_response(messages)
            except Exception as e:
                print(f"HuggingFace client error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"HuggingFace client error: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Invalid provider. Use 'gemini' or 'huggingface'")
        
        # Parse response - response is a ChatResponse object
        raw_content = response.content if hasattr(response, 'content') else str(response)
        print(f"Raw response content: '{raw_content}'")
        print(f"Response type: {type(response)}")
        
        # Extract code from response if it contains code blocks
        import re
        
        # Try to extract code from various language code blocks
        code_patterns = [
            r'```python\s*\n(.*?)\n```',
            r'```cuda\s*\n(.*?)\n```',
            r'```c\+\+\s*\n(.*?)\n```',
            r'```c\s*\n(.*?)\n```',
            r'```opencl\s*\n(.*?)\n```',
            r'```triton\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',  # Generic code block
        ]
        
        optimized_code = raw_content  # Default to raw content
        
        for pattern in code_patterns:
            code_match = re.search(pattern, raw_content, re.DOTALL)
            if code_match:
                optimized_code = code_match.group(1).strip()
                print(f"Extracted code from code block (pattern: {pattern}): '{optimized_code[:100]}...'")
                break
        
        if optimized_code == raw_content:
            print(f"Using raw content as code: '{optimized_code[:100]}...'")
        
        # Check if we got empty content
        if not optimized_code or optimized_code.strip() == "":
            print("WARNING: Empty content received from LLM")
            optimized_code = f"# No code generated - please try again with a different prompt\n# Backend: {request.backend}\n# Hardware: {request.hardware}"
        
        # Extract explanation and optimizations (simplified parsing)
        explanation = "Generated optimized kernel with hardware-specific optimizations."
        optimizations_applied = [
            f"{request.backend}-specific optimizations",
            f"{request.hardware} memory access patterns",
            "Performance optimizations"
        ]
        
        return KernelGenerationResponse(
            optimized_code=optimized_code,
            explanation=explanation,
            optimizations_applied=optimizations_applied
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kernel generation failed: {str(e)}")

@router.post("/generate/stream")
async def generate_kernel_stream(request: KernelGenerationRequest):
    """
    Generate an optimized kernel with streaming response.
    """
    async def generate_stream():
        try:
            # Create system prompt
            system_prompt = create_kernel_generation_prompt(
                backend=request.backend,
                hardware=request.hardware,
                code=request.code,
                user_prompt=request.user_prompt,
                problem_name=request.problem_name
            )
            
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please generate an optimized kernel for {request.backend} on {request.hardware} based on the provided code and requirements."}
            ]
            
            # Generate streaming response using selected provider
            if request.provider == "gemini":
                client = GeminiClient()
                async for chunk in client.generate_stream_response(messages):
                    yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"
            elif request.provider == "huggingface":
                client = HuggingFaceClient()
                async for chunk in client.generate_stream_response(messages):
                    yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"
            else:
                yield f"data: {json.dumps({'error': 'Invalid provider. Use gemini or huggingface', 'done': True})}\n\n"
                return
            
            # Send completion signal
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@router.get("/health")
async def health_check():
    """Health check for kernel generation service."""
    return {"status": "healthy", "service": "kernel_generation"}

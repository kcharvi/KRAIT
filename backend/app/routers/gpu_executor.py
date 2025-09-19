"""
GPU executor API endpoints for KRAIT.

Provides REST API endpoints for executing CUDA kernels on external GPU resources
via GitHub + Colab integration.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import time

from ..services.github_executor import GitHubExecutor
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize GitHub executor
github_executor = None

def get_github_executor() -> GitHubExecutor:
    """Get or initialize GitHub executor."""
    global github_executor
    
    if github_executor is None:
        if not settings.github_token or not settings.github_owner:
            raise HTTPException(
                status_code=500, 
                detail="GitHub integration not configured. Please set GITHUB_TOKEN and GITHUB_OWNER environment variables."
            )
        
        github_executor = GitHubExecutor(
            github_token=settings.github_token,
            repo_name=settings.github_repo_name or "krait",
            owner=settings.github_owner
        )
        
        # Test connection
        if not github_executor.test_connection():
            raise HTTPException(
                status_code=500,
                detail="Failed to connect to GitHub repository. Please check your credentials and repository access."
            )
    
    return github_executor


class KernelExecutionRequest(BaseModel):
    """Request model for kernel execution."""
    kernel_code: str
    hardware: str = "NVIDIA T4"
    provider: str = "github_colab"
    timeout: int = 600  # Increased to 10 minutes for GitHub-Colab cycle


class KernelCompilationRequest(BaseModel):
    """Request model for kernel compilation."""
    kernel_code: str
    hardware: str = "NVIDIA T4"
    backend: str = "CUDA"
    problem_name: str = "Unknown"
    user_prompt: str = ""


class KernelFixRequest(BaseModel):
    """Request model for kernel fixing."""
    kernel_code: str
    compilation_error: str
    hardware: str = "NVIDIA T4"
    backend: str = "CUDA"
    problem_name: str = "Unknown"
    user_prompt: str = ""


class KernelExecutionResponse(BaseModel):
    """Response model for kernel execution."""
    success: bool
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    provider: str
    execution_time: Optional[float] = None


@router.post("/execute-kernel", response_model=KernelExecutionResponse)
async def execute_kernel_on_gpu(request: KernelExecutionRequest):
    """
    Execute CUDA kernel on GPU via external provider.
    
    Args:
        request: Kernel execution request containing code and parameters
        
    Returns:
        Execution results with performance metrics
    """
    try:
        logger.info(f"🔍 DEBUG: Received kernel execution request")
        logger.info(f"🔍 DEBUG: Provider: {request.provider}")
        logger.info(f"🔍 DEBUG: Hardware: {request.hardware}")
        logger.info(f"🔍 DEBUG: Kernel code length: {len(request.kernel_code)}")
        logger.info(f"🔍 DEBUG: Timeout: {request.timeout}")
        
        # Check GitHub configuration
        logger.info(f"🔍 DEBUG: GitHub token present: {bool(settings.github_token)}")
        logger.info(f"🔍 DEBUG: GitHub owner: {settings.github_owner}")
        logger.info(f"🔍 DEBUG: GitHub repo: {settings.github_repo_name}")
        
        # For now, return mock data since GitHub integration requires setup
        if request.provider == "github_colab":
            # Check if GitHub is configured
            if not settings.github_token or not settings.github_owner:
                logger.warning("🔍 DEBUG: GitHub integration not configured, returning mock data")
                logger.info(f"🔍 DEBUG: Mock data response being sent")
                return KernelExecutionResponse(
                    success=True,
                    metrics={
                        "execution_time": 100.0,
                        "gpu_utilization": 75.0,
                        "memory_usage": 512,
                        "throughput": 1.0e9,
                        "provider": "mock",
                        "timestamp": time.time()
                    },
                    provider="mock",
                    execution_time=100.0
                )
            
            logger.info(f"🔍 DEBUG: Attempting to get GitHub executor")
            try:
                executor = get_github_executor()
                logger.info(f"🔍 DEBUG: GitHub executor created successfully")
            except Exception as e:
                logger.error(f"🔍 DEBUG: Failed to create GitHub executor: {e}")
                return KernelExecutionResponse(
                    success=False,
                    error=f"GitHub executor creation failed: {str(e)}",
                    provider="github_colab"
                )
            
            # Execute kernel
            logger.info(f"🔍 DEBUG: Starting kernel execution via GitHub")
            try:
                result = await executor.execute_cuda_kernel(
                    kernel_code=request.kernel_code,
                    hardware=request.hardware,
                    timeout=request.timeout
                )
                logger.info(f"🔍 DEBUG: Kernel execution completed: {result}")
            except Exception as e:
                logger.error(f"🔍 DEBUG: Kernel execution failed: {e}")
                return KernelExecutionResponse(
                    success=False,
                    error=f"Kernel execution failed: {str(e)}",
                    provider="github_colab"
                )
            
            # Check if execution was successful
            logger.info(f"🔍 DEBUG: Checking execution result: {result}")
            if "error" in result:
                logger.error(f"🔍 DEBUG: Execution failed with error: {result['error']}")
                return KernelExecutionResponse(
                    success=False,
                    error=result["error"],
                    provider=request.provider
                )
            
            # Extract execution time if available
            execution_time = result.get("execution_time")
            if execution_time:
                execution_time = float(execution_time)
            
            logger.info(f"🔍 DEBUG: Execution successful, returning metrics")
            response = KernelExecutionResponse(
                success=True,
                metrics=result,
                provider=request.provider,
                execution_time=execution_time
            )
            logger.info(f"🔍 DEBUG: Final response: {response}")
            return response
            
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported provider: {request.provider}. Supported providers: github_colab"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🔍 DEBUG: Kernel execution failed with exception: {e}")
        logger.error(f"🔍 DEBUG: Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"🔍 DEBUG: Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Kernel execution failed: {str(e)}"
        )


@router.get("/status")
async def get_gpu_status():
    """
    Get GPU execution status and queue information.
    
    Returns:
        Current status of GPU execution services
    """
    try:
        executor = get_github_executor()
        status = await executor.get_execution_status()
        return status
        
    except Exception as e:
        logger.error(f"Failed to get GPU status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/test-connection")
async def test_gpu_connection():
    """
    Test connection to GPU execution services.
    
    Returns:
        Connection test results
    """
    try:
        executor = get_github_executor()
        connection_ok = executor.test_connection()
        
        if connection_ok:
            return {
                "success": True,
                "message": "GPU execution service connection successful",
                "provider": "github_colab"
            }
        else:
            return {
                "success": False,
                "message": "GPU execution service connection failed",
                "provider": "github_colab"
            }
            
    except Exception as e:
        logger.error(f"GPU connection test failed: {e}")
        return {
            "success": False,
            "message": f"Connection test failed: {str(e)}",
            "provider": "github_colab"
        }


@router.get("/providers")
async def get_available_providers():
    """
    Get list of available GPU execution providers.
    
    Returns:
        List of available providers and their capabilities
    """
    return {
        "providers": [
            {
                "id": "github_colab",
                "name": "Google Colab (GitHub)",
                "description": "Execute kernels on Google Colab via GitHub integration",
                "hardware": ["NVIDIA T4", "NVIDIA V100", "NVIDIA A100"],
                "features": [
                    "Real-time execution",
                    "Performance profiling",
                    "Automatic cleanup",
                    "Free GPU access"
                ],
                "status": "available"
            }
        ]
    }


@router.post("/validate-kernel")
async def validate_kernel_code(request: KernelExecutionRequest):
    """
    Validate CUDA kernel code without executing it.
    
    Args:
        request: Kernel execution request containing code to validate
        
    Returns:
        Validation results
    """
    try:
        kernel_code = request.kernel_code.strip()
        
        if not kernel_code:
            return {
                "valid": False,
                "error": "Kernel code is empty"
            }
        
        # Basic CUDA syntax validation
        validation_errors = []
        
        # Check for basic CUDA keywords
        if "__global__" not in kernel_code and "__device__" not in kernel_code:
            validation_errors.append("No CUDA kernel functions found (missing __global__ or __device__)")
        
        # Check for main function
        if "main(" not in kernel_code and "main (" not in kernel_code:
            validation_errors.append("No main function found")
        
        # Check for basic CUDA includes
        if "#include" not in kernel_code:
            validation_errors.append("No include statements found - consider adding #include <cuda_runtime.h>")
        
        if validation_errors:
            return {
                "valid": False,
                "errors": validation_errors,
                "warnings": []
            }
        
        return {
            "valid": True,
            "warnings": [
                "Kernel validation is basic - compilation may still fail",
                "Consider testing with a simple kernel first"
            ]
        }
        
    except Exception as e:
        logger.error(f"Kernel validation failed: {e}")
        return {
            "valid": False,
            "error": f"Validation failed: {str(e)}"
        }


@router.post("/compile-kernel")
async def compile_kernel(request: KernelCompilationRequest):
    """
    Compile kernel code via Colab and return compilation results.
    
    Args:
        request: Kernel compilation request containing code and parameters
        
    Returns:
        Compilation results with success status and errors
    """
    try:
        logger.info(f"🔍 DEBUG: Compiling kernel via Colab for backend: {request.backend}")
        logger.info(f"🔍 DEBUG: Hardware: {request.hardware}")
        logger.info(f"🔍 DEBUG: Kernel code length: {len(request.kernel_code)}")
        
        # Check if GitHub is configured
        if not settings.github_token or not settings.github_owner:
            logger.warning("🔍 DEBUG: GitHub integration not configured, falling back to basic validation")
            
            # Basic validation fallback
            if request.backend.lower() == "triton":
                if "@triton.jit" in request.kernel_code and "import triton" in request.kernel_code:
                    return {
                        "success": True,
                        "backend": request.backend,
                        "hardware": request.hardware,
                        "message": "Triton syntax validation successful (local)",
                        "compiled_code": request.kernel_code
                    }
                else:
                    return {
                        "success": False,
                        "backend": request.backend,
                        "hardware": request.hardware,
                        "error": "Invalid Triton syntax: missing @triton.jit decorator or import",
                        "compiled_code": request.kernel_code
                    }
            else:
                return {
                    "success": False,
                    "backend": request.backend,
                    "hardware": request.hardware,
                    "error": "GitHub integration not configured - cannot compile CUDA on Colab",
                    "compiled_code": request.kernel_code
                }
        
        # Use Colab compilation
        try:
            executor = get_github_executor()
            logger.info(f"🔍 DEBUG: GitHub executor created for compilation")
        except Exception as e:
            logger.error(f"🔍 DEBUG: Failed to create GitHub executor: {e}")
            return {
                "success": False,
                "backend": request.backend,
                "hardware": request.hardware,
                "error": f"GitHub executor creation failed: {str(e)}",
                "compiled_code": request.kernel_code
            }
        
        # Compile kernel on Colab
        logger.info(f"🔍 DEBUG: Starting Colab compilation")
        try:
            result = await executor.compile_kernel_on_colab(
                kernel_code=request.kernel_code,
                hardware=request.hardware,
                backend=request.backend,
                timeout=600  # 10 minutes for compilation
            )
            logger.info(f"🔍 DEBUG: Colab compilation completed: {result}")
        except Exception as e:
            logger.error(f"🔍 DEBUG: Colab compilation failed: {e}")
            return {
                "success": False,
                "backend": request.backend,
                "hardware": request.hardware,
                "error": f"Colab compilation failed: {str(e)}",
                "compiled_code": request.kernel_code
            }
        
        # Check if compilation was successful
        if "error" in result:
            logger.error(f"🔍 DEBUG: Compilation failed with error: {result['error']}")
            return {
                "success": False,
                "backend": request.backend,
                "hardware": request.hardware,
                "error": result["error"],
                "compiled_code": request.kernel_code
            }
        
        # Return successful compilation result
        logger.info(f"🔍 DEBUG: Compilation successful")
        return {
            "success": True,
            "backend": request.backend,
            "hardware": request.hardware,
            "message": "Compilation successful on Colab",
            "warnings": result.get("warnings", []),
            "compiled_code": request.kernel_code,
            "corrected_code": result.get("corrected_code", request.kernel_code)  # Use corrected code from Colab
        }
        
    except Exception as e:
        logger.error(f"🔍 DEBUG: Compilation failed with exception: {e}")
        return {
            "success": False,
            "backend": request.backend,
            "hardware": request.hardware,
            "error": f"Compilation failed: {str(e)}",
            "compiled_code": request.kernel_code
        }


@router.post("/fix-kernel")
async def fix_kernel_with_llm(request: KernelFixRequest):
    """
    Use LLM to fix kernel compilation errors.
    
    Args:
        request: Kernel fix request containing code, errors, and context
        
    Returns:
        Fixed kernel code or error message
    """
    try:
        logger.info(f"🔍 DEBUG: Fixing kernel with LLM")
        logger.info(f"🔍 DEBUG: Backend: {request.backend}")
        logger.info(f"🔍 DEBUG: Hardware: {request.hardware}")
        logger.info(f"🔍 DEBUG: Error: {request.compilation_error}")
        
        # Import the critic service for LLM access
        from ..critic.llm_analysis import get_llm_client
        
        # Create a detailed prompt for fixing the kernel
        fix_prompt = f"""
You are an expert GPU kernel developer. Fix the following {request.backend} kernel code that failed to compile.

COMPILATION ERROR:
{request.compilation_error}

ORIGINAL KERNEL CODE:
```{request.backend.lower()}
{request.kernel_code}
```

CONTEXT:
- Hardware: {request.hardware}
- Backend: {request.backend}
- Problem: {request.problem_name}
- User Request: {request.user_prompt}

REQUIREMENTS:
1. Fix the compilation errors while maintaining the original functionality
2. Ensure the code follows {request.backend} best practices
3. Use proper syntax for shared memory, constants, and kernel declarations
4. Return ONLY the fixed kernel code, no explanations or markdown formatting
5. For CUDA: Use #define for array sizes, not const int
6. For Triton: Ensure proper @triton.jit decorator and imports

FIXED KERNEL CODE:
"""
        
        # Get LLM client and generate fix
        llm_client = get_llm_client()
        fixed_code = await llm_client.generate_content_async(fix_prompt)
        
        # Clean up the response (remove any markdown formatting)
        fixed_code = fixed_code.strip()
        if fixed_code.startswith("```"):
            # Remove code block markers
            lines = fixed_code.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            fixed_code = '\n'.join(lines)
        
        logger.info(f"🔍 DEBUG: Generated fixed code length: {len(fixed_code)}")
        
        return {
            "success": True,
            "fixed_code": fixed_code,
            "backend": request.backend,
            "hardware": request.hardware,
            "original_error": request.compilation_error
        }
        
    except Exception as e:
        logger.error(f"🔍 DEBUG: Kernel fixing failed: {e}")
        return {
            "success": False,
            "error": f"Failed to fix kernel: {str(e)}",
            "fixed_code": request.kernel_code
        }

"""
Pydantic models for Critic Agent responses and analysis
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


class SeverityLevel(str, Enum):
    """Severity levels for suggestions and issues"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CheckStatus(str, Enum):
    """Status of individual checks"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


class CorrectnessCheck(BaseModel):
    """Individual correctness check result"""
    name: str = Field(..., description="Name of the check")
    status: CheckStatus = Field(..., description="Check result status")
    message: str = Field(..., description="Description of the check result")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional check details")


class PerformanceMetrics(BaseModel):
    """Performance analysis metrics"""
    estimated_runtime_ms: Optional[float] = Field(default=None, description="Estimated runtime in milliseconds")
    flops_total: Optional[int] = Field(default=None, description="Total floating point operations")
    memory_bound: Optional[bool] = Field(default=None, description="Whether kernel is memory bound")
    compute_bound: Optional[bool] = Field(default=None, description="Whether kernel is compute bound")
    shared_mem_per_block_bytes: Optional[int] = Field(default=None, description="Shared memory usage per block")
    efficiency_score: Optional[float] = Field(default=None, ge=0, le=100, description="Overall efficiency score (0-100)")
    peak_memory_bandwidth_utilization: Optional[float] = Field(default=None, ge=0, le=100, description="Memory bandwidth utilization %")
    peak_compute_utilization: Optional[float] = Field(default=None, ge=0, le=100, description="Compute utilization %")


class Suggestion(BaseModel):
    """Optimization suggestion"""
    severity: SeverityLevel = Field(..., description="Severity level of the suggestion")
    category: str = Field(..., description="Category of the suggestion (e.g., 'memory', 'compute', 'correctness')")
    title: str = Field(..., description="Short title of the suggestion")
    message: str = Field(..., description="Detailed description of the suggestion")
    code_snippet: Optional[str] = Field(default=None, description="Suggested code improvement")
    impact: Optional[str] = Field(default=None, description="Expected impact of the suggestion")


class CorrectnessAnalysis(BaseModel):
    """Correctness analysis results"""
    status: CheckStatus = Field(..., description="Overall correctness status")
    score: int = Field(..., ge=0, le=100, description="Correctness score (0-100)")
    checks: List[CorrectnessCheck] = Field(default_factory=list, description="Individual correctness checks")
    issues: List[str] = Field(default_factory=list, description="List of identified issues")


class PerformanceAnalysis(BaseModel):
    """Performance analysis results"""
    metrics: PerformanceMetrics = Field(..., description="Performance metrics")
    tile_detection: Optional[Dict[str, Any]] = Field(default=None, description="Tiling pattern detection results")
    vectorization_detected: Optional[bool] = Field(default=None, description="Whether vectorization is detected")
    tensor_core_usage: Optional[bool] = Field(default=None, description="Whether Tensor Cores are used")
    memory_access_patterns: Optional[Dict[str, Any]] = Field(default=None, description="Memory access pattern analysis")


class HardwareUtilization(BaseModel):
    """Hardware utilization analysis"""
    gpu_utilization: Optional[float] = Field(default=None, ge=0, le=100, description="GPU utilization percentage")
    memory_utilization: Optional[float] = Field(default=None, ge=0, le=100, description="Memory utilization percentage")
    compute_utilization: Optional[float] = Field(default=None, ge=0, le=100, description="Compute utilization percentage")
    bottleneck: Optional[str] = Field(default=None, description="Identified performance bottleneck")


class CriticAnalysis(BaseModel):
    """Complete critic analysis result"""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    kernel_code: str = Field(..., description="Analyzed kernel code")
    hardware_config: str = Field(..., description="Target hardware configuration")
    backend: str = Field(..., description="Target backend (CUDA, Triton, etc.)")
    
    # Analysis results
    correctness: CorrectnessAnalysis = Field(..., description="Correctness analysis")
    performance: PerformanceAnalysis = Field(..., description="Performance analysis")
    hardware_utilization: Optional[HardwareUtilization] = Field(default=None, description="Hardware utilization analysis")
    
    # Suggestions and recommendations
    suggestions: List[Suggestion] = Field(default_factory=list, description="Optimization suggestions")
    overall_score: int = Field(..., ge=0, le=100, description="Overall analysis score (0-100)")
    
    # Metadata
    analysis_time_ms: Optional[float] = Field(default=None, description="Analysis time in milliseconds")
    generated_at: str = Field(..., description="Analysis timestamp")
    version: str = Field(default="1.0.0", description="Analysis version")


class CriticAnalysisRequest(BaseModel):
    """Request model for kernel analysis"""
    kernel_code: str = Field(..., description="Kernel code to analyze")
    hardware: str = Field(..., description="Target hardware (e.g., NVIDIA H100, AMD MI300X)")
    backend: str = Field(..., description="Target backend (e.g., CUDA, Triton, OpenCL)")
    problem_name: Optional[str] = Field(default=None, description="Name of the problem type")
    analysis_options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional analysis options")
    use_llm_review: bool = Field(default=False, description="Enable LLM-based correctness review")


class BatchAnalysisRequest(BaseModel):
    """Request model for batch kernel analysis"""
    kernels: List[CriticAnalysisRequest] = Field(..., description="List of kernels to analyze")
    parallel: bool = Field(default=True, description="Whether to run analysis in parallel")


class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis"""
    results: List[CriticAnalysis] = Field(..., description="Analysis results for each kernel")
    total_time_ms: float = Field(..., description="Total analysis time in milliseconds")
    success_count: int = Field(..., description="Number of successful analyses")
    error_count: int = Field(..., description="Number of failed analyses")
    errors: List[Dict[str, str]] = Field(default_factory=list, description="Error details for failed analyses")


class CheckType(BaseModel):
    """Available check type information"""
    name: str = Field(..., description="Check name")
    category: str = Field(..., description="Check category")
    description: str = Field(..., description="Check description")
    severity_levels: List[SeverityLevel] = Field(..., description="Available severity levels")
    enabled: bool = Field(default=True, description="Whether check is enabled")


class CriticHealthResponse(BaseModel):
    """Health check response for critic service"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    available_checks: List[CheckType] = Field(..., description="Available check types")
    supported_hardware: List[str] = Field(..., description="Supported hardware platforms")
    supported_backends: List[str] = Field(..., description="Supported backends")

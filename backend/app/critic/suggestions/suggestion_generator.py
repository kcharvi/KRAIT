"""
Intelligent suggestion generator for kernel analysis
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from ..critic_models import Suggestion, SeverityLevel, CorrectnessCheck, PerformanceMetrics
from ..parsers.parser_factory import ParserFactory


class SuggestionGenerator:
    """Generate context-aware suggestions for kernel optimization"""
    
    def __init__(self):
        self.name = "suggestion_generator"
        self.description = "Generate intelligent, context-aware optimization suggestions"
        self.parser_factory = ParserFactory()
        
        # Suggestion templates by category
        self.suggestion_templates = {
            "bounds_checking": {
                "cuda": [
                    "Add bounds checking: if (threadIdx.x < width && threadIdx.y < height)",
                    "Use block dimensions for bounds: if (blockIdx.x < gridDim.x)",
                    "Check array bounds before access: if (index < array_size)"
                ],
                "triton": [
                    "Use Triton's built-in bounds checking with proper block sizes",
                    "Ensure block dimensions don't exceed tensor dimensions"
                ],
                "opencl": [
                    "Use get_global_id() and get_global_size() for bounds checking",
                    "Add work item bounds: if (get_global_id(0) < global_size)"
                ]
            },
            "memory_optimization": {
                "cuda": [
                    "Use shared memory for frequently accessed data: __shared__ float tile[16][16]",
                    "Implement memory coalescing: threadIdx.x + blockIdx.x * blockDim.x",
                    "Use vectorized loads: float4 *ptr = (float4*)global_ptr",
                    "Consider read-only cache: __ldg(ptr)"
                ],
                "triton": [
                    "Use tl.load() and tl.store() for efficient memory access",
                    "Implement proper tiling with BLOCK_SIZE constants",
                    "Use Triton's built-in memory optimization features"
                ],
                "opencl": [
                    "Use local memory for data reuse: __local float tile[16][16]",
                    "Implement work group barriers: barrier(CLK_LOCAL_MEM_FENCE)",
                    "Use vectorized data types: float4, float8"
                ]
            },
            "compute_optimization": {
                "cuda": [
                    "Use Tensor Cores for matrix operations: wmma::fragment",
                    "Implement loop unrolling: #pragma unroll",
                    "Use fused multiply-add: __fmaf_rn()",
                    "Consider warp-level primitives: __shfl_sync()"
                ],
                "triton": [
                    "Use tl.dot() for matrix multiplication",
                    "Implement proper block tiling for compute efficiency",
                    "Use Triton's built-in reduction operations"
                ],
                "opencl": [
                    "Use vectorized operations: float4 arithmetic",
                    "Implement work group functions: work_group_reduce_add()",
                    "Use built-in math functions: native_* functions"
                ]
            },
            "synchronization": {
                "cuda": [
                    "Add synchronization after shared memory writes: __syncthreads()",
                    "Use memory fences for consistency: __threadfence()",
                    "Consider warp-level synchronization: __syncwarp()"
                ],
                "triton": [
                    "Triton handles synchronization automatically",
                    "Ensure proper block size for efficient execution"
                ],
                "opencl": [
                    "Use work group barriers: barrier(CLK_LOCAL_MEM_FENCE)",
                    "Implement proper synchronization for shared data"
                ]
            }
        }
        
        # Hardware-specific optimization hints
        self.hardware_hints = {
            "NVIDIA H100": {
                "tensor_cores": "Use Tensor Cores with FP16/BF16 for matrix operations",
                "memory": "Leverage HBM3 memory with large shared memory blocks",
                "compute": "Use CUDA 12.0+ features and warp-level primitives"
            },
            "NVIDIA A100": {
                "tensor_cores": "Use Tensor Cores with FP16 for 2x performance boost",
                "memory": "Utilize 40GB HBM2e memory efficiently",
                "compute": "Use CUDA 11.0+ features and multi-GPU scaling"
            },
            "AMD MI300X": {
                "tensor_cores": "Use Matrix Cores for AI workloads",
                "memory": "Leverage HBM3 memory with 128GB capacity",
                "compute": "Use ROCm 5.0+ and HIP for optimal performance"
            }
        }
    
    async def generate_suggestions(self, 
                                 kernel_code: str, 
                                 hardware: str, 
                                 backend: str,
                                 correctness_checks: List[CorrectnessCheck],
                                 performance_metrics: PerformanceMetrics,
                                 analysis_context: Dict[str, Any]) -> List[Suggestion]:
        """Generate comprehensive suggestions based on analysis results"""
        
        # Parse the kernel code for context
        parsed_kernel = await self.parser_factory.parse(kernel_code)
        
        suggestions = []
        
        # Generate correctness-based suggestions
        correctness_suggestions = await self._generate_correctness_suggestions(
            correctness_checks, parsed_kernel, backend
        )
        suggestions.extend(correctness_suggestions)
        
        # Generate performance-based suggestions
        performance_suggestions = await self._generate_performance_suggestions(
            performance_metrics, parsed_kernel, hardware, backend
        )
        suggestions.extend(performance_suggestions)
        
        # Generate context-aware suggestions
        context_suggestions = await self._generate_context_suggestions(
            parsed_kernel, hardware, backend, analysis_context
        )
        suggestions.extend(context_suggestions)
        
        # Generate hardware-specific suggestions
        hardware_suggestions = await self._generate_hardware_suggestions(
            parsed_kernel, hardware, backend
        )
        suggestions.extend(hardware_suggestions)
        
        # Prioritize and deduplicate suggestions
        suggestions = self._prioritize_suggestions(suggestions)
        suggestions = self._deduplicate_suggestions(suggestions)
        
        return suggestions
    
    async def _generate_correctness_suggestions(self, 
                                              correctness_checks: List[CorrectnessCheck],
                                              parsed_kernel,
                                              backend: str) -> List[Suggestion]:
        """Generate suggestions based on correctness analysis"""
        suggestions = []
        
        for check in correctness_checks:
            if check.status.value in ["fail", "warning"]:
                # Generate specific suggestions based on check type
                if "bounds" in check.name.lower():
                    suggestions.extend(self._generate_bounds_suggestions(check, parsed_kernel, backend))
                elif "sync" in check.name.lower():
                    suggestions.extend(self._generate_sync_suggestions(check, parsed_kernel, backend))
                elif "memory" in check.name.lower():
                    suggestions.extend(self._generate_memory_safety_suggestions(check, parsed_kernel, backend))
                elif "type" in check.name.lower():
                    suggestions.extend(self._generate_type_safety_suggestions(check, parsed_kernel, backend))
        
        return suggestions
    
    async def _generate_performance_suggestions(self,
                                              performance_metrics: PerformanceMetrics,
                                              parsed_kernel,
                                              hardware: str,
                                              backend: str) -> List[Suggestion]:
        """Generate suggestions based on performance analysis"""
        suggestions = []
        
        # Memory-bound kernel suggestions
        if performance_metrics.bound == "memory":
            suggestions.append(Suggestion(
                severity=SeverityLevel.HIGH,
                category="memory",
                title="Memory-Bound Kernel",
                message="Kernel is memory-bound. Consider optimizing memory access patterns.",
                code_snippet=self._get_memory_optimization_snippet(backend)
            ))
        
        # Low shared memory usage
        if performance_metrics.shared_mem_per_block_bytes < 1024:
            suggestions.append(Suggestion(
                severity=SeverityLevel.MEDIUM,
                category="memory",
                title="Low Shared Memory Usage",
                message="Consider using more shared memory to reduce global memory access.",
                code_snippet=self._get_shared_memory_snippet(backend)
            ))
        
        # No tiling detected
        if not performance_metrics.tiling_detected:
            suggestions.append(Suggestion(
                severity=SeverityLevel.HIGH,
                category="memory",
                title="No Tiling Detected",
                message="Implement tiling to improve memory access patterns and cache utilization.",
                code_snippet=self._get_tiling_snippet(backend)
            ))
        
        # No vectorization detected
        if not performance_metrics.vectorization_detected:
            suggestions.append(Suggestion(
                severity=SeverityLevel.MEDIUM,
                category="compute",
                title="No Vectorization Detected",
                message="Consider using vectorized operations for better performance.",
                code_snippet=self._get_vectorization_snippet(backend)
            ))
        
        # Tensor Core usage opportunity
        if "NVIDIA" in hardware and not performance_metrics.tensor_core_usage_detected:
            suggestions.append(Suggestion(
                severity=SeverityLevel.HIGH,
                category="compute",
                title="Tensor Core Opportunity",
                message="Consider using Tensor Cores for matrix operations on NVIDIA hardware.",
                code_snippet=self._get_tensor_core_snippet()
            ))
        
        return suggestions
    
    async def _generate_context_suggestions(self,
                                          parsed_kernel,
                                          hardware: str,
                                          backend: str,
                                          analysis_context: Dict[str, Any]) -> List[Suggestion]:
        """Generate context-aware suggestions based on code patterns"""
        suggestions = []
        
        # Analyze code patterns
        code_patterns = self._analyze_code_patterns(parsed_kernel)
        
        # Suggest optimizations based on patterns
        for pattern, details in code_patterns.items():
            if pattern == "nested_loops" and details["depth"] > 2:
                suggestions.append(Suggestion(
                    severity=SeverityLevel.MEDIUM,
                    category="compute",
                    title="Deep Loop Nesting",
                    message=f"Consider flattening {details['depth']}-level nested loops for better performance.",
                    code_snippet=self._get_loop_flattening_snippet(backend)
                ))
            
            elif pattern == "scalar_operations" and details["count"] > 10:
                suggestions.append(Suggestion(
                    severity=SeverityLevel.MEDIUM,
                    category="compute",
                    title="Many Scalar Operations",
                    message=f"Consider vectorizing {details['count']} scalar operations.",
                    code_snippet=self._get_vectorization_snippet(backend)
                ))
            
            elif pattern == "global_memory_access" and details["count"] > 5:
                suggestions.append(Suggestion(
                    severity=SeverityLevel.HIGH,
                    category="memory",
                    title="Excessive Global Memory Access",
                    message=f"Consider using shared memory for {details['count']} global memory accesses.",
                    code_snippet=self._get_shared_memory_snippet(backend)
                ))
        
        return suggestions
    
    async def _generate_hardware_suggestions(self,
                                           parsed_kernel,
                                           hardware: str,
                                           backend: str) -> List[Suggestion]:
        """Generate hardware-specific optimization suggestions"""
        suggestions = []
        
        if hardware in self.hardware_hints:
            hints = self.hardware_hints[hardware]
            
            for category, hint in hints.items():
                suggestions.append(Suggestion(
                    severity=SeverityLevel.MEDIUM,
                    category=category,
                    title=f"{hardware} Optimization",
                    message=hint,
                    code_snippet=self._get_hardware_specific_snippet(hardware, category, backend)
                ))
        
        return suggestions
    
    def _analyze_code_patterns(self, parsed_kernel) -> Dict[str, Any]:
        """Analyze code patterns for optimization opportunities"""
        patterns = {}
        
        # Count nested loops
        loop_depth = self._calculate_loop_depth(parsed_kernel.loops)
        if loop_depth > 0:
            patterns["nested_loops"] = {"depth": loop_depth}
        
        # Count scalar operations
        scalar_ops = len(re.findall(r'(\w+)\s*[+\-*/]\s*(\w+)', parsed_kernel.raw_code))
        if scalar_ops > 0:
            patterns["scalar_operations"] = {"count": scalar_ops}
        
        # Count global memory accesses
        global_accesses = len([acc for acc in parsed_kernel.memory_accesses if acc.is_global])
        if global_accesses > 0:
            patterns["global_memory_access"] = {"count": global_accesses}
        
        return patterns
    
    def _calculate_loop_depth(self, loops: List) -> int:
        """Calculate maximum loop nesting depth"""
        if not loops:
            return 0
        
        # Simple heuristic: count loops with similar variable names
        max_depth = 1
        for loop in loops:
            if loop.type == "for":
                # Check if this loop is nested
                depth = 1
                for other_loop in loops:
                    if (other_loop.line_number > loop.line_number and 
                        other_loop.variable != loop.variable):
                        depth += 1
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _generate_bounds_suggestions(self, check: CorrectnessCheck, parsed_kernel, backend: str) -> List[Suggestion]:
        """Generate bounds checking suggestions"""
        suggestions = []
        
        if backend.upper() == "CUDA":
            suggestions.append(Suggestion(
                severity=SeverityLevel.HIGH,
                category="correctness",
                title="Bounds Checking Required",
                message="Add proper bounds checking to prevent out-of-bounds access.",
                code_snippet="if (threadIdx.x < width && threadIdx.y < height) {\n    // Your kernel code here\n}"
            ))
        
        return suggestions
    
    def _generate_sync_suggestions(self, check: CorrectnessCheck, parsed_kernel, backend: str) -> List[Suggestion]:
        """Generate synchronization suggestions"""
        suggestions = []
        
        if backend.upper() == "CUDA":
            suggestions.append(Suggestion(
                severity=SeverityLevel.HIGH,
                category="correctness",
                title="Synchronization Required",
                message="Add proper synchronization after shared memory operations.",
                code_snippet="__shared__ float tile[16][16];\n// ... shared memory operations ...\n__syncthreads();"
            ))
        
        return suggestions
    
    def _generate_memory_safety_suggestions(self, check: CorrectnessCheck, parsed_kernel, backend: str) -> List[Suggestion]:
        """Generate memory safety suggestions"""
        suggestions = []
        
        suggestions.append(Suggestion(
            severity=SeverityLevel.HIGH,
            category="correctness",
            title="Memory Safety Issue",
            message="Fix potential memory safety issues to prevent crashes.",
            code_snippet="// Add proper null checks and bounds validation\nif (ptr != nullptr && index < size) {\n    // Safe memory access\n}"
        ))
        
        return suggestions
    
    def _generate_type_safety_suggestions(self, check: CorrectnessCheck, parsed_kernel, backend: str) -> List[Suggestion]:
        """Generate type safety suggestions"""
        suggestions = []
        
        suggestions.append(Suggestion(
            severity=SeverityLevel.MEDIUM,
            category="correctness",
            title="Type Safety Issue",
            message="Ensure type consistency to prevent runtime errors.",
            code_snippet="// Use explicit type casting\nfloat result = (float)int_value / (float)int_divisor;"
        ))
        
        return suggestions
    
    def _get_memory_optimization_snippet(self, backend: str) -> str:
        """Get memory optimization code snippet"""
        if backend.upper() == "CUDA":
            return """// Memory coalescing example
int idx = threadIdx.x + blockIdx.x * blockDim.x;
if (idx < size) {
    float4 *ptr = (float4*)&data[idx * 4];
    float4 val = *ptr;  // Coalesced load
}"""
        elif backend.upper() == "TRITON":
            return """# Triton memory optimization
@triton.jit
def kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)"""
        else:
            return "// Implement memory coalescing and vectorized loads"
    
    def _get_shared_memory_snippet(self, backend: str) -> str:
        """Get shared memory code snippet"""
        if backend.upper() == "CUDA":
            return """__global__ void kernel(float* data, int size) {
    __shared__ float tile[16][16];
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // Load data into shared memory
    if (tx < 16 && ty < 16) {
        tile[ty][tx] = data[ty * size + tx];
    }
    __syncthreads();
    
    // Use shared memory for computation
    // ... computation code ...
}"""
        elif backend.upper() == "OPENCL":
            return """__kernel void kernel(__global float* data, int size) {
    __local float tile[16][16];
    int tx = get_local_id(0), ty = get_local_id(1);
    
    // Load data into local memory
    if (tx < 16 && ty < 16) {
        tile[ty][tx] = data[ty * size + tx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Use local memory for computation
    // ... computation code ...
}"""
        else:
            return "// Implement shared/local memory tiling"
    
    def _get_tiling_snippet(self, backend: str) -> str:
        """Get tiling code snippet"""
        if backend.upper() == "CUDA":
            return """// Matrix multiplication with tiling
__global__ void matmul_tiled(float* A, float* B, float* C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    for (int k = 0; k < N; k += TILE_SIZE) {
        // Load tiles
        tile_A[ty][tx] = A[row * N + k + tx];
        tile_B[ty][tx] = B[(k + ty) * N + col];
        __syncthreads();
        
        // Compute
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tile_A[ty][i] * tile_B[i][tx];
        }
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}"""
        else:
            return "// Implement tiling for better cache utilization"
    
    def _get_vectorization_snippet(self, backend: str) -> str:
        """Get vectorization code snippet"""
        if backend.upper() == "CUDA":
            return """// Vectorized memory operations
__global__ void vectorized_kernel(float4* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 val = data[idx];  // Load 4 floats at once
        val.x *= 2.0f; val.y *= 2.0f; val.z *= 2.0f; val.w *= 2.0f;
        data[idx] = val;  // Store 4 floats at once
    }
}"""
        elif backend.upper() == "OPENCL":
            return """// OpenCL vectorized operations
__kernel void vectorized_kernel(__global float4* data, int n) {
    int idx = get_global_id(0);
    if (idx < n) {
        float4 val = data[idx];
        val *= 2.0f;  // Vectorized multiplication
        data[idx] = val;
    }
}"""
        else:
            return "// Use vectorized data types for better performance"
    
    def _get_tensor_core_snippet(self) -> str:
        """Get Tensor Core code snippet"""
        return """// Tensor Core matrix multiplication
#include <mma.h>

__global__ void tensor_core_matmul(half* A, half* B, float* C, int M, int N, int K) {
    using namespace nvcuda::wmma;
    
    // Declare fragments
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> acc_frag;
    
    // Initialize accumulator
    fill_fragment(acc_frag, 0.0f);
    
    // Load and compute
    load_matrix_sync(a_frag, A, K);
    load_matrix_sync(b_frag, B, K);
    mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    
    // Store result
    store_matrix_sync(C, acc_frag, N, C_LAYOUT);
}"""
    
    def _get_loop_flattening_snippet(self, backend: str) -> str:
        """Get loop flattening code snippet"""
        return """// Flatten nested loops for better performance
// Before: nested loops
for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
        // computation
    }
}

// After: flattened loop
for (int idx = 0; idx < height * width; idx++) {
    int i = idx / width;
    int j = idx % width;
    // computation
}"""
    
    def _get_hardware_specific_snippet(self, hardware: str, category: str, backend: str) -> str:
        """Get hardware-specific code snippet"""
        if hardware == "NVIDIA H100" and category == "tensor_cores":
            return self._get_tensor_core_snippet()
        elif "NVIDIA" in hardware and category == "memory":
            return self._get_memory_optimization_snippet(backend)
        else:
            return f"// {hardware} specific {category} optimization"
    
    def _prioritize_suggestions(self, suggestions: List[Suggestion]) -> List[Suggestion]:
        """Prioritize suggestions by severity and impact"""
        severity_order = {
            SeverityLevel.CRITICAL: 0,
            SeverityLevel.HIGH: 1,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.LOW: 3
        }
        
        return sorted(suggestions, key=lambda s: severity_order.get(s.severity, 4))
    
    def _deduplicate_suggestions(self, suggestions: List[Suggestion]) -> List[Suggestion]:
        """Remove duplicate suggestions"""
        seen = set()
        unique_suggestions = []
        
        for suggestion in suggestions:
            # Create a key based on title and message
            key = (suggestion.title, suggestion.message)
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions

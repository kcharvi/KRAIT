"""
Hardware-specific optimization hints and recommendations
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from ..critic_models import Suggestion, SeverityLevel


class HardwareOptimizer:
    """Generate hardware-specific optimization suggestions"""
    
    def __init__(self):
        self.name = "hardware_optimizer"
        self.description = "Generate hardware-specific optimization suggestions"
        
        # Hardware specifications and capabilities
        self.hardware_specs = {
            "NVIDIA H100": {
                "compute_capability": 9.0,
                "sm_count": 132,
                "max_threads_per_sm": 2048,
                "max_threads_per_block": 1024,
                "shared_memory_per_sm": 228 * 1024,  # 228 KB
                "shared_memory_per_block": 228 * 1024,
                "max_shared_memory_per_block": 228 * 1024,
                "memory_bandwidth": 3350,  # GB/s
                "peak_flops": 989e12,  # FP32
                "tensor_cores": True,
                "tensor_core_types": ["FP16", "BF16", "FP32", "INT8", "INT4"],
                "warp_size": 32,
                "max_blocks_per_sm": 32,
                "registers_per_sm": 65536,
                "l2_cache_size": 50 * 1024 * 1024,  # 50 MB
                "l1_cache_size": 256 * 1024,  # 256 KB
            },
            "NVIDIA A100": {
                "compute_capability": 8.0,
                "sm_count": 108,
                "max_threads_per_sm": 2048,
                "max_threads_per_block": 1024,
                "shared_memory_per_sm": 164 * 1024,  # 164 KB
                "shared_memory_per_block": 164 * 1024,
                "max_shared_memory_per_block": 164 * 1024,
                "memory_bandwidth": 2039,  # GB/s
                "peak_flops": 312e12,  # FP32
                "tensor_cores": True,
                "tensor_core_types": ["FP16", "BF16", "FP32", "INT8", "INT4"],
                "warp_size": 32,
                "max_blocks_per_sm": 32,
                "registers_per_sm": 65536,
                "l2_cache_size": 40 * 1024 * 1024,  # 40 MB
                "l1_cache_size": 192 * 1024,  # 192 KB
            },
            "AMD MI300X": {
                "compute_capability": 9.0,
                "cu_count": 304,  # Compute Units
                "max_threads_per_cu": 2560,
                "max_threads_per_workgroup": 1024,
                "local_memory_per_cu": 64 * 1024,  # 64 KB
                "local_memory_per_workgroup": 64 * 1024,
                "max_local_memory_per_workgroup": 64 * 1024,
                "memory_bandwidth": 5600,  # GB/s
                "peak_flops": 1634e12,  # FP32
                "matrix_cores": True,
                "matrix_core_types": ["FP16", "BF16", "FP32", "INT8"],
                "wavefront_size": 64,
                "max_workgroups_per_cu": 40,
                "registers_per_cu": 65536,
                "l2_cache_size": 8 * 1024 * 1024,  # 8 MB per CU
                "l1_cache_size": 16 * 1024,  # 16 KB per CU
            },
            "CPU": {
                "cores": 64,  # Typical high-end CPU
                "threads_per_core": 2,
                "l1_cache_size": 32 * 1024,  # 32 KB
                "l2_cache_size": 1024 * 1024,  # 1 MB
                "l3_cache_size": 64 * 1024 * 1024,  # 64 MB
                "memory_bandwidth": 100,  # GB/s
                "peak_flops": 100e9,  # FP32
                "simd_width": 8,  # AVX-512
                "vectorization": True,
            }
        }
        
        # Optimization patterns for each hardware
        self.optimization_patterns = {
            "NVIDIA H100": {
                "tensor_cores": {
                    "pattern": r"float\s+\w+\s*[+\-*/]\s*float\s+\w+",
                    "suggestion": "Use Tensor Cores with FP16/BF16 for 2-4x performance boost",
                    "code_example": """
// Use Tensor Cores for matrix operations
#include <mma.h>
using namespace nvcuda::wmma;

fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> acc_frag;

load_matrix_sync(a_frag, A, K);
load_matrix_sync(b_frag, B, K);
mma_sync(acc_frag, a_frag, b_frag, acc_frag);
store_matrix_sync(C, acc_frag, N, C_LAYOUT);"""
                },
                "memory_optimization": {
                    "pattern": r"__shared__\s+float\s+\w+\[\d+\]",
                    "suggestion": "Use larger shared memory blocks (up to 228KB) for better cache utilization",
                    "code_example": """
__shared__ float tile[32][32];  // Use larger tiles
// Load data with coalesced access
int tx = threadIdx.x, ty = threadIdx.y;
int bx = blockIdx.x, by = blockIdx.y;

int row = by * 32 + ty;
int col = bx * 32 + tx;

if (row < M && col < N) {
    tile[ty][tx] = A[row * K + col];
}"""
                },
                "warp_optimization": {
                    "pattern": r"if\s*\(\s*threadIdx\.x\s*<\s*\d+",
                    "suggestion": "Use warp-level primitives for better efficiency",
                    "code_example": """
// Use warp-level reduction
float val = thread_data;
for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
}
if (threadIdx.x == 0) {
    result[blockIdx.x] = val;
}"""
                }
            },
            "NVIDIA A100": {
                "tensor_cores": {
                    "pattern": r"float\s+\w+\s*[+\-*/]\s*float\s+\w+",
                    "suggestion": "Use Tensor Cores with FP16 for matrix operations",
                    "code_example": """
// A100 Tensor Core optimization
__global__ void tensor_matmul(half* A, half* B, float* C, int M, int N, int K) {
    using namespace nvcuda::wmma;
    
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> acc_frag;
    
    fill_fragment(acc_frag, 0.0f);
    load_matrix_sync(a_frag, A, K);
    load_matrix_sync(b_frag, B, K);
    mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    store_matrix_sync(C, acc_frag, N, C_LAYOUT);
}"""
                },
                "memory_optimization": {
                    "pattern": r"__shared__\s+float\s+\w+\[\d+\]",
                    "suggestion": "Optimize shared memory usage (max 164KB per block)",
                    "code_example": """
__shared__ float tile[16][16];  // Optimal for A100
// Use read-only cache for global memory
__ldg(&global_data[idx]);"""
                }
            },
            "AMD MI300X": {
                "matrix_cores": {
                    "pattern": r"float\s+\w+\s*[+\-*/]\s*float\s+\w+",
                    "suggestion": "Use Matrix Cores for AI workloads on MI300X",
                    "code_example": """
// AMD Matrix Core optimization
__kernel void matrix_multiply(__global float* A, __global float* B, __global float* C) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    
    // Use work group functions for reduction
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[gid * K + k] * B[k * N + gid];
    }
    
    // Work group reduction
    sum = work_group_reduce_add(sum);
    
    if (lid == 0) {
        C[gid] = sum;
    }
}"""
                },
                "memory_optimization": {
                    "pattern": r"__local\s+float\s+\w+\[\d+\]",
                    "suggestion": "Use local memory efficiently (max 64KB per workgroup)",
                    "code_example": """
__local float tile[16][16];
int lid = get_local_id(0);
int gid = get_global_id(0);

// Load data into local memory
if (lid < 256) {
    tile[lid / 16][lid % 16] = A[gid];
}
barrier(CLK_LOCAL_MEM_FENCE);"""
                }
            }
        }
    
    async def generate_hardware_suggestions(self, 
                                          kernel_code: str, 
                                          hardware: str, 
                                          backend: str,
                                          performance_metrics: Dict[str, Any]) -> List[Suggestion]:
        """Generate hardware-specific optimization suggestions"""
        
        if hardware not in self.hardware_specs:
            return []
        
        suggestions = []
        hardware_spec = self.hardware_specs[hardware]
        
        # Generate suggestions based on hardware capabilities
        suggestions.extend(await self._generate_tensor_core_suggestions(kernel_code, hardware, backend))
        suggestions.extend(await self._generate_memory_suggestions(kernel_code, hardware, backend, hardware_spec))
        suggestions.extend(await self._generate_compute_suggestions(kernel_code, hardware, backend, hardware_spec))
        suggestions.extend(await self._generate_cache_suggestions(kernel_code, hardware, backend, hardware_spec))
        
        return suggestions
    
    async def _generate_tensor_core_suggestions(self, kernel_code: str, hardware: str, backend: str) -> List[Suggestion]:
        """Generate Tensor Core optimization suggestions"""
        suggestions = []
        
        if hardware not in ["NVIDIA H100", "NVIDIA A100"]:
            return suggestions
        
        # Check if Tensor Cores are already being used
        tensor_core_patterns = [
            r'wmma\.|mma\.',
            r'tensor',
            r'float16.*float16',
            r'__half'
        ]
        
        has_tensor_cores = any(re.search(pattern, kernel_code, re.IGNORECASE) for pattern in tensor_core_patterns)
        
        if not has_tensor_cores:
            # Check for matrix multiplication patterns
            matmul_patterns = [
                r'(\w+)\s*=\s*(\w+)\s*\*\s*(\w+)',
                r'(\w+)\s*\[(\w+)\]\s*=\s*(\w+)\s*\[(\w+)\]\s*\*\s*(\w+)\s*\[(\w+)\]'
            ]
            
            has_matmul = any(re.search(pattern, kernel_code, re.IGNORECASE) for pattern in matmul_patterns)
            
            if has_matmul:
                suggestions.append(Suggestion(
                    severity=SeverityLevel.HIGH,
                    category="compute",
                    title="Tensor Core Opportunity",
                    message=f"Use Tensor Cores on {hardware} for 2-4x performance boost in matrix operations",
                    code_snippet=self.optimization_patterns[hardware]["tensor_cores"]["code_example"]
                ))
        
        return suggestions
    
    async def _generate_memory_suggestions(self, kernel_code: str, hardware: str, backend: str, hardware_spec: Dict) -> List[Suggestion]:
        """Generate memory optimization suggestions"""
        suggestions = []
        
        # Check shared memory usage
        shared_mem_patterns = [
            r'__shared__\s+\w+\s+(\w+)\s*\[(\d+)\]',
            r'__local\s+\w+\s+(\w+)\s*\[(\d+)\]'
        ]
        
        for pattern in shared_mem_patterns:
            matches = re.findall(pattern, kernel_code)
            for var_name, size_str in matches:
                try:
                    size = int(size_str)
                    max_shared_mem = hardware_spec.get("max_shared_memory_per_block", 0)
                    
                    if size < max_shared_mem * 0.5:  # Using less than 50% of available shared memory
                        suggestions.append(Suggestion(
                            severity=SeverityLevel.MEDIUM,
                            category="memory",
                            title="Underutilized Shared Memory",
                            message=f"Consider using more shared memory (currently {size} bytes, max {max_shared_mem} bytes)",
                            code_snippet=self._generate_shared_memory_snippet(hardware, backend, max_shared_mem)
                        ))
                    elif size > max_shared_mem:  # Exceeding shared memory limit
                        suggestions.append(Suggestion(
                            severity=SeverityLevel.HIGH,
                            category="memory",
                            title="Shared Memory Overflow",
                            message=f"Shared memory usage ({size} bytes) exceeds limit ({max_shared_mem} bytes)",
                            code_snippet=self._generate_shared_memory_snippet(hardware, backend, max_shared_mem)
                        ))
                except ValueError:
                    continue
        
        # Check for memory coalescing
        if not self._has_memory_coalescing(kernel_code, backend):
            suggestions.append(Suggestion(
                severity=SeverityLevel.HIGH,
                category="memory",
                title="Memory Coalescing Missing",
                message="Implement memory coalescing for better memory bandwidth utilization",
                code_snippet=self._generate_coalescing_snippet(backend)
            ))
        
        return suggestions
    
    async def _generate_compute_suggestions(self, kernel_code: str, hardware: str, backend: str, hardware_spec: Dict) -> List[Suggestion]:
        """Generate compute optimization suggestions"""
        suggestions = []
        
        # Check for warp/wavefront optimization opportunities
        if "NVIDIA" in hardware:
            if not self._has_warp_optimization(kernel_code):
                suggestions.append(Suggestion(
                    severity=SeverityLevel.MEDIUM,
                    category="compute",
                    title="Warp Optimization Missing",
                    message="Use warp-level primitives for better efficiency on NVIDIA hardware",
                    code_snippet=self._generate_warp_snippet()
                ))
        
        elif "AMD" in hardware:
            if not self._has_wavefront_optimization(kernel_code):
                suggestions.append(Suggestion(
                    severity=SeverityLevel.MEDIUM,
                    category="compute",
                    title="Wavefront Optimization Missing",
                    message="Use work group functions for better efficiency on AMD hardware",
                    code_snippet=self._generate_wavefront_snippet()
                ))
        
        # Check for vectorization opportunities
        if not self._has_vectorization(kernel_code, backend):
            suggestions.append(Suggestion(
                severity=SeverityLevel.MEDIUM,
                category="compute",
                title="Vectorization Opportunity",
                message="Use vectorized operations for better performance",
                code_snippet=self._generate_vectorization_snippet(backend)
            ))
        
        return suggestions
    
    async def _generate_cache_suggestions(self, kernel_code: str, hardware: str, backend: str, hardware_spec: Dict) -> List[Suggestion]:
        """Generate cache optimization suggestions"""
        suggestions = []
        
        # Check for L1 cache optimization
        if not self._has_l1_cache_optimization(kernel_code, backend):
            suggestions.append(Suggestion(
                severity=SeverityLevel.LOW,
                category="memory",
                title="L1 Cache Optimization",
                message="Optimize data access patterns for better L1 cache utilization",
                code_snippet=self._generate_cache_snippet(backend)
            ))
        
        # Check for L2 cache optimization
        if not self._has_l2_cache_optimization(kernel_code, backend):
            suggestions.append(Suggestion(
                severity=SeverityLevel.LOW,
                category="memory",
                title="L2 Cache Optimization",
                message="Consider data layout optimization for better L2 cache utilization",
                code_snippet=self._generate_l2_cache_snippet(backend)
            ))
        
        return suggestions
    
    def _has_memory_coalescing(self, kernel_code: str, backend: str) -> bool:
        """Check if memory access is coalesced"""
        if backend.upper() == "CUDA":
            coalescing_patterns = [
                r'threadIdx\.x\s*\+\s*blockIdx\.x\s*\*\s*blockDim\.x',
                r'threadIdx\.y\s*\+\s*blockIdx\.y\s*\*\s*blockDim\.y'
            ]
        elif backend.upper() == "OPENCL":
            coalescing_patterns = [
                r'get_global_id\(0\)',
                r'get_global_id\(1\)'
            ]
        else:
            return True  # Assume coalesced for other backends
        
        return any(re.search(pattern, kernel_code, re.IGNORECASE) for pattern in coalescing_patterns)
    
    def _has_warp_optimization(self, kernel_code: str) -> bool:
        """Check if warp-level primitives are used"""
        warp_patterns = [
            r'__shfl_',
            r'__ballot',
            r'__any',
            r'__all'
        ]
        return any(re.search(pattern, kernel_code, re.IGNORECASE) for pattern in warp_patterns)
    
    def _has_wavefront_optimization(self, kernel_code: str) -> bool:
        """Check if work group functions are used"""
        wavefront_patterns = [
            r'work_group_',
            r'get_local_id',
            r'get_local_size'
        ]
        return any(re.search(pattern, kernel_code, re.IGNORECASE) for pattern in wavefront_patterns)
    
    def _has_vectorization(self, kernel_code: str, backend: str) -> bool:
        """Check if vectorized operations are used"""
        vector_patterns = [
            r'float4|double2|int4',
            r'__m128|__m256|__m512',
            r'vload4|vstore4'
        ]
        return any(re.search(pattern, kernel_code, re.IGNORECASE) for pattern in vector_patterns)
    
    def _has_l1_cache_optimization(self, kernel_code: str, backend: str) -> bool:
        """Check if L1 cache optimization is present"""
        # Simple heuristic: check for data reuse patterns
        return "shared" in kernel_code.lower() or "local" in kernel_code.lower()
    
    def _has_l2_cache_optimization(self, kernel_code: str, backend: str) -> bool:
        """Check if L2 cache optimization is present"""
        # Simple heuristic: check for tiling patterns
        return "tile" in kernel_code.lower() or "block" in kernel_code.lower()
    
    def _generate_shared_memory_snippet(self, hardware: str, backend: str, max_size: int) -> str:
        """Generate shared memory optimization snippet"""
        if backend.upper() == "CUDA":
            return f"""__shared__ float tile[16][16];  // Optimal for {hardware}
// Use up to {max_size} bytes of shared memory
int tx = threadIdx.x, ty = threadIdx.y;
int bx = blockIdx.x, by = blockIdx.y;

// Load data with coalesced access
int row = by * 16 + ty;
int col = bx * 16 + tx;
if (row < M && col < N) {{
    tile[ty][tx] = A[row * N + col];
}}
__syncthreads();"""
        elif backend.upper() == "OPENCL":
            return f"""__local float tile[16][16];  // Optimal for {hardware}
// Use up to {max_size} bytes of local memory
int tx = get_local_id(0), ty = get_local_id(1);
int bx = get_group_id(0), by = get_group_id(1);

// Load data with coalesced access
int row = by * 16 + ty;
int col = bx * 16 + tx;
if (row < M && col < N) {{
    tile[ty][tx] = A[row * N + col];
}}
barrier(CLK_LOCAL_MEM_FENCE);"""
        else:
            return f"// Optimize shared memory usage for {hardware} (max {max_size} bytes)"
    
    def _generate_coalescing_snippet(self, backend: str) -> str:
        """Generate memory coalescing snippet"""
        if backend.upper() == "CUDA":
            return """// Memory coalescing for CUDA
int idx = threadIdx.x + blockIdx.x * blockDim.x;
if (idx < size) {
    float val = data[idx];  // Coalesced access
    // Process val
    result[idx] = val;
}"""
        elif backend.upper() == "OPENCL":
            return """// Memory coalescing for OpenCL
int idx = get_global_id(0);
if (idx < size) {
    float val = data[idx];  // Coalesced access
    // Process val
    result[idx] = val;
}"""
        else:
            return "// Implement memory coalescing for better bandwidth"
    
    def _generate_warp_snippet(self) -> str:
        """Generate warp optimization snippet"""
        return """// Warp-level optimization for NVIDIA
__global__ void warp_optimized_kernel(float* data, float* result, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    if (idx < n) {
        float val = data[idx];
        
        // Warp-level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        
        if (tid == 0) {
            result[bid] = val;
        }
    }
}"""
    
    def _generate_wavefront_snippet(self) -> str:
        """Generate wavefront optimization snippet"""
        return """// Wavefront optimization for AMD
__kernel void wavefront_optimized_kernel(__global float* data, __global float* result, int n) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int gid_local = get_group_id(0);
    
    if (gid < n) {
        float val = data[gid];
        
        // Work group reduction
        val = work_group_reduce_add(val);
        
        if (lid == 0) {
            result[gid_local] = val;
        }
    }
}"""
    
    def _generate_vectorization_snippet(self, backend: str) -> str:
        """Generate vectorization snippet"""
        if backend.upper() == "CUDA":
            return """// Vectorized operations for CUDA
__global__ void vectorized_kernel(float4* data, float4* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 val = data[idx];  // Load 4 floats at once
        val.x *= 2.0f; val.y *= 2.0f; val.z *= 2.0f; val.w *= 2.0f;
        result[idx] = val;  // Store 4 floats at once
    }
}"""
        elif backend.upper() == "OPENCL":
            return """// Vectorized operations for OpenCL
__kernel void vectorized_kernel(__global float4* data, __global float4* result, int n) {
    int idx = get_global_id(0);
    if (idx < n) {
        float4 val = data[idx];  // Load 4 floats at once
        val *= 2.0f;  // Vectorized multiplication
        result[idx] = val;  // Store 4 floats at once
    }
}"""
        else:
            return "// Use vectorized data types for better performance"
    
    def _generate_cache_snippet(self, backend: str) -> str:
        """Generate cache optimization snippet"""
        return """// Cache optimization
// 1. Use shared/local memory for frequently accessed data
// 2. Implement tiling for better cache utilization
// 3. Minimize memory access patterns that cause cache misses
// 4. Use data prefetching where possible"""
    
    def _generate_l2_cache_snippet(self, backend: str) -> str:
        """Generate L2 cache optimization snippet"""
        return """// L2 cache optimization
// 1. Implement proper data layout (AoS vs SoA)
// 2. Use tiling to fit working set in L2 cache
// 3. Minimize data movement between L1 and L2
// 4. Consider data compression for large datasets"""

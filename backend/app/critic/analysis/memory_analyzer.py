"""
Memory usage analyzer for kernel analysis
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from ..critic_models import PerformanceMetrics, Suggestion, SeverityLevel
from ..parsers.parser_factory import ParserFactory


class MemoryAnalyzer:
    """Analyze memory usage patterns and shared memory utilization"""
    
    def __init__(self):
        self.name = "memory_analyzer"
        self.description = "Analyze memory usage patterns and shared memory utilization"
        self.parser_factory = ParserFactory()
        
        # Memory access patterns
        self.memory_patterns = {
            'global_load': r'(\w+)\s*\[([^\]]+)\]',  # Global memory access
            'shared_load': r'__shared__\s+\w+\s+(\w+)\s*\[([^\]]+)\]',  # Shared memory declaration
            'constant_load': r'__constant__\s+\w+\s+(\w+)',  # Constant memory
            'local_load': r'__local\s+\w+\s+(\w+)',  # Local memory (OpenCL)
            'register_load': r'(\w+)\s*=\s*[^=]+',  # Register usage
        }
        
        # Data type sizes (in bytes)
        self.type_sizes = {
            'char': 1, 'uchar': 1, 'schar': 1,
            'short': 2, 'ushort': 2,
            'int': 4, 'uint': 4, 'float': 4,
            'long': 8, 'ulong': 8, 'double': 8,
            'half': 2, 'float16': 2,
            'float2': 8, 'float4': 16, 'double2': 16,
        }
    
    async def analyze_memory(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Analyze memory usage patterns"""
        # Parse the kernel code
        parsed_kernel = await self.parser_factory.parse(kernel_code)
        if not parsed_kernel:
            return {
                "shared_memory_bytes": 0,
                "global_memory_accesses": 0,
                "memory_efficiency": 0.0,
                "suggestions": ["Failed to parse kernel code"]
            }
        
        # Analyze shared memory usage
        shared_mem_analysis = await self._analyze_shared_memory(parsed_kernel, backend)
        
        # Analyze global memory access patterns
        global_mem_analysis = await self._analyze_global_memory(parsed_kernel, backend)
        
        # Analyze memory efficiency
        efficiency_analysis = await self._analyze_memory_efficiency(parsed_kernel, backend)
        
        # Generate suggestions
        suggestions = self._generate_memory_suggestions(parsed_kernel, shared_mem_analysis, global_mem_analysis, hardware)
        
        return {
            "shared_memory_bytes": shared_mem_analysis["total_bytes"],
            "shared_memory_variables": shared_mem_analysis["variables"],
            "global_memory_accesses": global_mem_analysis["access_count"],
            "memory_access_patterns": global_mem_analysis["patterns"],
            "memory_efficiency": efficiency_analysis["efficiency_score"],
            "coalescing_score": efficiency_analysis["coalescing_score"],
            "bank_conflicts": shared_mem_analysis["bank_conflicts"],
            "suggestions": suggestions
        }
    
    async def _analyze_shared_memory(self, parsed_kernel, backend: str) -> Dict[str, Any]:
        """Analyze shared memory usage"""
        shared_vars = []
        total_bytes = 0
        bank_conflicts = 0
        
        for var in parsed_kernel.variables:
            if var.is_shared:
                # Calculate size
                var_size = self._calculate_variable_size(var)
                total_bytes += var_size
                
                shared_vars.append({
                    "name": var.name,
                    "type": var.type,
                    "size_bytes": var_size,
                    "array_size": var.array_size
                })
                
                # Check for potential bank conflicts
                if var.array_size and self._has_bank_conflict(var):
                    bank_conflicts += 1
        
        # Check if shared memory usage exceeds limits
        max_shared_mem = self._get_max_shared_memory(backend)
        usage_percentage = (total_bytes / max_shared_mem) * 100 if max_shared_mem > 0 else 0
        
        return {
            "total_bytes": total_bytes,
            "variables": shared_vars,
            "usage_percentage": usage_percentage,
            "bank_conflicts": bank_conflicts,
            "max_shared_memory": max_shared_mem
        }
    
    async def _analyze_global_memory(self, parsed_kernel, backend: str) -> Dict[str, Any]:
        """Analyze global memory access patterns"""
        access_count = 0
        patterns = {
            "coalesced": 0,
            "strided": 0,
            "random": 0
        }
        
        for access in parsed_kernel.memory_accesses:
            if access.is_global:
                access_count += 1
                
                # Analyze access pattern
                pattern = self._classify_access_pattern(access, parsed_kernel)
                patterns[pattern] += 1
        
        return {
            "access_count": access_count,
            "patterns": patterns,
            "coalescing_ratio": patterns["coalesced"] / max(1, access_count)
        }
    
    async def _analyze_memory_efficiency(self, parsed_kernel, backend: str) -> Dict[str, Any]:
        """Analyze overall memory efficiency"""
        # Calculate memory efficiency score
        efficiency_score = 0.0
        coalescing_score = 0.0
        
        # Check for vectorized loads/stores
        vectorized_ops = len(re.findall(r'float4|double2|int4', parsed_kernel.raw_code))
        if vectorized_ops > 0:
            efficiency_score += 20
        
        # Check for shared memory usage
        shared_mem_vars = len([v for v in parsed_kernel.variables if v.is_shared])
        if shared_mem_vars > 0:
            efficiency_score += 15
        
        # Check for constant memory usage
        const_mem_vars = len([v for v in parsed_kernel.variables if 'const' in v.type.lower()])
        if const_mem_vars > 0:
            efficiency_score += 10
        
        # Check for proper memory access patterns
        coalesced_accesses = len(re.findall(r'threadIdx\.x\s*\+', parsed_kernel.raw_code))
        if coalesced_accesses > 0:
            coalescing_score = min(100.0, coalesced_accesses * 10)
            efficiency_score += coalescing_score * 0.3
        
        return {
            "efficiency_score": min(100.0, efficiency_score),
            "coalescing_score": coalescing_score
        }
    
    def _calculate_variable_size(self, var) -> int:
        """Calculate size of a variable in bytes"""
        # Get base type size
        base_type = var.type.split('*')[0].strip()  # Remove pointer indicators
        base_size = self.type_sizes.get(base_type, 4)  # Default to 4 bytes
        
        # Calculate array size
        if var.array_size:
            try:
                # Try to evaluate array size
                array_size = eval(var.array_size)
                return base_size * array_size
            except:
                # If evaluation fails, estimate
                return base_size * 100  # Conservative estimate
        else:
            return base_size
    
    def _has_bank_conflict(self, var) -> bool:
        """Check if variable has potential bank conflicts"""
        if not var.array_size:
            return False
        
        try:
            array_size = eval(var.array_size)
            # Check if array size is a multiple of 32 (common bank conflict pattern)
            return array_size % 32 == 0 and array_size > 32
        except:
            return False
    
    def _get_max_shared_memory(self, backend: str) -> int:
        """Get maximum shared memory for the backend"""
        limits = {
            "CUDA": 48 * 1024,  # 48KB typical
            "OPENCL": 32 * 1024,  # 32KB typical
            "TRITON": 64 * 1024,  # 64KB typical
        }
        return limits.get(backend.upper(), 32 * 1024)
    
    def _classify_access_pattern(self, access, parsed_kernel) -> str:
        """Classify memory access pattern"""
        if not access.indexing_pattern:
            return "random"
        
        index_expr = access.indexing_pattern
        
        # Check for coalesced access (consecutive thread indices)
        if re.search(r'threadIdx\.x\s*\+', index_expr):
            return "coalesced"
        
        # Check for strided access
        if re.search(r'threadIdx\.x\s*\*\s*\d+', index_expr):
            return "strided"
        
        # Check for random access
        if re.search(r'blockIdx|gridIdx', index_expr):
            return "random"
        
        return "random"
    
    def _generate_memory_suggestions(self, parsed_kernel, shared_mem_analysis: Dict, global_mem_analysis: Dict, hardware: str) -> List[str]:
        """Generate memory optimization suggestions"""
        suggestions = []
        
        # Shared memory suggestions
        if shared_mem_analysis["usage_percentage"] > 80:
            suggestions.append("Shared memory usage is high - consider optimizing or using global memory")
        elif shared_mem_analysis["usage_percentage"] < 10:
            suggestions.append("Consider using more shared memory to reduce global memory access")
        
        if shared_mem_analysis["bank_conflicts"] > 0:
            suggestions.append("Potential bank conflicts detected - consider padding arrays")
        
        # Global memory suggestions
        coalescing_ratio = global_mem_analysis["coalescing_ratio"]
        if coalescing_ratio < 0.5:
            suggestions.append("Low memory coalescing - consider restructuring memory access patterns")
        
        # Vectorization suggestions
        if not re.search(r'float4|double2|int4', parsed_kernel.raw_code):
            suggestions.append("Consider using vectorized loads/stores (float4, double2) for better memory bandwidth")
        
        # Hardware-specific suggestions
        if "NVIDIA" in hardware:
            if not re.search(r'__ldg|__ldcg', parsed_kernel.raw_code):
                suggestions.append("Consider using read-only cache (__ldg) for read-only global memory")
        
        return suggestions

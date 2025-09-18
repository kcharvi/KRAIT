"""
Vectorization checker for kernel analysis
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from ..critic_models import PerformanceMetrics, Suggestion, SeverityLevel
from ..parsers.parser_factory import ParserFactory


class VectorizationChecker:
    """Check for vectorization opportunities and patterns"""
    
    def __init__(self):
        self.name = "vectorization_checker"
        self.description = "Check for vectorization opportunities and patterns"
        self.parser_factory = ParserFactory()
        
        # Vectorization patterns
        self.vector_patterns = {
            # CUDA vector types
            'cuda_float2': r'float2\s+\w+',
            'cuda_float4': r'float4\s+\w+',
            'cuda_double2': r'double2\s+\w+',
            'cuda_int2': r'int2\s+\w+',
            'cuda_int4': r'int4\s+\w+',
            
            # OpenCL vector types
            'opencl_float2': r'float2\s+\w+',
            'opencl_float4': r'float4\s+\w+',
            'opencl_float8': r'float8\s+\w+',
            'opencl_float16': r'float16\s+\w+',
            
            # SIMD intrinsics
            'simd_load': r'__m128|__m256|__m512',
            'simd_add': r'_mm_add_ps|_mm_add_pd',
            'simd_mul': r'_mm_mul_ps|_mm_mul_pd',
            'simd_fma': r'_mm_fmadd_ps|_mm_fmadd_pd',
            
            # Triton vectorization
            'triton_vector': r'tl\.load|tl\.store',
            'triton_dot': r'tl\.dot',
            
            # Loop vectorization hints
            'pragma_vectorize': r'#pragma\s+vectorize|#pragma\s+simd',
            'pragma_unroll': r'#pragma\s+unroll',
        }
        
        # Vector operation patterns
        self.vector_ops = {
            'vector_load': r'(\w+)\s*=\s*(\w+)\[(\w+)\]',  # Vector load
            'vector_store': r'(\w+)\[(\w+)\]\s*=\s*(\w+)',  # Vector store
            'vector_add': r'(\w+)\s*\+\s*(\w+)',  # Vector addition
            'vector_mul': r'(\w+)\s*\*\s*(\w+)',  # Vector multiplication
        }
    
    async def check_vectorization(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Check for vectorization opportunities"""
        # Parse the kernel code
        parsed_kernel = await self.parser_factory.parse(kernel_code)
        if not parsed_kernel:
            return {
                "vectorization_detected": False,
                "vectorization_score": 0.0,
                "suggestions": ["Failed to parse kernel code"]
            }
        
        # Detect vectorization patterns
        vectorization_results = await self._detect_vectorization_patterns(parsed_kernel, backend)
        
        # Analyze vectorization opportunities
        opportunity_analysis = await self._analyze_vectorization_opportunities(parsed_kernel, backend)
        
        # Calculate vectorization score
        vectorization_score = self._calculate_vectorization_score(vectorization_results, opportunity_analysis)
        
        # Generate suggestions
        suggestions = self._generate_vectorization_suggestions(parsed_kernel, vectorization_results, opportunity_analysis, hardware, backend)
        
        return {
            "vectorization_detected": vectorization_results["detected"],
            "vectorization_patterns": vectorization_results["patterns"],
            "vectorization_score": vectorization_score,
            "opportunities": opportunity_analysis["opportunities"],
            "efficiency_gain": opportunity_analysis["efficiency_gain"],
            "suggestions": suggestions
        }
    
    async def _detect_vectorization_patterns(self, parsed_kernel, backend: str) -> Dict[str, Any]:
        """Detect existing vectorization patterns"""
        detected_patterns = []
        detected = False
        
        # Check for vector types
        for pattern_name, pattern in self.vector_patterns.items():
            if re.search(pattern, parsed_kernel.raw_code, re.IGNORECASE | re.MULTILINE):
                detected_patterns.append(pattern_name)
                detected = True
        
        # Check for vector operations
        vector_ops = []
        for op_name, pattern in self.vector_ops.items():
            matches = re.findall(pattern, parsed_kernel.raw_code, re.IGNORECASE | re.MULTILINE)
            if matches:
                vector_ops.append({
                    "operation": op_name,
                    "count": len(matches),
                    "matches": matches
                })
                detected = True
        
        return {
            "detected": detected,
            "patterns": detected_patterns,
            "vector_operations": vector_ops
        }
    
    async def _analyze_vectorization_opportunities(self, parsed_kernel, backend: str) -> Dict[str, Any]:
        """Analyze vectorization opportunities"""
        opportunities = []
        efficiency_gain = 0.0
        
        # Check for scalar operations that could be vectorized
        scalar_ops = self._find_scalar_operations(parsed_kernel)
        
        for op in scalar_ops:
            if self._can_be_vectorized(op, backend):
                opportunities.append({
                    "line": op["line_number"],
                    "operation": op["operation"],
                    "type": op["type"],
                    "potential_gain": self._estimate_vectorization_gain(op, backend)
                })
                efficiency_gain += op["potential_gain"]
        
        # Check for loop vectorization opportunities
        loop_opportunities = self._find_loop_vectorization_opportunities(parsed_kernel, backend)
        opportunities.extend(loop_opportunities)
        
        return {
            "opportunities": opportunities,
            "efficiency_gain": efficiency_gain,
            "total_opportunities": len(opportunities)
        }
    
    def _find_scalar_operations(self, parsed_kernel) -> List[Dict[str, Any]]:
        """Find scalar operations that could be vectorized"""
        scalar_ops = []
        
        for i, line in enumerate(parsed_kernel.lines):
            line = line.strip()
            
            # Skip comments and preprocessor directives
            if line.startswith('//') or line.startswith('/*') or line.startswith('#'):
                continue
            
            # Check for arithmetic operations
            if re.search(r'(\w+)\s*[+\-*/]\s*(\w+)', line):
                scalar_ops.append({
                    "line_number": i + 1,
                    "operation": line,
                    "type": "arithmetic",
                    "potential_gain": 2.0  # 2x speedup potential
                })
            
            # Check for memory operations
            if re.search(r'(\w+)\s*=\s*(\w+)\[(\w+)\]', line):
                scalar_ops.append({
                    "line_number": i + 1,
                    "operation": line,
                    "type": "memory_load",
                    "potential_gain": 4.0  # 4x speedup potential
                })
            
            if re.search(r'(\w+)\[(\w+)\]\s*=\s*(\w+)', line):
                scalar_ops.append({
                    "line_number": i + 1,
                    "operation": line,
                    "type": "memory_store",
                    "potential_gain": 4.0  # 4x speedup potential
                })
        
        return scalar_ops
    
    def _can_be_vectorized(self, op: Dict[str, Any], backend: str) -> bool:
        """Check if an operation can be vectorized"""
        # Check if already vectorized
        if any(vector_type in op["operation"] for vector_type in ['float2', 'float4', 'double2', 'int2', 'int4']):
            return False
        
        # Check for vectorization-friendly operations
        if op["type"] in ["arithmetic", "memory_load", "memory_store"]:
            return True
        
        return False
    
    def _estimate_vectorization_gain(self, op: Dict[str, Any], backend: str) -> float:
        """Estimate potential speedup from vectorization"""
        base_gain = op.get("potential_gain", 1.0)
        
        # Adjust based on backend capabilities
        backend_multipliers = {
            "CUDA": 1.0,
            "OPENCL": 0.8,
            "TRITON": 1.2,
            "C++": 0.6,
        }
        
        multiplier = backend_multipliers.get(backend.upper(), 0.5)
        return base_gain * multiplier
    
    def _find_loop_vectorization_opportunities(self, parsed_kernel, backend: str) -> List[Dict[str, Any]]:
        """Find loop vectorization opportunities"""
        opportunities = []
        
        for loop in parsed_kernel.loops:
            if loop.type == "for":
                # Check if loop can be vectorized
                if self._is_loop_vectorizable(loop, parsed_kernel, backend):
                    opportunities.append({
                        "line": loop.line_number,
                        "operation": f"for loop with {loop.variable}",
                        "type": "loop_vectorization",
                        "potential_gain": self._estimate_loop_vectorization_gain(loop, backend)
                    })
        
        return opportunities
    
    def _is_loop_vectorizable(self, loop, parsed_kernel, backend: str) -> bool:
        """Check if a loop can be vectorized"""
        # Check for simple increment patterns
        if loop.step and loop.step.strip() in ['1', '++', '+1']:
            return True
        
        # Check for stride patterns
        if loop.step and re.search(r'\+=\s*\d+', loop.step):
            stride = self._extract_stride(loop.step)
            if stride and stride in [2, 4, 8, 16]:  # Common vectorization strides
                return True
        
        return False
    
    def _extract_stride(self, step_expr: str) -> Optional[int]:
        """Extract stride from step expression"""
        match = re.search(r'\+=\s*(\d+)', step_expr)
        if match:
            return int(match.group(1))
        return None
    
    def _estimate_loop_vectorization_gain(self, loop, backend: str) -> float:
        """Estimate vectorization gain for a loop"""
        base_gain = 4.0  # Typical vectorization speedup
        
        # Adjust based on backend
        backend_multipliers = {
            "CUDA": 1.0,
            "OPENCL": 0.8,
            "TRITON": 1.2,
            "C++": 0.6,
        }
        
        multiplier = backend_multipliers.get(backend.upper(), 0.5)
        return base_gain * multiplier
    
    def _calculate_vectorization_score(self, vectorization_results: Dict, opportunity_analysis: Dict) -> float:
        """Calculate overall vectorization score"""
        score = 0.0
        
        # Base score from existing vectorization
        if vectorization_results["detected"]:
            score += 50.0  # Base score for having vectorization
        
        # Bonus for multiple vectorization patterns
        pattern_count = len(vectorization_results["patterns"])
        score += min(30.0, pattern_count * 5.0)
        
        # Bonus for vector operations
        vector_op_count = len(vectorization_results["vector_operations"])
        score += min(20.0, vector_op_count * 3.0)
        
        # Penalty for missed opportunities
        missed_opportunities = opportunity_analysis["total_opportunities"]
        if missed_opportunities > 0:
            score -= min(40.0, missed_opportunities * 2.0)
        
        return max(0.0, min(100.0, score))
    
    def _generate_vectorization_suggestions(self, parsed_kernel, vectorization_results: Dict, opportunity_analysis: Dict, hardware: str, backend: str) -> List[str]:
        """Generate vectorization suggestions"""
        suggestions = []
        
        if not vectorization_results["detected"]:
            suggestions.append("Consider using vectorized data types for better performance")
            
            if backend.upper() == "CUDA":
                suggestions.append("Use float4 or double2 for vectorized operations")
            elif backend.upper() == "OPENCL":
                suggestions.append("Use float4, float8, or float16 for vectorized operations")
            elif backend.upper() == "TRITON":
                suggestions.append("Use Triton's built-in vectorization with proper block sizes")
        
        # Suggest specific vectorization opportunities
        for opportunity in opportunity_analysis["opportunities"][:5]:  # Limit to top 5
            if opportunity["type"] == "memory_load":
                suggestions.append(f"Consider vectorized load at line {opportunity['line']}: {opportunity['operation']}")
            elif opportunity["type"] == "arithmetic":
                suggestions.append(f"Consider vectorized arithmetic at line {opportunity['line']}: {opportunity['operation']}")
        
        # Hardware-specific suggestions
        if "NVIDIA" in hardware:
            if not re.search(r'float4|double2', parsed_kernel.raw_code):
                suggestions.append("Use float4 or double2 for better memory bandwidth utilization on NVIDIA GPUs")
        
        elif "AMD" in hardware:
            if not re.search(r'float4|float8', parsed_kernel.raw_code):
                suggestions.append("Use float4 or float8 for better performance on AMD GPUs")
        
        # Loop vectorization suggestions
        if not re.search(r'#pragma\s+vectorize|#pragma\s+simd', parsed_kernel.raw_code):
            suggestions.append("Consider adding vectorization pragmas to loops")
        
        return suggestions

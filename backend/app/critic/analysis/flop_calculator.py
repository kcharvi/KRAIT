"""
FLOP (Floating Point Operations) calculator for kernel analysis
"""

import re
from typing import Dict, Any, List
from ..parsers.parser_factory import ParserFactory


class FLOPCalculator:
    """Calculate theoretical FLOPs for kernel operations"""
    
    def __init__(self):
        self.name = "flop_calculator"
        self.description = "Calculate theoretical floating point operations"
        self.parser_factory = ParserFactory()
        
        # FLOP patterns for different operations
        self.flop_patterns = {
            # Basic arithmetic operations
            'add': r'(\w+)\s*\+\s*(\w+)',
            'sub': r'(\w+)\s*-\s*(\w+)',
            'mul': r'(\w+)\s*\*\s*(\w+)',
            'div': r'(\w+)\s*/\s*(\w+)',
            'fma': r'fma\s*\([^)]+\)',  # Fused multiply-add
            
            # Math functions
            'sqrt': r'sqrt\s*\([^)]+\)',
            'exp': r'exp\s*\([^)]+\)',
            'log': r'log\s*\([^)]+\)',
            'sin': r'sin\s*\([^)]+\)',
            'cos': r'cos\s*\([^)]+\)',
            'tan': r'tan\s*\([^)]+\)',
            
            # Matrix operations
            'matmul': r'(\w+)\s*@\s*(\w+)',  # Python matrix multiplication
            'dot': r'tl\.dot\s*\([^)]+\)',  # Triton dot product
            'gemm': r'cublasGemm|gemm\s*\([^)]+\)',  # BLAS GEMM
            
            # Reduction operations
            'sum': r'sum\s*\([^)]+\)|tl\.sum\s*\([^)]+\)',
            'max': r'max\s*\([^)]+\)|tl\.max\s*\([^)]+\)',
            'min': r'min\s*\([^)]+\)|tl\.min\s*\([^)]+\)',
        }
        
        # FLOP costs for different operations
        self.flop_costs = {
            'add': 1,
            'sub': 1,
            'mul': 1,
            'div': 1,
            'fma': 2,  # Fused multiply-add counts as 2 FLOPs
            'sqrt': 10,  # Approximate cost
            'exp': 20,   # Approximate cost
            'log': 20,   # Approximate cost
            'sin': 15,   # Approximate cost
            'cos': 15,   # Approximate cost
            'tan': 20,   # Approximate cost
            'matmul': 2,  # Per element: 2 FLOPs for matrix multiplication
            'dot': 2,     # Per element: 2 FLOPs for dot product
            'gemm': 2,    # Per element: 2 FLOPs for GEMM
            'sum': 1,     # Per element: 1 FLOP for addition
            'max': 1,     # Per element: 1 FLOP for comparison
            'min': 1,     # Per element: 1 FLOP for comparison
        }
    
    async def calculate_flops(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Calculate FLOPs for the given kernel code"""
        # Parse the kernel code
        parsed_kernel = await self.parser_factory.parse(kernel_code)
        if not parsed_kernel:
            return {
                "total_flops": 0,
                "operation_breakdown": {},
                "efficiency_score": 0,
                "suggestions": ["Failed to parse kernel code"]
            }
        
        # Calculate FLOPs by operation type
        operation_counts = {}
        total_flops = 0
        
        for op_type, pattern in self.flop_patterns.items():
            matches = re.findall(pattern, kernel_code, re.IGNORECASE | re.MULTILINE)
            count = len(matches)
            if count > 0:
                operation_counts[op_type] = count
                total_flops += count * self.flop_costs[op_type]
        
        # Calculate loop-based FLOPs
        loop_flops = await self._calculate_loop_flops(parsed_kernel, operation_counts)
        total_flops += loop_flops
        
        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(parsed_kernel, total_flops, hardware)
        
        # Generate suggestions
        suggestions = self._generate_flop_suggestions(parsed_kernel, operation_counts, total_flops, hardware)
        
        return {
            "total_flops": total_flops,
            "operation_breakdown": operation_counts,
            "loop_flops": loop_flops,
            "efficiency_score": efficiency_score,
            "suggestions": suggestions,
            "hardware_utilization": self._estimate_hardware_utilization(total_flops, hardware)
        }
    
    async def _calculate_loop_flops(self, parsed_kernel, operation_counts: Dict[str, int]) -> int:
        """Calculate FLOPs based on loop structures"""
        loop_flops = 0
        
        for loop in parsed_kernel.loops:
            if loop.type == "for":
                # Estimate loop iterations (simplified)
                iterations = self._estimate_loop_iterations(loop)
                
                # Count operations inside the loop
                loop_ops = 0
                for op_type, count in operation_counts.items():
                    if self._is_operation_in_loop(op_type, loop, parsed_kernel):
                        loop_ops += count * self.flop_costs[op_type]
                
                loop_flops += iterations * loop_ops
        
        return loop_flops
    
    def _estimate_loop_iterations(self, loop) -> int:
        """Estimate number of loop iterations (simplified)"""
        # This is a simplified estimation
        # In practice, you'd need more sophisticated analysis
        
        if "blockDim" in loop.end or "gridDim" in loop.end:
            return 1024  # Typical block size
        elif "N" in loop.end or "M" in loop.end or "K" in loop.end:
            return 1000  # Typical matrix dimension
        elif loop.end.isdigit():
            return int(loop.end)
        else:
            return 100  # Default estimate
    
    def _is_operation_in_loop(self, op_type: str, loop, parsed_kernel) -> bool:
        """Check if an operation is inside a specific loop"""
        # Simplified check - in practice, you'd need proper scope analysis
        loop_start = loop.line_number
        loop_end = loop_start + 10  # Assume 10 lines for loop body
        
        # Check if operation appears in loop range
        for i in range(loop_start, min(loop_end, len(parsed_kernel.lines))):
            if i < len(parsed_kernel.lines):
                line = parsed_kernel.lines[i]
                if re.search(self.flop_patterns[op_type], line, re.IGNORECASE):
                    return True
        
        return False
    
    def _calculate_efficiency_score(self, parsed_kernel, total_flops: int, hardware: str) -> float:
        """Calculate efficiency score based on FLOP density and patterns"""
        lines_of_code = len([line for line in parsed_kernel.lines if line.strip() and not line.strip().startswith('//')])
        
        if lines_of_code == 0:
            return 0.0
        
        # FLOP density (FLOPs per line of code)
        flop_density = total_flops / lines_of_code
        
        # Base efficiency score
        efficiency = min(100.0, flop_density * 10)  # Scale factor
        
        # Bonus for vectorized operations
        if any("float4" in line or "double2" in line for line in parsed_kernel.lines):
            efficiency += 10
        
        # Bonus for fused operations
        if "fma" in parsed_kernel.raw_code.lower():
            efficiency += 5
        
        # Penalty for excessive branching
        branch_count = len(re.findall(r'if\s*\(|switch\s*\(', parsed_kernel.raw_code))
        if branch_count > lines_of_code * 0.3:  # More than 30% branches
            efficiency -= 15
        
        return max(0.0, min(100.0, efficiency))
    
    def _estimate_hardware_utilization(self, total_flops: int, hardware: str) -> Dict[str, float]:
        """Estimate hardware utilization based on FLOPs"""
        # Hardware specifications (approximate)
        hardware_specs = {
            "NVIDIA H100": {"peak_flops": 989e12, "memory_bandwidth": 3350e9},
            "NVIDIA A100": {"peak_flops": 312e12, "memory_bandwidth": 2039e9},
            "AMD MI300X": {"peak_flops": 1634e12, "memory_bandwidth": 5600e9},
            "CPU": {"peak_flops": 100e9, "memory_bandwidth": 100e9},
        }
        
        spec = hardware_specs.get(hardware, hardware_specs["CPU"])
        
        # This is a simplified calculation
        # In practice, you'd need runtime measurements
        theoretical_utilization = min(100.0, (total_flops / spec["peak_flops"]) * 100)
        
        return {
            "compute_utilization": theoretical_utilization,
            "memory_utilization": 0.0,  # Would need memory analysis
            "overall_efficiency": theoretical_utilization
        }
    
    def _generate_flop_suggestions(self, parsed_kernel, operation_counts: Dict[str, int], total_flops: int, hardware: str) -> List[str]:
        """Generate suggestions for FLOP optimization"""
        suggestions = []
        
        # Check for inefficient operations
        if operation_counts.get('div', 0) > operation_counts.get('mul', 0):
            suggestions.append("Consider replacing divisions with multiplications where possible")
        
        # Check for expensive math functions
        expensive_ops = ['exp', 'log', 'sin', 'cos', 'tan', 'sqrt']
        for op in expensive_ops:
            if operation_counts.get(op, 0) > 0:
                suggestions.append(f"Consider optimizing {op} operations - they are computationally expensive")
        
        # Check for vectorization opportunities
        if not any("float4" in line or "double2" in line for line in parsed_kernel.lines):
            if total_flops > 1000:
                suggestions.append("Consider vectorizing operations for better performance")
        
        # Check for FMA opportunities
        if operation_counts.get('mul', 0) > 0 and operation_counts.get('add', 0) > 0:
            if 'fma' not in parsed_kernel.raw_code.lower():
                suggestions.append("Consider using fused multiply-add (FMA) operations")
        
        # Hardware-specific suggestions
        if "NVIDIA" in hardware and "tensor" not in parsed_kernel.raw_code.lower():
            suggestions.append("Consider using Tensor Cores for matrix operations on NVIDIA hardware")
        
        return suggestions

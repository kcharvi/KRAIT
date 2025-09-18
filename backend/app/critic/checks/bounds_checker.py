"""
Bounds checking module for kernel analysis
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from ..critic_models import CorrectnessCheck, CheckStatus, SeverityLevel
from ..parsers.parser_factory import ParserFactory


class BoundsChecker:
    """Check for proper bounds checking in kernel code"""
    
    def __init__(self):
        self.name = "bounds_checker"
        self.description = "Checks for proper array bounds and index validation"
        self.parser_factory = ParserFactory()
    
    async def run_check(self, kernel_code: str, hardware: str, backend: str) -> CorrectnessCheck:
        """
        Check for bounds checking patterns in kernel code using AST analysis
        """
        if backend.upper() not in ["CUDA", "C++", "C", "TRITON", "OPENCL"]:
            return CorrectnessCheck(
                name=self.name,
                status=CheckStatus.SKIP,
                message=f"Bounds checking not applicable for {backend} backend"
            )
        
        # Parse the kernel code
        parsed_kernel = await self.parser_factory.parse(kernel_code)
        if not parsed_kernel:
            return CorrectnessCheck(
                name=self.name,
                status=CheckStatus.FAIL,
                message="Failed to parse kernel code for bounds analysis"
            )
        
        # Analyze bounds checking
        analysis_result = await self._analyze_bounds_checking(parsed_kernel, backend)
        
        return CorrectnessCheck(
            name=self.name,
            status=analysis_result["status"],
            message=analysis_result["message"],
            details=analysis_result["details"]
        )
    
    async def _analyze_bounds_checking(self, parsed_kernel, backend: str) -> Dict[str, Any]:
        """Analyze bounds checking using parsed kernel information"""
        issues = []
        suggestions = []
        patterns_found = []
        
        # Check for array accesses without bounds checking
        unsafe_accesses = []
        for access in parsed_kernel.memory_accesses:
            if not self._has_bounds_check(access, parsed_kernel, backend):
                unsafe_accesses.append(access)
        
        # Check for proper bounds checking patterns
        bounds_patterns = self._get_bounds_patterns(backend)
        for pattern_name, pattern in bounds_patterns.items():
            if re.search(pattern, parsed_kernel.raw_code, re.IGNORECASE | re.MULTILINE):
                patterns_found.append(pattern_name)
        
        # Analyze severity
        if unsafe_accesses:
            if len(unsafe_accesses) > 5:
                severity = CheckStatus.FAIL
                message = f"Critical: {len(unsafe_accesses)} unsafe memory accesses detected"
            else:
                severity = CheckStatus.WARNING
                message = f"Warning: {len(unsafe_accesses)} potentially unsafe memory accesses"
            
            issues.extend([f"Unsafe access to {acc.variable} at line {acc.line_number}" for acc in unsafe_accesses])
        else:
            if patterns_found:
                severity = CheckStatus.PASS
                message = f"Good bounds checking detected: {', '.join(patterns_found)}"
            else:
                severity = CheckStatus.WARNING
                message = "No explicit bounds checking patterns detected"
        
        # Generate suggestions
        suggestions = self._generate_bounds_suggestions(parsed_kernel, backend, unsafe_accesses)
        
        return {
            "status": severity,
            "message": message,
            "details": {
                "patterns_found": patterns_found,
                "unsafe_accesses": len(unsafe_accesses),
                "issues": issues,
                "suggestions": suggestions,
                "total_memory_accesses": len(parsed_kernel.memory_accesses)
            }
        }
    
    def _has_bounds_check(self, access: Any, parsed_kernel, backend: str) -> bool:
        """Check if a memory access has proper bounds checking"""
        # Look for bounds checking in the same function/scope
        access_line = access.line_number
        
        # Check lines before the access for bounds checking
        start_line = max(0, access_line - 10)
        end_line = min(len(parsed_kernel.lines), access_line + 1)
        
        context_lines = parsed_kernel.lines[start_line:end_line]
        context_code = '\n'.join(context_lines)
        
        # Check for common bounds checking patterns
        bounds_patterns = [
            r'if\s*\(\s*[^)]*<\s*[^)]*\)',  # if (index < size)
            r'if\s*\(\s*[^)]*>=\s*[^)]*\)',  # if (index >= 0)
            r'if\s*\(\s*[^)]*<=\s*[^)]*\)',  # if (index <= max)
        ]
        
        for pattern in bounds_patterns:
            if re.search(pattern, context_code, re.IGNORECASE):
                return True
        
        return False
    
    def _generate_bounds_suggestions(self, parsed_kernel, backend: str, unsafe_accesses: List) -> List[str]:
        """Generate specific suggestions for bounds checking improvements"""
        suggestions = []
        
        if backend.upper() == "CUDA":
            if any("threadIdx" in line for line in parsed_kernel.lines):
                suggestions.append("Add thread index bounds: if (threadIdx.x < width && threadIdx.y < height)")
            
            if any("blockIdx" in line for line in parsed_kernel.lines):
                suggestions.append("Add block index bounds: if (blockIdx.x < gridDim.x && blockIdx.y < gridDim.y)")
            
            if unsafe_accesses:
                suggestions.append("Add bounds checking before array access: if (index < array_size)")
        
        elif backend.upper() == "TRITON":
            suggestions.append("Use Triton's built-in bounds checking with proper block sizes")
            suggestions.append("Ensure block dimensions don't exceed tensor dimensions")
        
        elif backend.upper() == "OPENCL":
            suggestions.append("Use get_global_id() and get_global_size() for bounds checking")
            suggestions.append("Add work item bounds: if (get_global_id(0) < global_size)")
        
        return suggestions
    
    def _get_bounds_patterns(self, backend: str) -> Dict[str, str]:
        """Get bounds checking patterns for specific backend"""
        patterns = {}
        
        if backend.upper() == "CUDA":
            patterns = {
                "thread_bounds": r'if\s*\(\s*threadIdx\.(x|y|z)\s*<\s*\w+',
                "block_bounds": r'if\s*\(\s*blockIdx\.(x|y|z)\s*<\s*\w+',
                "global_bounds": r'if\s*\(\s*[^)]*idx[^)]*\s*<\s*\w+',
                "size_check": r'if\s*\(\s*[^)]*size[^)]*\s*[<>=]',
                "dimension_check": r'if\s*\(\s*[^)]*dim[^)]*\s*[<>=]',
                "early_return": r'if\s*\([^)]*\)\s*return\s*;',
            }
        elif backend.upper() in ["C++", "C"]:
            patterns = {
                "array_bounds": r'if\s*\(\s*[^)]*<\s*[^)]*size',
                "index_check": r'if\s*\(\s*[^)]*idx[^)]*\s*[<>=]',
                "null_check": r'if\s*\(\s*[^)]*!=\s*NULL',
                "range_check": r'if\s*\(\s*[^)]*>=.*&&.*<=',
            }
        
        return patterns
    
    def get_suggestions(self, kernel_code: str, backend: str) -> List[str]:
        """Get specific suggestions for improving bounds checking"""
        suggestions = []
        
        if backend.upper() == "CUDA":
            if "threadIdx" in kernel_code and "blockIdx" in kernel_code:
                if not re.search(r'if\s*\(\s*threadIdx\.(x|y|z)\s*<\s*\w+', kernel_code):
                    suggestions.append("Add thread index bounds checking: if (threadIdx.x < width)")
            
            if "blockIdx" in kernel_code:
                if not re.search(r'if\s*\(\s*blockIdx\.(x|y|z)\s*<\s*\w+', kernel_code):
                    suggestions.append("Add block index bounds checking: if (blockIdx.x < gridDim.x)")
        
        return suggestions

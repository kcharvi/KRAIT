"""
Memory safety checking module for kernel analysis
"""

import re
from typing import Dict, Any, List
from ..critic_models import CorrectnessCheck, CheckStatus


class MemorySafetyChecker:
    """Check for memory safety issues in kernel code"""
    
    def __init__(self):
        self.name = "memory_safety_checker"
        self.description = "Checks for memory safety issues and potential vulnerabilities"
    
    async def run_check(self, kernel_code: str, hardware: str, backend: str) -> CorrectnessCheck:
        """
        Check for memory safety issues in kernel code
        """
        safety_analysis = self._analyze_memory_safety(kernel_code, backend)
        
        if safety_analysis["issues"]:
            return CorrectnessCheck(
                name=self.name,
                status=CheckStatus.FAIL,
                message=f"Memory safety issues detected: {len(safety_analysis['issues'])} issues found",
                details=safety_analysis
            )
        elif safety_analysis["warnings"]:
            return CorrectnessCheck(
                name=self.name,
                status=CheckStatus.WARNING,
                message=f"Memory safety warnings: {len(safety_analysis['warnings'])} warnings found",
                details=safety_analysis
            )
        else:
            return CorrectnessCheck(
                name=self.name,
                status=CheckStatus.PASS,
                message="No obvious memory safety issues detected",
                details=safety_analysis
            )
    
    def _analyze_memory_safety(self, kernel_code: str, backend: str) -> Dict[str, Any]:
        """Analyze memory safety patterns in the code"""
        analysis = {
            "issues": [],
            "warnings": [],
            "safe_patterns": [],
            "unsafe_patterns": [],
            "pointer_usage": [],
            "array_access": []
        }
        
        # Check for unsafe pointer operations
        analysis["pointer_usage"] = self._check_pointer_usage(kernel_code, backend)
        
        # Check for array access patterns
        analysis["array_access"] = self._check_array_access(kernel_code, backend)
        
        # Check for buffer overflow patterns
        analysis["unsafe_patterns"] = self._check_unsafe_patterns(kernel_code, backend)
        
        # Check for safe patterns
        analysis["safe_patterns"] = self._check_safe_patterns(kernel_code, backend)
        
        # Compile issues and warnings
        analysis["issues"] = self._compile_issues(analysis)
        analysis["warnings"] = self._compile_warnings(analysis)
        
        return analysis
    
    def _check_pointer_usage(self, kernel_code: str, backend: str) -> List[Dict[str, Any]]:
        """Check for pointer usage patterns"""
        pointer_patterns = []
        
        # Look for pointer arithmetic
        pointer_arithmetic = re.finditer(r'(\w+)\s*\+\s*(\w+)', kernel_code)
        for match in pointer_arithmetic:
            pointer_patterns.append({
                "type": "pointer_arithmetic",
                "pattern": match.group(0),
                "line": kernel_code[:match.start()].count('\n') + 1,
                "severity": "warning"
            })
        
        # Look for pointer dereferencing
        pointer_deref = re.finditer(r'\*(\w+)', kernel_code)
        for match in pointer_deref:
            pointer_patterns.append({
                "type": "pointer_dereference",
                "pattern": match.group(0),
                "line": kernel_code[:match.start()].count('\n') + 1,
                "severity": "info"
            })
        
        return pointer_patterns
    
    def _check_array_access(self, kernel_code: str, backend: str) -> List[Dict[str, Any]]:
        """Check for array access patterns"""
        array_patterns = []
        
        # Look for array indexing
        array_indexing = re.finditer(r'(\w+)\[([^\]]+)\]', kernel_code)
        for match in array_indexing:
            array_name = match.group(1)
            index_expr = match.group(2)
            
            # Check if index expression looks safe
            is_safe = self._is_safe_index_expression(index_expr)
            
            array_patterns.append({
                "type": "array_access",
                "array": array_name,
                "index": index_expr,
                "line": kernel_code[:match.start()].count('\n') + 1,
                "is_safe": is_safe,
                "severity": "warning" if not is_safe else "info"
            })
        
        return array_patterns
    
    def _check_unsafe_patterns(self, kernel_code: str, backend: str) -> List[Dict[str, Any]]:
        """Check for unsafe memory patterns"""
        unsafe_patterns = []
        
        # Look for memcpy without bounds checking
        memcpy_pattern = r'memcpy\s*\([^)]*\)'
        if re.search(memcpy_pattern, kernel_code, re.IGNORECASE):
            unsafe_patterns.append({
                "type": "memcpy_without_bounds",
                "description": "memcpy without explicit bounds checking",
                "severity": "high"
            })
        
        # Look for strcpy/strcat
        strcpy_pattern = r'(strcpy|strcat|sprintf)\s*\('
        if re.search(strcpy_pattern, kernel_code, re.IGNORECASE):
            unsafe_patterns.append({
                "type": "unsafe_string_function",
                "description": "Use of unsafe string functions",
                "severity": "high"
            })
        
        # Look for uninitialized variables
        uninit_pattern = r'(\w+)\s*;.*\1\s*\['
        if re.search(uninit_pattern, kernel_code):
            unsafe_patterns.append({
                "type": "uninitialized_array",
                "description": "Possible uninitialized array usage",
                "severity": "medium"
            })
        
        return unsafe_patterns
    
    def _check_safe_patterns(self, kernel_code: str, backend: str) -> List[Dict[str, Any]]:
        """Check for safe memory patterns"""
        safe_patterns = []
        
        # Look for bounds checking before array access
        bounds_check = r'if\s*\([^)]*<\s*[^)]*\)\s*\{[^}]*\[[^\]]*\]'
        if re.search(bounds_check, kernel_code):
            safe_patterns.append({
                "type": "bounds_checking",
                "description": "Bounds checking before array access"
            })
        
        # Look for null pointer checks
        null_check = r'if\s*\(\s*[^)]*!=\s*NULL\s*\)'
        if re.search(null_check, kernel_code):
            safe_patterns.append({
                "type": "null_pointer_check",
                "description": "Null pointer checking"
            })
        
        # Look for safe string functions
        safe_str_pattern = r'(strncpy|strncat|snprintf)\s*\('
        if re.search(safe_str_pattern, kernel_code, re.IGNORECASE):
            safe_patterns.append({
                "type": "safe_string_function",
                "description": "Use of safe string functions"
            })
        
        return safe_patterns
    
    def _is_safe_index_expression(self, index_expr: str) -> bool:
        """Check if an index expression looks safe"""
        # Simple heuristic: check if it contains bounds checking
        safe_indicators = [
            'threadIdx', 'blockIdx', 'blockDim', 'gridDim',  # CUDA thread/block indices
            'size', 'length', 'count',  # Size variables
            'min', 'max',  # Min/max functions
        ]
        
        return any(indicator in index_expr for indicator in safe_indicators)
    
    def _compile_issues(self, analysis: Dict[str, Any]) -> List[str]:
        """Compile critical issues"""
        issues = []
        
        for pattern in analysis["unsafe_patterns"]:
            if pattern["severity"] == "high":
                issues.append(f"High severity: {pattern['description']}")
        
        for access in analysis["array_access"]:
            if not access["is_safe"]:
                issues.append(f"Unsafe array access: {access['array']}[{access['index']}]")
        
        return issues
    
    def _compile_warnings(self, analysis: Dict[str, Any]) -> List[str]:
        """Compile warnings"""
        warnings = []
        
        for pattern in analysis["unsafe_patterns"]:
            if pattern["severity"] == "medium":
                warnings.append(f"Medium severity: {pattern['description']}")
        
        for ptr in analysis["pointer_usage"]:
            if ptr["severity"] == "warning":
                warnings.append(f"Pointer arithmetic: {ptr['pattern']}")
        
        return warnings
    
    def get_suggestions(self, kernel_code: str, backend: str) -> List[str]:
        """Get suggestions for improving memory safety"""
        suggestions = []
        
        # Check for missing bounds checking
        if re.search(r'\[[^\]]+\]', kernel_code) and not re.search(r'if\s*\([^)]*<\s*[^)]*\)', kernel_code):
            suggestions.append("Add bounds checking before array access")
        
        # Check for unsafe string functions
        if re.search(r'(strcpy|strcat|sprintf)\s*\(', kernel_code, re.IGNORECASE):
            suggestions.append("Replace unsafe string functions with safe alternatives (strncpy, strncat, snprintf)")
        
        # Check for memcpy without bounds
        if re.search(r'memcpy\s*\([^)]*\)', kernel_code, re.IGNORECASE):
            suggestions.append("Add explicit bounds checking for memcpy operations")
        
        return suggestions

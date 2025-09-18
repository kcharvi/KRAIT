"""
Type safety checking module for kernel analysis
"""

import re
from typing import Dict, Any, List
from ..critic_models import CorrectnessCheck, CheckStatus


class TypeSafetyChecker:
    """Check for type safety issues in kernel code"""
    
    def __init__(self):
        self.name = "type_safety_checker"
        self.description = "Checks for type safety issues and potential type-related bugs"
    
    async def run_check(self, kernel_code: str, hardware: str, backend: str) -> CorrectnessCheck:
        """
        Check for type safety issues in kernel code with hardware/backend awareness
        """
        # Basic static analysis
        safety_analysis = self._analyze_type_safety(kernel_code, backend)
        
        # Hardware-specific type analysis
        hardware_analysis = self._analyze_hardware_specific_types(kernel_code, hardware, backend)
        safety_analysis.update(hardware_analysis)
        
        # Backend-specific type analysis
        backend_analysis = self._analyze_backend_specific_types(kernel_code, hardware, backend)
        safety_analysis.update(backend_analysis)
        
        # Note: LLM analysis is handled separately via dedicated endpoint
        
        if safety_analysis["critical_issues"]:
            return CorrectnessCheck(
                name=self.name,
                status=CheckStatus.FAIL,
                message=f"Critical type safety issues: {len(safety_analysis['critical_issues'])} found",
                details=safety_analysis
            )
        elif safety_analysis["warnings"]:
            return CorrectnessCheck(
                name=self.name,
                status=CheckStatus.WARNING,
                message=f"Type safety warnings: {len(safety_analysis['warnings'])} found",
                details=safety_analysis
            )
        else:
            return CorrectnessCheck(
                name=self.name,
                status=CheckStatus.PASS,
                message="No obvious type safety issues detected",
                details=safety_analysis
            )
    
    def _analyze_type_safety(self, kernel_code: str, backend: str) -> Dict[str, Any]:
        """Analyze type safety patterns in the code"""
        analysis = {
            "critical_issues": [],
            "warnings": [],
            "type_casts": [],
            "template_usage": [],
            "implicit_conversions": [],
            "safe_patterns": []
        }
        
        # Check for type casting
        analysis["type_casts"] = self._check_type_casting(kernel_code, backend)
        
        # Check for template usage
        analysis["template_usage"] = self._check_template_usage(kernel_code, backend)
        
        # Check for implicit conversions
        analysis["implicit_conversions"] = self._check_implicit_conversions(kernel_code, backend)
        
        # Check for safe patterns
        analysis["safe_patterns"] = self._check_safe_type_patterns(kernel_code, backend)
        
        # Compile issues and warnings
        analysis["critical_issues"] = self._compile_critical_issues(analysis)
        analysis["warnings"] = self._compile_warnings(analysis)
        
        return analysis
    
    def _check_type_casting(self, kernel_code: str, backend: str) -> List[Dict[str, Any]]:
        """Check for type casting patterns"""
        casts = []
        
        # C-style casts
        c_style_casts = re.finditer(r'\([^)]*\)\s*[a-zA-Z_][a-zA-Z0-9_]*', kernel_code)
        for match in c_style_casts:
            cast_expr = match.group(0)
            casts.append({
                "type": "c_style_cast",
                "expression": cast_expr,
                "line": kernel_code[:match.start()].count('\n') + 1,
                "severity": "warning"
            })
        
        # C++ static_cast
        static_casts = re.finditer(r'static_cast\s*<\s*([^>]+)\s*>\s*\([^)]+\)', kernel_code)
        for match in static_casts:
            target_type = match.group(1).strip()
            casts.append({
                "type": "static_cast",
                "target_type": target_type,
                "expression": match.group(0),
                "line": kernel_code[:match.start()].count('\n') + 1,
                "severity": "info"
            })
        
        # C++ reinterpret_cast
        reinterpret_casts = re.finditer(r'reinterpret_cast\s*<\s*([^>]+)\s*>\s*\([^)]+\)', kernel_code)
        for match in reinterpret_casts:
            target_type = match.group(1).strip()
            casts.append({
                "type": "reinterpret_cast",
                "target_type": target_type,
                "expression": match.group(0),
                "line": kernel_code[:match.start()].count('\n') + 1,
                "severity": "warning"
            })
        
        # C++ const_cast
        const_casts = re.finditer(r'const_cast\s*<\s*([^>]+)\s*>\s*\([^)]+\)', kernel_code)
        for match in const_casts:
            target_type = match.group(1).strip()
            casts.append({
                "type": "const_cast",
                "target_type": target_type,
                "expression": match.group(0),
                "line": kernel_code[:match.start()].count('\n') + 1,
                "severity": "warning"
            })
        
        return casts
    
    def _check_template_usage(self, kernel_code: str, backend: str) -> List[Dict[str, Any]]:
        """Check for template usage patterns"""
        templates = []
        
        # Template function declarations
        template_funcs = re.finditer(r'template\s*<\s*[^>]*>\s*\w+', kernel_code)
        for match in template_funcs:
            templates.append({
                "type": "template_function",
                "expression": match.group(0),
                "line": kernel_code[:match.start()].count('\n') + 1,
                "severity": "info"
            })
        
        # Template class declarations
        template_classes = re.finditer(r'template\s*<\s*[^>]*>\s*class\s+\w+', kernel_code)
        for match in template_classes:
            templates.append({
                "type": "template_class",
                "expression": match.group(0),
                "line": kernel_code[:match.start()].count('\n') + 1,
                "severity": "info"
            })
        
        return templates
    
    def _check_implicit_conversions(self, kernel_code: str, backend: str) -> List[Dict[str, Any]]:
        """Check for potentially dangerous implicit conversions"""
        conversions = []
        
        # Float to int conversions
        float_to_int = re.finditer(r'(\w+)\s*=\s*([0-9]*\.?[0-9]+)\s*;', kernel_code)
        for match in float_to_int:
            var_name = match.group(1)
            float_val = match.group(2)
            if '.' in float_val:
                conversions.append({
                    "type": "float_to_int",
                    "variable": var_name,
                    "value": float_val,
                    "line": kernel_code[:match.start()].count('\n') + 1,
                    "severity": "warning"
                })
        
        # Pointer to int conversions
        ptr_to_int = re.finditer(r'(\w+)\s*=\s*\([^)]*\)\s*(\w+)', kernel_code)
        for match in ptr_to_int:
            var_name = match.group(1)
            expr = match.group(0)
            if 'int' in expr and ('*' in expr or 'ptr' in expr.lower()):
                conversions.append({
                    "type": "pointer_to_int",
                    "variable": var_name,
                    "expression": expr,
                    "line": kernel_code[:match.start()].count('\n') + 1,
                    "severity": "warning"
                })
        
        return conversions
    
    def _check_safe_type_patterns(self, kernel_code: str, backend: str) -> List[Dict[str, Any]]:
        """Check for safe type patterns"""
        safe_patterns = []
        
        # Explicit type declarations
        explicit_types = re.finditer(r'(int|float|double|char|bool)\s+\w+\s*=', kernel_code)
        for match in explicit_types:
            safe_patterns.append({
                "type": "explicit_type_declaration",
                "expression": match.group(0),
                "line": kernel_code[:match.start()].count('\n') + 1
            })
        
        # Type checking with sizeof
        sizeof_usage = re.finditer(r'sizeof\s*\([^)]+\)', kernel_code)
        for match in sizeof_usage:
            safe_patterns.append({
                "type": "sizeof_usage",
                "expression": match.group(0),
                "line": kernel_code[:match.start()].count('\n') + 1
            })
        
        # Type traits usage (C++)
        type_traits = re.finditer(r'(is_same|is_integral|is_floating_point)\s*<[^>]+>', kernel_code)
        for match in type_traits:
            safe_patterns.append({
                "type": "type_trait_usage",
                "expression": match.group(0),
                "line": kernel_code[:match.start()].count('\n') + 1
            })
        
        return safe_patterns
    
    def _compile_critical_issues(self, analysis: Dict[str, Any]) -> List[str]:
        """Compile critical type safety issues"""
        issues = []
        
        for cast in analysis["type_casts"]:
            if cast["severity"] == "warning" and cast["type"] in ["reinterpret_cast", "const_cast"]:
                issues.append(f"Dangerous cast: {cast['expression']}")
        
        for conversion in analysis["implicit_conversions"]:
            if conversion["severity"] == "warning":
                issues.append(f"Implicit conversion: {conversion['type']} - {conversion['variable']}")
        
        return issues
    
    async def _llm_type_safety_analysis(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Use LLM for advanced type safety analysis"""
        prompt = f"""
        Analyze the following {backend} kernel code for type safety issues specific to {hardware}:
        
        ```{backend.lower()}
        {kernel_code}
        ```
        
        Focus on:
        1. **Hardware-specific type optimizations** for {hardware}
        2. **Backend-specific type requirements** for {backend}
        3. **Performance implications** of type choices
        4. **Memory layout and alignment** issues
        5. **Cross-function type consistency**
        6. **Template instantiation** correctness
        
        Provide:
        - Critical type safety issues
        - Performance recommendations
        - Hardware-specific optimizations
        - Backend compatibility issues
        - Code examples for fixes
        """
        
        try:
            response = await self.llm_analyzer.llm.ainvoke(prompt)
            return {
                "raw_response": response.content,
                "analysis_type": "type_safety_llm",
                "hardware": hardware,
                "backend": backend
            }
        except Exception as e:
            return {"error": f"LLM analysis failed: {str(e)}"}
    
    def _compile_warnings(self, analysis: Dict[str, Any]) -> List[str]:
        """Compile type safety warnings"""
        warnings = []
        
        for cast in analysis["type_casts"]:
            if cast["severity"] == "warning":
                warnings.append(f"Type cast: {cast['type']} - {cast['expression']}")
        
        for conversion in analysis["implicit_conversions"]:
            if conversion["severity"] == "warning":
                warnings.append(f"Implicit conversion: {conversion['type']} - {conversion['variable']}")
        
        return warnings
    
    def get_suggestions(self, kernel_code: str, backend: str) -> List[str]:
        """Get suggestions for improving type safety"""
        suggestions = []
        
        # Check for C-style casts
        if re.search(r'\([^)]*\)\s*[a-zA-Z_][a-zA-Z0-9_]*', kernel_code):
            suggestions.append("Replace C-style casts with C++ static_cast or reinterpret_cast")
        
        # Check for missing explicit types
        if re.search(r'auto\s+\w+\s*=', kernel_code):
            suggestions.append("Consider explicit type declarations instead of 'auto' for clarity")
        
        # Check for missing type checking
        if re.search(r'sizeof\s*\([^)]+\)', kernel_code) == None:
            suggestions.append("Consider using sizeof() for type-safe size calculations")
        
        return suggestions
    
    def _analyze_hardware_specific_types(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Analyze type usage specific to hardware capabilities"""
        analysis = {
            "hardware_recommendations": [],
            "precision_issues": [],
            "memory_alignment_issues": [],
            "tensor_core_usage": []
        }
        
        # NVIDIA-specific analysis
        if "NVIDIA" in hardware:
            # Check for Tensor Core usage
            if "H100" in hardware or "A100" in hardware:
                analysis["tensor_core_usage"] = self._check_tensor_core_types(kernel_code)
            
            # Check for mixed precision
            analysis["precision_issues"] = self._check_mixed_precision(kernel_code, hardware)
            
            # Check for memory alignment
            analysis["memory_alignment_issues"] = self._check_memory_alignment(kernel_code, hardware)
        
        # AMD-specific analysis
        elif "AMD" in hardware:
            # Check for ROCm-specific types
            analysis["hardware_recommendations"] = self._check_rocm_types(kernel_code)
            
            # Check for memory alignment (AMD has different requirements)
            analysis["memory_alignment_issues"] = self._check_amd_alignment(kernel_code)
        
        return analysis
    
    def _analyze_backend_specific_types(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Analyze type usage specific to backend APIs"""
        analysis = {
            "api_type_issues": [],
            "template_issues": [],
            "backend_recommendations": []
        }
        
        if backend == "CUDA":
            analysis["api_type_issues"] = self._check_cuda_api_types(kernel_code)
            analysis["template_issues"] = self._check_cuda_templates(kernel_code)
        elif backend == "OpenCL":
            analysis["api_type_issues"] = self._check_opencl_types(kernel_code)
        elif backend == "Triton":
            analysis["api_type_issues"] = self._check_triton_types(kernel_code)
        
        return analysis
    
    def _check_tensor_core_types(self, kernel_code: str) -> List[Dict[str, Any]]:
        """Check for proper Tensor Core type usage"""
        issues = []
        
        # Check for FP16 usage in Tensor Core operations
        if "mma" in kernel_code.lower() or "wmma" in kernel_code.lower():
            if "half" not in kernel_code and "fp16" not in kernel_code.lower():
                issues.append({
                    "type": "missing_fp16",
                    "message": "Tensor Core operations should use FP16 for optimal performance",
                    "severity": "warning"
                })
        
        return issues
    
    async def _llm_type_safety_analysis(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Use LLM for advanced type safety analysis"""
        prompt = f"""
        Analyze the following {backend} kernel code for type safety issues specific to {hardware}:
        
        ```{backend.lower()}
        {kernel_code}
        ```
        
        Focus on:
        1. **Hardware-specific type optimizations** for {hardware}
        2. **Backend-specific type requirements** for {backend}
        3. **Performance implications** of type choices
        4. **Memory layout and alignment** issues
        5. **Cross-function type consistency**
        6. **Template instantiation** correctness
        
        Provide:
        - Critical type safety issues
        - Performance recommendations
        - Hardware-specific optimizations
        - Backend compatibility issues
        - Code examples for fixes
        """
        
        try:
            response = await self.llm_analyzer.llm.ainvoke(prompt)
            return {
                "raw_response": response.content,
                "analysis_type": "type_safety_llm",
                "hardware": hardware,
                "backend": backend
            }
        except Exception as e:
            return {"error": f"LLM analysis failed: {str(e)}"}
    
    def _check_mixed_precision(self, kernel_code: str, hardware: str) -> List[Dict[str, Any]]:
        """Check for mixed precision issues"""
        issues = []
        
        # Check for FP32/FP16 mixing
        has_fp32 = "float" in kernel_code and "double" not in kernel_code
        has_fp16 = "half" in kernel_code or "fp16" in kernel_code.lower()
        
        if has_fp32 and has_fp16:
            issues.append({
                "type": "mixed_precision",
                "message": "Mixed FP32/FP16 precision detected - ensure proper casting",
                "severity": "warning"
            })
        
        return issues
    
    async def _llm_type_safety_analysis(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Use LLM for advanced type safety analysis"""
        prompt = f"""
        Analyze the following {backend} kernel code for type safety issues specific to {hardware}:
        
        ```{backend.lower()}
        {kernel_code}
        ```
        
        Focus on:
        1. **Hardware-specific type optimizations** for {hardware}
        2. **Backend-specific type requirements** for {backend}
        3. **Performance implications** of type choices
        4. **Memory layout and alignment** issues
        5. **Cross-function type consistency**
        6. **Template instantiation** correctness
        
        Provide:
        - Critical type safety issues
        - Performance recommendations
        - Hardware-specific optimizations
        - Backend compatibility issues
        - Code examples for fixes
        """
        
        try:
            response = await self.llm_analyzer.llm.ainvoke(prompt)
            return {
                "raw_response": response.content,
                "analysis_type": "type_safety_llm",
                "hardware": hardware,
                "backend": backend
            }
        except Exception as e:
            return {"error": f"LLM analysis failed: {str(e)}"}
    
    def _check_memory_alignment(self, kernel_code: str, hardware: str) -> List[Dict[str, Any]]:
        """Check for memory alignment issues"""
        issues = []
        
        # Check for unaligned memory access
        if "float4" in kernel_code or "double2" in kernel_code:
            if "alignas" not in kernel_code and "__align__" not in kernel_code:
                issues.append({
                    "type": "alignment_issue",
                    "message": "Vector types should be properly aligned for optimal performance",
                    "severity": "warning"
                })
        
        return issues
    
    async def _llm_type_safety_analysis(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Use LLM for advanced type safety analysis"""
        prompt = f"""
        Analyze the following {backend} kernel code for type safety issues specific to {hardware}:
        
        ```{backend.lower()}
        {kernel_code}
        ```
        
        Focus on:
        1. **Hardware-specific type optimizations** for {hardware}
        2. **Backend-specific type requirements** for {backend}
        3. **Performance implications** of type choices
        4. **Memory layout and alignment** issues
        5. **Cross-function type consistency**
        6. **Template instantiation** correctness
        
        Provide:
        - Critical type safety issues
        - Performance recommendations
        - Hardware-specific optimizations
        - Backend compatibility issues
        - Code examples for fixes
        """
        
        try:
            response = await self.llm_analyzer.llm.ainvoke(prompt)
            return {
                "raw_response": response.content,
                "analysis_type": "type_safety_llm",
                "hardware": hardware,
                "backend": backend
            }
        except Exception as e:
            return {"error": f"LLM analysis failed: {str(e)}"}
    
    def _check_rocm_types(self, kernel_code: str) -> List[Dict[str, Any]]:
        """Check for ROCm-specific type usage"""
        recommendations = []
        
        # Check for HIP-specific types
        if "hip" in kernel_code.lower():
            if "hipFloatComplex" not in kernel_code and "hipDoubleComplex" not in kernel_code:
                recommendations.append({
                    "type": "hip_types",
                    "message": "Consider using HIP-specific complex types for better performance",
                    "severity": "info"
                })
        
        return recommendations
    
    def _check_amd_alignment(self, kernel_code: str) -> List[Dict[str, Any]]:
        """Check for AMD-specific alignment requirements"""
        issues = []
        
        # AMD GPUs have different alignment requirements
        if "float4" in kernel_code:
            issues.append({
                "type": "amd_alignment",
                "message": "AMD GPUs require 16-byte alignment for float4 operations",
                "severity": "warning"
            })
        
        return issues
    
    async def _llm_type_safety_analysis(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Use LLM for advanced type safety analysis"""
        prompt = f"""
        Analyze the following {backend} kernel code for type safety issues specific to {hardware}:
        
        ```{backend.lower()}
        {kernel_code}
        ```
        
        Focus on:
        1. **Hardware-specific type optimizations** for {hardware}
        2. **Backend-specific type requirements** for {backend}
        3. **Performance implications** of type choices
        4. **Memory layout and alignment** issues
        5. **Cross-function type consistency**
        6. **Template instantiation** correctness
        
        Provide:
        - Critical type safety issues
        - Performance recommendations
        - Hardware-specific optimizations
        - Backend compatibility issues
        - Code examples for fixes
        """
        
        try:
            response = await self.llm_analyzer.llm.ainvoke(prompt)
            return {
                "raw_response": response.content,
                "analysis_type": "type_safety_llm",
                "hardware": hardware,
                "backend": backend
            }
        except Exception as e:
            return {"error": f"LLM analysis failed: {str(e)}"}
    
    def _check_cuda_api_types(self, kernel_code: str) -> List[Dict[str, Any]]:
        """Check for CUDA API type usage"""
        issues = []
        
        # Check for proper CUDA types
        if "cudaMalloc" in kernel_code:
            if "void**" not in kernel_code and "void *" not in kernel_code:
                issues.append({
                    "type": "cuda_malloc_type",
                    "message": "cudaMalloc should use void** for proper type safety",
                    "severity": "error"
                })
        
        return issues
    
    async def _llm_type_safety_analysis(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Use LLM for advanced type safety analysis"""
        prompt = f"""
        Analyze the following {backend} kernel code for type safety issues specific to {hardware}:
        
        ```{backend.lower()}
        {kernel_code}
        ```
        
        Focus on:
        1. **Hardware-specific type optimizations** for {hardware}
        2. **Backend-specific type requirements** for {backend}
        3. **Performance implications** of type choices
        4. **Memory layout and alignment** issues
        5. **Cross-function type consistency**
        6. **Template instantiation** correctness
        
        Provide:
        - Critical type safety issues
        - Performance recommendations
        - Hardware-specific optimizations
        - Backend compatibility issues
        - Code examples for fixes
        """
        
        try:
            response = await self.llm_analyzer.llm.ainvoke(prompt)
            return {
                "raw_response": response.content,
                "analysis_type": "type_safety_llm",
                "hardware": hardware,
                "backend": backend
            }
        except Exception as e:
            return {"error": f"LLM analysis failed: {str(e)}"}
    
    def _check_opencl_types(self, kernel_code: str) -> List[Dict[str, Any]]:
        """Check for OpenCL type usage"""
        issues = []
        
        # Check for OpenCL-specific types
        if "cl_mem" in kernel_code:
            if "cl_int" not in kernel_code:
                issues.append({
                    "type": "opencl_error_handling",
                    "message": "OpenCL functions should check cl_int return values",
                    "severity": "warning"
                })
        
        return issues
    
    async def _llm_type_safety_analysis(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Use LLM for advanced type safety analysis"""
        prompt = f"""
        Analyze the following {backend} kernel code for type safety issues specific to {hardware}:
        
        ```{backend.lower()}
        {kernel_code}
        ```
        
        Focus on:
        1. **Hardware-specific type optimizations** for {hardware}
        2. **Backend-specific type requirements** for {backend}
        3. **Performance implications** of type choices
        4. **Memory layout and alignment** issues
        5. **Cross-function type consistency**
        6. **Template instantiation** correctness
        
        Provide:
        - Critical type safety issues
        - Performance recommendations
        - Hardware-specific optimizations
        - Backend compatibility issues
        - Code examples for fixes
        """
        
        try:
            response = await self.llm_analyzer.llm.ainvoke(prompt)
            return {
                "raw_response": response.content,
                "analysis_type": "type_safety_llm",
                "hardware": hardware,
                "backend": backend
            }
        except Exception as e:
            return {"error": f"LLM analysis failed: {str(e)}"}
    
    def _check_triton_types(self, kernel_code: str) -> List[Dict[str, Any]]:
        """Check for Triton type usage"""
        issues = []
        
        # Check for Triton-specific types
        if "triton" in kernel_code.lower():
            if "tl" not in kernel_code:
                issues.append({
                    "type": "triton_import",
                    "message": "Triton kernels should import tl (triton.language)",
                    "severity": "error"
                })
        
        return issues
    
    async def _llm_type_safety_analysis(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Use LLM for advanced type safety analysis"""
        prompt = f"""
        Analyze the following {backend} kernel code for type safety issues specific to {hardware}:
        
        ```{backend.lower()}
        {kernel_code}
        ```
        
        Focus on:
        1. **Hardware-specific type optimizations** for {hardware}
        2. **Backend-specific type requirements** for {backend}
        3. **Performance implications** of type choices
        4. **Memory layout and alignment** issues
        5. **Cross-function type consistency**
        6. **Template instantiation** correctness
        
        Provide:
        - Critical type safety issues
        - Performance recommendations
        - Hardware-specific optimizations
        - Backend compatibility issues
        - Code examples for fixes
        """
        
        try:
            response = await self.llm_analyzer.llm.ainvoke(prompt)
            return {
                "raw_response": response.content,
                "analysis_type": "type_safety_llm",
                "hardware": hardware,
                "backend": backend
            }
        except Exception as e:
            return {"error": f"LLM analysis failed: {str(e)}"}
    
    def _check_cuda_templates(self, kernel_code: str) -> List[Dict[str, Any]]:
        """Check for CUDA template usage"""
        issues = []
        
        # Check for proper CUDA template syntax
        if "template" in kernel_code and "cuda" in kernel_code.lower():
            if "__global__" not in kernel_code:
                issues.append({
                    "type": "cuda_template",
                    "message": "CUDA template functions should be marked with __global__",
                    "severity": "warning"
                })
        
        return issues
    
    async def _llm_type_safety_analysis(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Use LLM for advanced type safety analysis"""
        prompt = f"""
        Analyze the following {backend} kernel code for type safety issues specific to {hardware}:
        
        ```{backend.lower()}
        {kernel_code}
        ```
        
        Focus on:
        1. **Hardware-specific type optimizations** for {hardware}
        2. **Backend-specific type requirements** for {backend}
        3. **Performance implications** of type choices
        4. **Memory layout and alignment** issues
        5. **Cross-function type consistency**
        6. **Template instantiation** correctness
        
        Provide:
        - Critical type safety issues
        - Performance recommendations
        - Hardware-specific optimizations
        - Backend compatibility issues
        - Code examples for fixes
        """
        
        try:
            response = await self.llm_analyzer.llm.ainvoke(prompt)
            return {
                "raw_response": response.content,
                "analysis_type": "type_safety_llm",
                "hardware": hardware,
                "backend": backend
            }
        except Exception as e:
            return {"error": f"LLM analysis failed: {str(e)}"}

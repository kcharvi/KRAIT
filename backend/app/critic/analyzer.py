"""
Core analysis engine for the Critic Agent
"""

import re
import time
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from .critic_models import (
    CriticAnalysis, CriticAnalysisRequest, CorrectnessAnalysis, 
    PerformanceAnalysis, PerformanceMetrics, Suggestion, 
    CorrectnessCheck, CheckStatus, SeverityLevel, HardwareUtilization
)
from .checks.bounds_checker import BoundsChecker
from .checks.synchronization_checker import SynchronizationChecker
from .checks.memory_safety import MemorySafetyChecker
from .checks.type_safety import TypeSafetyChecker
from .llm_analysis import LLMCorrectnessAnalyzer


class KernelAnalyzer:
    """
    Core analyzer for GPU kernel code analysis
    """
    
    def __init__(self):
        self.analysis_start_time = None
        self.supported_hardware = [
            "NVIDIA H100", "NVIDIA A100", "NVIDIA V100", "NVIDIA RTX 4090",
            "AMD MI300X", "AMD MI250X", "AMD MI200", "CPU"
        ]
        self.supported_backends = ["CUDA", "Triton", "OpenCL", "C++", "C"]
        self.llm_analyzer = LLMCorrectnessAnalyzer()
    
    async def analyze_kernel(self, request: CriticAnalysisRequest) -> CriticAnalysis:
        """
        Perform comprehensive analysis of a kernel
        """
        self.analysis_start_time = time.time()
        analysis_id = str(uuid.uuid4())
        
        try:
            print(f"Starting analysis for kernel code length: {len(request.kernel_code)}")
            print(f"Hardware: {request.hardware}, Backend: {request.backend}")
            
            # Run correctness analysis
            correctness = await self._analyze_correctness(request.kernel_code, request.hardware, request.backend)
            print(f"Correctness analysis completed: {correctness.status}, {len(correctness.checks)} checks")
            
            # Run performance analysis
            performance = await self._analyze_performance(request.kernel_code, request.hardware, request.backend)
            print(f"Performance analysis completed")
            
            # Generate suggestions
            suggestions = await self._generate_suggestions(request.kernel_code, request.hardware, request.backend, correctness, performance)
            # Optional LLM correctness review (concise)
            llm_suggestion = await self._llm_correctness_review(request.kernel_code, request.hardware, request.backend, request.use_llm_review)
            if llm_suggestion:
                suggestions.append(llm_suggestion)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(correctness, performance)
            
            # Calculate analysis time
            analysis_time_ms = (time.time() - self.analysis_start_time) * 1000
            
            return CriticAnalysis(
                analysis_id=analysis_id,
                kernel_code=request.kernel_code,
                hardware_config=request.hardware,
                backend=request.backend,
                correctness=correctness,
                performance=performance,
                suggestions=suggestions,
                overall_score=overall_score,
                analysis_time_ms=analysis_time_ms,
                generated_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            # Return error analysis
            return CriticAnalysis(
                analysis_id=analysis_id,
                kernel_code=request.kernel_code,
                hardware_config=request.hardware,
                backend=request.backend,
                correctness=CorrectnessAnalysis(
                    status=CheckStatus.FAIL,
                    score=0,
                    checks=[],
                    issues=[f"Analysis failed: {str(e)}"]
                ),
                performance=PerformanceAnalysis(metrics=PerformanceMetrics()),
                suggestions=[],
                overall_score=0,
                analysis_time_ms=(time.time() - self.analysis_start_time) * 1000,
                generated_at=datetime.now().isoformat()
            )
    
    async def _analyze_correctness(self, kernel_code: str, hardware: str, backend: str) -> CorrectnessAnalysis:
        """
        Analyze kernel correctness
        """
        checks = []
        issues = []
        
        # Check 1: Bounds checking
        try:
            bounds_check = await BoundsChecker().run_check(kernel_code, hardware, backend)
            print(f"Bounds check result: {bounds_check.status} - {bounds_check.message}")
        except Exception as e:
            print(f"Bounds check failed, using fallback: {e}")
            # Fallback to simple regex-based check if advanced checker fails
            bounds_check = self._check_bounds(kernel_code, backend)
        checks.append(bounds_check)
        if bounds_check.status == CheckStatus.FAIL:
            issues.append("Missing or insufficient bounds checking")
        
        # Check 2: Synchronization
        try:
            sync_check = await SynchronizationChecker().run_check(kernel_code, hardware, backend)
            print(f"Sync check result: {sync_check.status} - {sync_check.message}")
        except Exception as e:
            print(f"Sync check failed, using fallback: {e}")
            # Fallback to simple regex-based check if advanced checker fails
            sync_check = self._check_synchronization(kernel_code, backend)
        checks.append(sync_check)
        if sync_check.status == CheckStatus.WARNING:
            issues.append("Missing synchronization barriers")
        
        # Check 3: Memory safety
        try:
            memory_check = await MemorySafetyChecker().run_check(kernel_code, hardware, backend)
            print(f"Memory safety check result: {memory_check.status} - {memory_check.message}")
        except Exception as e:
            print(f"Memory safety check failed, using fallback: {e}")
            memory_check = self._check_memory_safety(kernel_code, backend)
        checks.append(memory_check)
        if memory_check.status == CheckStatus.FAIL:
            issues.append("Potential memory safety issues")
        
        # Check 4: Type safety
        try:
            type_check = await TypeSafetyChecker().run_check(kernel_code, hardware, backend)
            print(f"Type safety check result: {type_check.status} - {type_check.message}")
        except Exception as e:
            print(f"Type safety check failed, using fallback: {e}")
            type_check = self._check_type_safety(kernel_code, backend)
        checks.append(type_check)
        if type_check.status == CheckStatus.WARNING:
            issues.append("Type safety concerns")
        
        # Calculate correctness score
        passed_checks = sum(1 for check in checks if check.status == CheckStatus.PASS)
        total_checks = len(checks)
        score = int((passed_checks / total_checks) * 100) if total_checks > 0 else 0
        
        # Determine overall status
        if any(check.status == CheckStatus.FAIL for check in checks):
            status = CheckStatus.FAIL
        elif any(check.status == CheckStatus.WARNING for check in checks):
            status = CheckStatus.WARNING
        else:
            status = CheckStatus.PASS
        
        result = CorrectnessAnalysis(
            status=status,
            score=score,
            checks=checks,
            issues=issues
        )
        print(f"Final correctness analysis: {len(checks)} checks, {len(issues)} issues, score: {score}")
        return result
    
    async def _analyze_performance(self, kernel_code: str, hardware: str, backend: str) -> PerformanceAnalysis:
        """
        Analyze kernel performance
        """
        # Calculate basic metrics
        flops = self._calculate_flops(kernel_code)
        shared_mem = self._calculate_shared_memory(kernel_code)
        
        # Detect tiling patterns
        tile_detection = self._detect_tiling(kernel_code)
        
        # Check for vectorization
        vectorization_detected = self._detect_vectorization(kernel_code)
        
        # Check for Tensor Core usage
        tensor_core_usage = self._detect_tensor_core_usage(kernel_code)
        
        # Analyze memory access patterns
        memory_patterns = self._analyze_memory_access(kernel_code)
        
        # Estimate runtime (simplified)
        estimated_runtime = self._estimate_runtime(flops, shared_mem, hardware)
        
        # Determine if memory or compute bound
        memory_bound, compute_bound = self._determine_binding(flops, shared_mem, hardware)
        
        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(kernel_code, hardware, backend)
        
        metrics = PerformanceMetrics(
            estimated_runtime_ms=estimated_runtime,
            flops_total=flops,
            memory_bound=memory_bound,
            compute_bound=compute_bound,
            shared_mem_per_block_bytes=shared_mem,
            efficiency_score=efficiency_score
        )
        
        return PerformanceAnalysis(
            metrics=metrics,
            tile_detection=tile_detection,
            vectorization_detected=vectorization_detected,
            tensor_core_usage=tensor_core_usage,
            memory_access_patterns=memory_patterns
        )
    
    async def _generate_suggestions(self, kernel_code: str, hardware: str, backend: str, 
                                  correctness: CorrectnessAnalysis, performance: PerformanceAnalysis) -> List[Suggestion]:
        """
        Generate optimization suggestions
        """
        suggestions = []
        
        # Correctness-based suggestions
        for issue in correctness.issues:
            if "bounds" in issue.lower():
                suggestions.append(Suggestion(
                    severity=SeverityLevel.HIGH,
                    category="correctness",
                    title="Add bounds checking",
                    message="Add proper bounds checking to prevent out-of-bounds access",
                    code_snippet="if (idx < size) { /* safe access */ }",
                    impact="Prevents crashes and undefined behavior"
                ))
        
        # Performance-based suggestions
        if performance.metrics.memory_bound:
            suggestions.append(Suggestion(
                severity=SeverityLevel.MEDIUM,
                category="memory",
                title="Optimize memory access",
                message="Kernel appears to be memory bound. Consider tiling or shared memory usage",
                impact="Potential 2-10x speedup"
            ))
        
        if not performance.vectorization_detected:
            suggestions.append(Suggestion(
                severity=SeverityLevel.MEDIUM,
                category="compute",
                title="Add vectorization",
                message="Consider using vectorized operations for better performance",
                impact="Potential 2-4x speedup"
            ))
        
        if not performance.tensor_core_usage and hardware in ["NVIDIA H100", "NVIDIA A100"]:
            suggestions.append(Suggestion(
                severity=SeverityLevel.HIGH,
                category="compute",
                title="Use Tensor Cores",
                message="Consider using Tensor Cores for matrix operations on this hardware",
                impact="Potential 10-50x speedup for matrix operations"
            ))
        
        return suggestions

    async def _llm_correctness_review(self, kernel_code: str, hardware: str, backend: str, use_llm: bool = False) -> Optional[Suggestion]:
        """Optional LLM-based reasoning pass for complex correctness cases.
        Controlled by use_llm parameter from frontend request.
        """
        if not use_llm:
            return None
        
        try:
            # Run comprehensive LLM analysis
            analysis_results = {}
            
            # Control flow analysis
            control_flow = await self.llm_analyzer.analyze_control_flow(kernel_code, hardware, backend)
            analysis_results["control_flow"] = control_flow
            
            # Context-dependent bounds analysis
            context_bounds = await self.llm_analyzer.analyze_context_dependent_bounds(kernel_code, hardware, backend)
            analysis_results["context_bounds"] = context_bounds
            
            # Dynamic memory allocation analysis
            dynamic_memory = await self.llm_analyzer.analyze_dynamic_memory_allocation(kernel_code, hardware, backend)
            analysis_results["dynamic_memory"] = dynamic_memory
            
            # Cross-function analysis
            cross_function = await self.llm_analyzer.analyze_cross_function_dependencies(kernel_code, hardware, backend)
            analysis_results["cross_function"] = cross_function
            
            # Generate suggestions from analysis
            llm_suggestions = await self.llm_analyzer.generate_llm_suggestions(analysis_results)
            
            # Return a summary suggestion
            if llm_suggestions:
                return Suggestion(
                    severity=SeverityLevel.MEDIUM,
                    category="llm_analysis",
                    title="Advanced LLM Analysis",
                    message=f"Completed comprehensive analysis: {len(llm_suggestions)} insights found",
                    impact="Deep analysis of control flow, bounds checking, memory allocation, and function dependencies"
                )
            else:
                return Suggestion(
                    severity=SeverityLevel.LOW,
                    category="llm_analysis",
                    title="Advanced LLM Analysis",
                    message="Analysis completed - no critical issues detected",
                    impact="Comprehensive review passed"
                )
                
        except Exception as e:
            return Suggestion(
                severity=SeverityLevel.MEDIUM,
                category="llm_analysis",
                title="LLM Analysis Error",
                message=f"Advanced analysis failed: {str(e)}",
                impact="Falling back to static analysis only"
            )
    
    def _check_bounds(self, kernel_code: str, backend: str) -> CorrectnessCheck:
        """Check for proper bounds checking"""
        if backend.upper() == "CUDA":
            # Look for bounds checking patterns in CUDA
            bounds_patterns = [
                r'if\s*\(\s*.*\s*<\s*.*\s*\)',
                r'if\s*\(\s*.*\s*<=\s*.*\s*\)',
                r'if\s*\(\s*.*\s*>\s*.*\s*\)',
                r'if\s*\(\s*.*\s*>=\s*.*\s*\)'
            ]
            
            has_bounds = any(re.search(pattern, kernel_code, re.IGNORECASE) for pattern in bounds_patterns)
            
            if has_bounds:
                return CorrectnessCheck(
                    name="bounds_checking",
                    status=CheckStatus.PASS,
                    message="Bounds checking detected"
                )
            else:
                return CorrectnessCheck(
                    name="bounds_checking",
                    status=CheckStatus.FAIL,
                    message="No bounds checking detected"
                )
        
        return CorrectnessCheck(
            name="bounds_checking",
            status=CheckStatus.SKIP,
            message="Bounds checking not applicable for this backend"
        )
    
    def _check_synchronization(self, kernel_code: str, backend: str) -> CorrectnessCheck:
        """Check for proper synchronization"""
        if backend.upper() == "CUDA":
            has_sync = "__syncthreads()" in kernel_code
            
            if has_sync:
                return CorrectnessCheck(
                    name="synchronization",
                    status=CheckStatus.PASS,
                    message="Synchronization barriers detected"
                )
            else:
                return CorrectnessCheck(
                    name="synchronization",
                    status=CheckStatus.WARNING,
                    message="No synchronization barriers detected"
                )
        
        return CorrectnessCheck(
            name="synchronization",
            status=CheckStatus.SKIP,
            message="Synchronization not applicable for this backend"
        )
    
    def _check_memory_safety(self, kernel_code: str, backend: str) -> CorrectnessCheck:
        """Check for memory safety issues"""
        # Look for potential unsafe memory access patterns
        unsafe_patterns = [
            r'\[\s*[^]]*\s*\]\s*=',  # Array assignment without bounds
            r'pointer\s*\+\s*[^;]*',  # Pointer arithmetic
        ]
        
        has_unsafe = any(re.search(pattern, kernel_code, re.IGNORECASE) for pattern in unsafe_patterns)
        
        if has_unsafe:
            return CorrectnessCheck(
                name="memory_safety",
                status=CheckStatus.WARNING,
                message="Potential memory safety issues detected"
            )
        else:
            return CorrectnessCheck(
                name="memory_safety",
                status=CheckStatus.PASS,
                message="No obvious memory safety issues"
            )
    
    def _check_type_safety(self, kernel_code: str, backend: str) -> CorrectnessCheck:
        """Check for type safety"""
        # Look for type casting and template usage
        type_patterns = [
            r'static_cast\s*<',
            r'reinterpret_cast\s*<',
            r'\(.*\)\s*[a-zA-Z]',  # C-style casting
        ]
        
        has_casting = any(re.search(pattern, kernel_code) for pattern in type_patterns)
        
        if has_casting:
            return CorrectnessCheck(
                name="type_safety",
                status=CheckStatus.WARNING,
                message="Type casting detected - ensure safety"
            )
        else:
            return CorrectnessCheck(
                name="type_safety",
                status=CheckStatus.PASS,
                message="No obvious type safety issues"
            )
    
    def _calculate_flops(self, kernel_code: str) -> int:
        """Calculate estimated FLOPs"""
        # Count arithmetic operations
        flop_patterns = [
            r'\+', r'-', r'\*', r'/',  # Basic arithmetic
            r'fma\s*\(',  # Fused multiply-add
            r'exp\s*\(', r'log\s*\(', r'sin\s*\(', r'cos\s*\(',  # Math functions
        ]
        
        flops = 0
        for pattern in flop_patterns:
            matches = re.findall(pattern, kernel_code, re.IGNORECASE)
            flops += len(matches)
        
        return flops
    
    def _calculate_shared_memory(self, kernel_code: str) -> int:
        """Calculate shared memory usage"""
        # Look for __shared__ declarations
        shared_pattern = r'__shared__\s+.*?\[([^\]]+)\]'
        matches = re.findall(shared_pattern, kernel_code)
        
        total_bytes = 0
        for match in matches:
            # Simple size calculation (assume 4 bytes per element)
            try:
                size = eval(match.replace('*', ' * '))  # Basic evaluation
                total_bytes += size * 4
            except:
                total_bytes += 1024  # Default estimate
        
        return total_bytes
    
    def _detect_tiling(self, kernel_code: str) -> Dict[str, Any]:
        """Detect tiling patterns"""
        # Look for tiling patterns
        tile_patterns = {
            'shared_memory': '__shared__' in kernel_code,
            'block_dim': 'blockDim' in kernel_code,
            'thread_idx': 'threadIdx' in kernel_code,
            'block_idx': 'blockIdx' in kernel_code,
        }
        
        return tile_patterns
    
    def _detect_vectorization(self, kernel_code: str) -> bool:
        """Detect vectorized operations"""
        vector_patterns = [
            r'float4', r'double2', r'int4',  # Vector types
            r'\.x\b', r'\.y\b', r'\.z\b', r'\.w\b',  # Vector access
            r'vload', r'vstore',  # Vector load/store
        ]
        
        return any(re.search(pattern, kernel_code, re.IGNORECASE) for pattern in vector_patterns)
    
    def _detect_tensor_core_usage(self, kernel_code: str) -> bool:
        """Detect Tensor Core usage"""
        tensor_patterns = [
            r'wmma::',  # WMMA API
            r'mma\.',   # MMA instructions
            r'tensor',  # Tensor operations
        ]
        
        return any(re.search(pattern, kernel_code, re.IGNORECASE) for pattern in tensor_patterns)
    
    def _analyze_memory_access(self, kernel_code: str) -> Dict[str, Any]:
        """Analyze memory access patterns"""
        return {
            'coalesced_access': 'coalesced' in kernel_code.lower(),
            'stride_patterns': len(re.findall(r'\[.*\*.*\]', kernel_code)),
            'bank_conflicts': 'bank' in kernel_code.lower(),
        }
    
    def _estimate_runtime(self, flops: int, shared_mem: int, hardware: str) -> float:
        """Estimate kernel runtime (simplified)"""
        # Very basic estimation based on hardware specs
        hardware_specs = {
            "NVIDIA H100": {"peak_flops": 1e15, "memory_bw": 3.35e12},
            "NVIDIA A100": {"peak_flops": 3.12e14, "memory_bw": 2.04e12},
            "AMD MI300X": {"peak_flops": 8.2e14, "memory_bw": 5.3e12},
        }
        
        if hardware in hardware_specs:
            spec = hardware_specs[hardware]
            # Simple estimation: max of compute and memory bound
            compute_time = flops / spec["peak_flops"] * 1000  # ms
            memory_time = shared_mem / spec["memory_bw"] * 1000  # ms
            return max(compute_time, memory_time)
        
        return 1.0  # Default 1ms
    
    def _determine_binding(self, flops: int, shared_mem: int, hardware: str) -> tuple[bool, bool]:
        """Determine if kernel is memory or compute bound"""
        # Simple heuristic based on FLOPs to memory ratio
        if flops > shared_mem * 10:  # More compute than memory
            return False, True
        else:
            return True, False
    
    def _calculate_efficiency_score(self, kernel_code: str, hardware: str, backend: str) -> float:
        """Calculate overall efficiency score"""
        score = 50.0  # Base score
        
        # Add points for good practices
        if '__shared__' in kernel_code:
            score += 20  # Shared memory usage
        if 'threadIdx' in kernel_code and 'blockIdx' in kernel_code:
            score += 15  # Proper thread indexing
        if self._detect_vectorization(kernel_code):
            score += 15  # Vectorization
        if self._detect_tensor_core_usage(kernel_code):
            score += 20  # Tensor Core usage
        
        return min(100.0, score)
    
    def _calculate_overall_score(self, correctness: CorrectnessAnalysis, performance: PerformanceAnalysis) -> int:
        """Calculate overall analysis score"""
        correctness_weight = 0.4
        performance_weight = 0.6
        
        overall_score = (
            correctness.score * correctness_weight +
            performance.metrics.efficiency_score * performance_weight
        )
        
        return int(overall_score)

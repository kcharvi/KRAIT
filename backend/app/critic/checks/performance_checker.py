"""
Performance checking module for kernel analysis
"""

import re
from typing import Dict, Any, List, Tuple
from ..critic_models import PerformanceMetrics, Suggestion, SeverityLevel
from ..analysis.flop_calculator import FLOPCalculator
from ..analysis.memory_analyzer import MemoryAnalyzer
from ..analysis.tiling_detector import TilingDetector
from ..analysis.vectorization_checker import VectorizationChecker


class PerformanceChecker:
    """Check for performance-related issues and optimizations in kernel code"""
    
    def __init__(self):
        self.name = "performance_checker"
        self.description = "Checks for performance issues and optimization opportunities"
        self.flop_calculator = FLOPCalculator()
        self.memory_analyzer = MemoryAnalyzer()
        self.tiling_detector = TilingDetector()
        self.vectorization_checker = VectorizationChecker()
    
    async def run_check(self, kernel_code: str, hardware: str, backend: str) -> Tuple[PerformanceMetrics, List[Suggestion]]:
        """
        Comprehensive performance analysis using all analysis modules
        """
        # Run all performance analysis modules
        flop_analysis = await self.flop_calculator.calculate_flops(kernel_code, hardware, backend)
        memory_analysis = await self.memory_analyzer.analyze_memory(kernel_code, hardware, backend)
        tiling_analysis = await self.tiling_detector.detect_tiling(kernel_code, hardware, backend)
        vectorization_analysis = await self.vectorization_checker.check_vectorization(kernel_code, hardware, backend)
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            flops_total=flop_analysis.get("total_flops", 0),
            estimated_runtime_ms=self._estimate_runtime(flop_analysis, memory_analysis, hardware),
            bound=self._determine_bound(flop_analysis, memory_analysis),
            shared_mem_per_block_bytes=memory_analysis.get("shared_memory_bytes", 0),
            tiling_detected=tiling_analysis.get("tiling_detected", False),
            tile_m=tiling_analysis.get("tile_sizes", {}).get("M", None),
            tile_n=tiling_analysis.get("tile_sizes", {}).get("N", None),
            tile_k=tiling_analysis.get("tile_sizes", {}).get("K", None),
            vectorization_detected=vectorization_analysis.get("vectorization_detected", False),
            tensor_core_usage_detected=self._detect_tensor_core_usage(kernel_code, hardware),
            loop_unrolling_detected=self._detect_loop_unrolling(kernel_code)
        )
        
        # Generate suggestions
        suggestions = self._generate_performance_suggestions(
            flop_analysis, memory_analysis, tiling_analysis, vectorization_analysis, hardware, backend
        )
        
        return metrics, suggestions
    
    def _estimate_runtime(self, flop_analysis: Dict, memory_analysis: Dict, hardware: str) -> float:
        """Estimate kernel runtime in milliseconds"""
        total_flops = flop_analysis.get("total_flops", 0)
        
        # Hardware peak performance (FLOPs per second)
        hardware_peak = {
            "NVIDIA H100": 989e12,
            "NVIDIA A100": 312e12,
            "AMD MI300X": 1634e12,
            "CPU": 100e9,
        }
        
        peak_flops = hardware_peak.get(hardware, 100e9)
        if peak_flops > 0:
            runtime_seconds = total_flops / peak_flops
            return runtime_seconds * 1000  # Convert to milliseconds
        
        return 0.0
    
    def _determine_bound(self, flop_analysis: Dict, memory_analysis: Dict) -> str:
        """Determine if kernel is compute-bound or memory-bound"""
        total_flops = flop_analysis.get("total_flops", 0)
        memory_accesses = memory_analysis.get("global_memory_accesses", 0)
        
        # Simple heuristic: if many memory accesses relative to FLOPs, likely memory-bound
        if memory_accesses > 0 and total_flops / memory_accesses < 10:
            return "memory"
        elif total_flops > memory_accesses * 100:
            return "compute"
        else:
            return "unknown"
    
    def _detect_tensor_core_usage(self, kernel_code: str, hardware: str) -> bool:
        """Detect Tensor Core usage"""
        if "NVIDIA" not in hardware:
            return False
        
        tensor_core_patterns = [
            r'wmma\.|mma\.',  # WMMA intrinsics
            r'tensor',  # Tensor operations
            r'float16.*float16',  # FP16 operations
            r'__half',  # Half precision
        ]
        
        return any(re.search(pattern, kernel_code, re.IGNORECASE) for pattern in tensor_core_patterns)
    
    def _detect_loop_unrolling(self, kernel_code: str) -> bool:
        """Detect loop unrolling"""
        unroll_patterns = [
            r'#pragma\s+unroll',
            r'__pragma\s+unroll',
            r'#pragma\s+unroll\s*\(\d+\)',
        ]
        
        return any(re.search(pattern, kernel_code, re.IGNORECASE) for pattern in unroll_patterns)
    
    def _generate_performance_suggestions(self, flop_analysis: Dict, memory_analysis: Dict, 
                                        tiling_analysis: Dict, vectorization_analysis: Dict, 
                                        hardware: str, backend: str) -> List[Suggestion]:
        """Generate comprehensive performance suggestions"""
        suggestions = []
        
        # FLOP-based suggestions
        flop_suggestions = flop_analysis.get("suggestions", [])
        for suggestion in flop_suggestions:
            suggestions.append(Suggestion(
                severity=SeverityLevel.MEDIUM,
                category="compute",
                title="FLOP Optimization",
                message=suggestion
            ))
        
        # Memory-based suggestions
        memory_suggestions = memory_analysis.get("suggestions", [])
        for suggestion in memory_suggestions:
            suggestions.append(Suggestion(
                severity=SeverityLevel.HIGH,
                category="memory",
                title="Memory Optimization",
                message=suggestion
            ))
        
        # Tiling suggestions
        tiling_suggestions = tiling_analysis.get("suggestions", [])
        for suggestion in tiling_suggestions:
            suggestions.append(Suggestion(
                severity=SeverityLevel.HIGH,
                category="memory",
                title="Tiling Optimization",
                message=suggestion
            ))
        
        # Vectorization suggestions
        vectorization_suggestions = vectorization_analysis.get("suggestions", [])
        for suggestion in vectorization_suggestions:
            suggestions.append(Suggestion(
                severity=SeverityLevel.MEDIUM,
                category="compute",
                title="Vectorization",
                message=suggestion
            ))
        
        # Hardware-specific suggestions
        if "NVIDIA" in hardware and not vectorization_analysis.get("vectorization_detected", False):
            suggestions.append(Suggestion(
                severity=SeverityLevel.HIGH,
                category="compute",
                title="NVIDIA Optimization",
                message="Consider using Tensor Cores and vectorized operations for NVIDIA hardware"
            ))
        
        return suggestions
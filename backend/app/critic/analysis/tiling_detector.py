"""
Tiling pattern detector for kernel analysis
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from ..critic_models import PerformanceMetrics, Suggestion, SeverityLevel
from ..parsers.parser_factory import ParserFactory


class TilingDetector:
    """Detect and analyze tiling patterns in kernel code"""
    
    def __init__(self):
        self.name = "tiling_detector"
        self.description = "Detect and analyze tiling patterns for memory optimization"
        self.parser_factory = ParserFactory()
        
        # Tiling patterns
        self.tiling_patterns = {
            'cuda_tiling': r'__shared__\s+\w+\s+(\w+)\s*\[(\d+)\]\s*\[(\d+)\]',  # CUDA 2D tiling
            'cuda_1d_tiling': r'__shared__\s+\w+\s+(\w+)\s*\[(\d+)\]',  # CUDA 1D tiling
            'opencl_tiling': r'__local\s+\w+\s+(\w+)\s*\[(\d+)\]\s*\[(\d+)\]',  # OpenCL 2D tiling
            'triton_tiling': r'BLOCK_SIZE\s*=\s*(\d+)|TILE_SIZE\s*=\s*(\d+)',  # Triton tiling
            'loop_tiling': r'for\s*\(\s*\w+\s*=\s*\w+\s*;\s*\w+\s*<\s*(\w+)\s*;\s*\w+\s*\+\=\s*(\w+)\s*\)',  # Loop tiling
        }
        
        # Common tile sizes
        self.common_tile_sizes = [8, 16, 32, 64, 128, 256]
    
    async def detect_tiling(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Detect tiling patterns in kernel code"""
        # Parse the kernel code
        parsed_kernel = await self.parser_factory.parse(kernel_code)
        if not parsed_kernel:
            return {
                "tiling_detected": False,
                "tile_sizes": {},
                "tiling_efficiency": 0.0,
                "suggestions": ["Failed to parse kernel code"]
            }
        
        # Detect different types of tiling
        tiling_results = await self._detect_tiling_patterns(parsed_kernel, backend)
        
        # Analyze tiling efficiency
        efficiency_analysis = await self._analyze_tiling_efficiency(parsed_kernel, tiling_results, hardware)
        
        # Generate suggestions
        suggestions = self._generate_tiling_suggestions(parsed_kernel, tiling_results, hardware, backend)
        
        return {
            "tiling_detected": tiling_results["detected"],
            "tile_sizes": tiling_results["tile_sizes"],
            "tiling_type": tiling_results["tiling_type"],
            "tiling_efficiency": efficiency_analysis["efficiency_score"],
            "memory_reuse_ratio": efficiency_analysis["memory_reuse_ratio"],
            "cache_efficiency": efficiency_analysis["cache_efficiency"],
            "suggestions": suggestions
        }
    
    async def _detect_tiling_patterns(self, parsed_kernel, backend: str) -> Dict[str, Any]:
        """Detect various tiling patterns"""
        detected = False
        tile_sizes = {}
        tiling_type = "none"
        
        # Check for shared memory tiling
        shared_mem_tiles = self._detect_shared_memory_tiling(parsed_kernel, backend)
        if shared_mem_tiles:
            detected = True
            tiling_type = "shared_memory"
            tile_sizes.update(shared_mem_tiles)
        
        # Check for loop tiling
        loop_tiles = self._detect_loop_tiling(parsed_kernel)
        if loop_tiles:
            detected = True
            if tiling_type == "none":
                tiling_type = "loop_tiling"
            tile_sizes.update(loop_tiles)
        
        # Check for Triton tiling
        triton_tiles = self._detect_triton_tiling(parsed_kernel)
        if triton_tiles:
            detected = True
            tiling_type = "triton_tiling"
            tile_sizes.update(triton_tiles)
        
        return {
            "detected": detected,
            "tile_sizes": tile_sizes,
            "tiling_type": tiling_type
        }
    
    def _detect_shared_memory_tiling(self, parsed_kernel, backend: str) -> Dict[str, int]:
        """Detect shared memory tiling patterns"""
        tile_sizes = {}
        
        for var in parsed_kernel.variables:
            if var.is_shared and var.array_size:
                # Parse array dimensions
                dimensions = self._parse_array_dimensions(var.array_size)
                if len(dimensions) >= 2:
                    tile_sizes[f"{var.name}_M"] = dimensions[0]
                    tile_sizes[f"{var.name}_N"] = dimensions[1]
                    if len(dimensions) >= 3:
                        tile_sizes[f"{var.name}_K"] = dimensions[2]
                elif len(dimensions) == 1:
                    tile_sizes[f"{var.name}_1D"] = dimensions[0]
        
        return tile_sizes
    
    def _detect_loop_tiling(self, parsed_kernel) -> Dict[str, int]:
        """Detect loop tiling patterns"""
        tile_sizes = {}
        
        for loop in parsed_kernel.loops:
            if loop.type == "for" and loop.step:
                # Try to extract tile size from step
                step_value = self._extract_numeric_value(loop.step)
                if step_value and step_value in self.common_tile_sizes:
                    tile_sizes[f"loop_{loop.variable}"] = step_value
        
        return tile_sizes
    
    def _detect_triton_tiling(self, parsed_kernel) -> Dict[str, int]:
        """Detect Triton tiling patterns"""
        tile_sizes = {}
        
        # Look for Triton-specific tiling patterns
        triton_patterns = [
            r'BLOCK_SIZE\s*=\s*(\d+)',
            r'TILE_SIZE\s*=\s*(\d+)',
            r'BLOCK_M\s*=\s*(\d+)',
            r'BLOCK_N\s*=\s*(\d+)',
            r'BLOCK_K\s*=\s*(\d+)',
        ]
        
        for pattern in triton_patterns:
            matches = re.findall(pattern, parsed_kernel.raw_code)
            for match in matches:
                size = int(match)
                if size in self.common_tile_sizes:
                    tile_sizes[f"triton_{pattern.split('_')[0].lower()}"] = size
        
        return tile_sizes
    
    def _parse_array_dimensions(self, array_size: str) -> List[int]:
        """Parse array dimensions from string"""
        try:
            # Handle multi-dimensional arrays like "16][32" or "16*32"
            if '][' in array_size:
                dimensions = array_size.split('][')
            elif '*' in array_size:
                dimensions = array_size.split('*')
            else:
                dimensions = [array_size]
            
            return [int(dim.strip()) for dim in dimensions if dim.strip().isdigit()]
        except:
            return []
    
    def _extract_numeric_value(self, expr: str) -> Optional[int]:
        """Extract numeric value from expression"""
        try:
            # Simple extraction of numeric values
            numbers = re.findall(r'\d+', expr)
            if numbers:
                return int(numbers[0])
        except:
            pass
        return None
    
    async def _analyze_tiling_efficiency(self, parsed_kernel, tiling_results: Dict, hardware: str) -> Dict[str, Any]:
        """Analyze tiling efficiency"""
        efficiency_score = 0.0
        memory_reuse_ratio = 0.0
        cache_efficiency = 0.0
        
        if not tiling_results["detected"]:
            return {
                "efficiency_score": 0.0,
                "memory_reuse_ratio": 0.0,
                "cache_efficiency": 0.0
            }
        
        # Calculate efficiency based on tile sizes
        tile_sizes = tiling_results["tile_sizes"]
        if tile_sizes:
            # Check if tile sizes are optimal
            optimal_sizes = self._get_optimal_tile_sizes(hardware)
            efficiency_score = self._calculate_tile_efficiency(tile_sizes, optimal_sizes)
            
            # Calculate memory reuse ratio
            memory_reuse_ratio = self._calculate_memory_reuse_ratio(parsed_kernel, tile_sizes)
            
            # Calculate cache efficiency
            cache_efficiency = self._calculate_cache_efficiency(tile_sizes, hardware)
        
        return {
            "efficiency_score": efficiency_score,
            "memory_reuse_ratio": memory_reuse_ratio,
            "cache_efficiency": cache_efficiency
        }
    
    def _get_optimal_tile_sizes(self, hardware: str) -> Dict[str, int]:
        """Get optimal tile sizes for hardware"""
        optimal_sizes = {
            "NVIDIA H100": {"M": 32, "N": 32, "K": 32},
            "NVIDIA A100": {"M": 16, "N": 16, "K": 16},
            "AMD MI300X": {"M": 32, "N": 32, "K": 32},
            "CPU": {"M": 8, "N": 8, "K": 8},
        }
        return optimal_sizes.get(hardware, optimal_sizes["CPU"])
    
    def _calculate_tile_efficiency(self, tile_sizes: Dict[str, int], optimal_sizes: Dict[str, int]) -> float:
        """Calculate tile efficiency score"""
        if not tile_sizes:
            return 0.0
        
        efficiency = 0.0
        count = 0
        
        for key, size in tile_sizes.items():
            if key.endswith('_M') and 'M' in optimal_sizes:
                efficiency += min(100.0, (size / optimal_sizes['M']) * 100)
                count += 1
            elif key.endswith('_N') and 'N' in optimal_sizes:
                efficiency += min(100.0, (size / optimal_sizes['N']) * 100)
                count += 1
            elif key.endswith('_K') and 'K' in optimal_sizes:
                efficiency += min(100.0, (size / optimal_sizes['K']) * 100)
                count += 1
        
        return efficiency / max(1, count)
    
    def _calculate_memory_reuse_ratio(self, parsed_kernel, tile_sizes: Dict[str, int]) -> float:
        """Calculate memory reuse ratio"""
        # Simplified calculation based on tile sizes
        if not tile_sizes:
            return 0.0
        
        # Estimate reuse based on tile dimensions
        total_tile_elements = sum(tile_sizes.values())
        if total_tile_elements > 0:
            # Higher tile sizes generally mean better reuse
            return min(1.0, total_tile_elements / 1000)  # Normalize
        
        return 0.0
    
    def _calculate_cache_efficiency(self, tile_sizes: Dict[str, int], hardware: str) -> float:
        """Calculate cache efficiency"""
        if not tile_sizes:
            return 0.0
        
        # Check if tile sizes are powers of 2 (good for cache)
        power_of_2_count = 0
        total_count = 0
        
        for size in tile_sizes.values():
            total_count += 1
            if size & (size - 1) == 0:  # Check if power of 2
                power_of_2_count += 1
        
        return (power_of_2_count / max(1, total_count)) * 100
    
    def _generate_tiling_suggestions(self, parsed_kernel, tiling_results: Dict, hardware: str, backend: str) -> List[str]:
        """Generate tiling optimization suggestions"""
        suggestions = []
        
        if not tiling_results["detected"]:
            suggestions.append("Consider implementing tiling to improve memory access patterns")
            suggestions.append("Use shared memory to cache frequently accessed data")
            return suggestions
        
        # Analyze tile sizes
        tile_sizes = tiling_results["tile_sizes"]
        optimal_sizes = self._get_optimal_tile_sizes(hardware)
        
        # Check for suboptimal tile sizes
        for key, size in tile_sizes.items():
            if key.endswith('_M') and 'M' in optimal_sizes:
                if size < optimal_sizes['M']:
                    suggestions.append(f"Consider increasing M tile size from {size} to {optimal_sizes['M']}")
                elif size > optimal_sizes['M'] * 2:
                    suggestions.append(f"Consider decreasing M tile size from {size} to {optimal_sizes['M']}")
            
            if key.endswith('_N') and 'N' in optimal_sizes:
                if size < optimal_sizes['N']:
                    suggestions.append(f"Consider increasing N tile size from {size} to {optimal_sizes['N']}")
                elif size > optimal_sizes['N'] * 2:
                    suggestions.append(f"Consider decreasing N tile size from {size} to {optimal_sizes['N']}")
        
        # Check for power of 2 tile sizes
        non_power_of_2 = [size for size in tile_sizes.values() if size & (size - 1) != 0]
        if non_power_of_2:
            suggestions.append("Consider using powers of 2 for tile sizes to improve cache efficiency")
        
        # Backend-specific suggestions
        if backend.upper() == "CUDA":
            if not any("__shared__" in line for line in parsed_kernel.lines):
                suggestions.append("Consider using __shared__ memory for tiling in CUDA")
        
        elif backend.upper() == "TRITON":
            if not re.search(r'BLOCK_SIZE|TILE_SIZE', parsed_kernel.raw_code):
                suggestions.append("Consider defining BLOCK_SIZE or TILE_SIZE constants in Triton")
        
        return suggestions

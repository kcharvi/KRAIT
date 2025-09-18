"""
Synchronization checking module for kernel analysis
"""

import re
from typing import Dict, Any, List
from ..critic_models import CorrectnessCheck, CheckStatus


class SynchronizationChecker:
    """Check for proper synchronization in kernel code"""
    
    def __init__(self):
        self.name = "synchronization_checker"
        self.description = "Checks for proper synchronization barriers and memory consistency"
    
    async def run_check(self, kernel_code: str, hardware: str, backend: str) -> CorrectnessCheck:
        """
        Check for synchronization patterns in kernel code
        """
        if backend.upper() not in ["CUDA", "OpenCL"]:
            return CorrectnessCheck(
                name=self.name,
                status=CheckStatus.SKIP,
                message=f"Synchronization checking not applicable for {backend} backend"
            )
        
        sync_analysis = self._analyze_synchronization(kernel_code, backend)
        
        if sync_analysis["has_sync"]:
            return CorrectnessCheck(
                name=self.name,
                status=CheckStatus.PASS,
                message=f"Synchronization detected: {', '.join(sync_analysis['sync_types'])}",
                details=sync_analysis
            )
        else:
            return CorrectnessCheck(
                name=self.name,
                status=CheckStatus.WARNING,
                message="No synchronization barriers detected - may cause race conditions",
                details=sync_analysis
            )
    
    def _analyze_synchronization(self, kernel_code: str, backend: str) -> Dict[str, Any]:
        """Analyze synchronization patterns in the code"""
        analysis = {
            "has_sync": False,
            "sync_types": [],
            "sync_locations": [],
            "shared_memory_usage": False,
            "potential_race_conditions": []
        }
        
        if backend.upper() == "CUDA":
            # Check for CUDA synchronization
            cuda_sync_patterns = {
                "__syncthreads()": r'__syncthreads\s*\(\s*\)',
                "__syncwarp()": r'__syncwarp\s*\(\s*\)',
                "__threadfence()": r'__threadfence\s*\(\s*\)',
                "__threadfence_block()": r'__threadfence_block\s*\(\s*\)',
                "__threadfence_system()": r'__threadfence_system\s*\(\s*\)',
            }
            
            for sync_type, pattern in cuda_sync_patterns.items():
                matches = re.finditer(pattern, kernel_code, re.IGNORECASE)
                for match in matches:
                    analysis["sync_types"].append(sync_type)
                    analysis["sync_locations"].append({
                        "type": sync_type,
                        "line": kernel_code[:match.start()].count('\n') + 1,
                        "position": match.start()
                    })
                    analysis["has_sync"] = True
        
        elif backend.upper() == "OpenCL":
            # Check for OpenCL synchronization
            opencl_sync_patterns = {
                "barrier()": r'barrier\s*\(\s*CLK_LOCAL_MEM_FENCE\s*\)',
                "mem_fence()": r'mem_fence\s*\(\s*CLK_LOCAL_MEM_FENCE\s*\)',
                "read_mem_fence()": r'read_mem_fence\s*\(\s*CLK_LOCAL_MEM_FENCE\s*\)',
                "write_mem_fence()": r'write_mem_fence\s*\(\s*CLK_LOCAL_MEM_FENCE\s*\)',
            }
            
            for sync_type, pattern in opencl_sync_patterns.items():
                matches = re.finditer(pattern, kernel_code, re.IGNORECASE)
                for match in matches:
                    analysis["sync_types"].append(sync_type)
                    analysis["sync_locations"].append({
                        "type": sync_type,
                        "line": kernel_code[:match.start()].count('\n') + 1,
                        "position": match.start()
                    })
                    analysis["has_sync"] = True
        
        # Check for shared memory usage
        analysis["shared_memory_usage"] = self._has_shared_memory_usage(kernel_code, backend)
        
        # Check for potential race conditions
        analysis["potential_race_conditions"] = self._detect_race_conditions(kernel_code, backend)
        
        return analysis
    
    def _has_shared_memory_usage(self, kernel_code: str, backend: str) -> bool:
        """Check if kernel uses shared memory"""
        if backend.upper() == "CUDA":
            return "__shared__" in kernel_code
        elif backend.upper() == "OpenCL":
            return "__local" in kernel_code
        return False
    
    def _detect_race_conditions(self, kernel_code: str, backend: str) -> List[str]:
        """Detect potential race conditions"""
        race_conditions = []
        
        # Check for shared memory writes without synchronization
        if self._has_shared_memory_usage(kernel_code, backend):
            if backend.upper() == "CUDA":
                if "__shared__" in kernel_code and "__syncthreads()" not in kernel_code:
                    race_conditions.append("Shared memory writes without __syncthreads()")
            elif backend.upper() == "OpenCL":
                if "__local" in kernel_code and "barrier" not in kernel_code:
                    race_conditions.append("Local memory writes without barrier()")
        
        # Check for atomic operations that might need synchronization
        atomic_patterns = ["atomicAdd", "atomicSub", "atomicExch", "atomicCAS"]
        has_atomics = any(pattern in kernel_code for pattern in atomic_patterns)
        
        if has_atomics and not self._has_synchronization(kernel_code, backend):
            race_conditions.append("Atomic operations without proper synchronization")
        
        return race_conditions
    
    def _has_synchronization(self, kernel_code: str, backend: str) -> bool:
        """Check if kernel has any synchronization"""
        if backend.upper() == "CUDA":
            return "__syncthreads" in kernel_code or "__syncwarp" in kernel_code
        elif backend.upper() == "OpenCL":
            return "barrier" in kernel_code or "mem_fence" in kernel_code
        return False
    
    def get_suggestions(self, kernel_code: str, backend: str) -> List[str]:
        """Get suggestions for improving synchronization"""
        suggestions = []
        
        if backend.upper() == "CUDA":
            if "__shared__" in kernel_code and "__syncthreads()" not in kernel_code:
                suggestions.append("Add __syncthreads() after shared memory writes")
            
            if "threadIdx" in kernel_code and "__syncwarp()" not in kernel_code:
                suggestions.append("Consider __syncwarp() for warp-level synchronization")
        
        elif backend.upper() == "OpenCL":
            if "__local" in kernel_code and "barrier" not in kernel_code:
                suggestions.append("Add barrier(CLK_LOCAL_MEM_FENCE) after local memory writes")
        
        return suggestions

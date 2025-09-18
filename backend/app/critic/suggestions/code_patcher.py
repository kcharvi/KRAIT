"""
Code patcher for generating suggested code modifications
"""

import re
import ast
from typing import Dict, Any, List, Tuple, Optional
from ..critic_models import Suggestion, SeverityLevel
from ..parsers.parser_factory import ParserFactory


class CodePatcher:
    """Generate and apply code patches for kernel optimization suggestions"""
    
    def __init__(self):
        self.name = "code_patcher"
        self.description = "Generate and apply code patches for kernel optimization"
        self.parser_factory = ParserFactory()
        
        # Patch templates for common optimizations
        self.patch_templates = {
            "bounds_checking": {
                "cuda": {
                    "before": r"(\w+)\[(\w+)\]\s*=",
                    "after": r"if (\2 < size) {\n    \1[\2] =",
                    "closing": "\n}"
                },
                "opencl": {
                    "before": r"(\w+)\[(\w+)\]\s*=",
                    "after": r"if (\2 < size) {\n    \1[\2] =",
                    "closing": "\n}"
                }
            },
            "synchronization": {
                "cuda": {
                    "before": r"__shared__\s+.*?;\s*(.*?)(?=\n\s*\w)",
                    "after": r"\1\n__syncthreads();",
                    "closing": ""
                },
                "opencl": {
                    "before": r"__local\s+.*?;\s*(.*?)(?=\n\s*\w)",
                    "after": r"\1\nbarrier(CLK_LOCAL_MEM_FENCE);",
                    "closing": ""
                }
            },
            "vectorization": {
                "cuda": {
                    "before": r"float\s+(\w+)\s*=\s*(\w+)\[(\w+)\];",
                    "after": r"float4 \1 = ((float4*)\2)[\3/4];",
                    "closing": ""
                },
                "opencl": {
                    "before": r"float\s+(\w+)\s*=\s*(\w+)\[(\w+)\];",
                    "after": r"float4 \1 = vload4(\3/4, \2);",
                    "closing": ""
                }
            },
            "tiling": {
                "cuda": {
                    "before": r"for\s*\(\s*int\s+(\w+)\s*=\s*0\s*;\s*\1\s*<\s*(\w+)\s*;\s*\1\+\+\)",
                    "after": r"for (int \1 = 0; \1 < \2; \1 += TILE_SIZE)",
                    "closing": ""
                }
            }
        }
        
        # Code generation templates
        self.code_templates = {
            "cuda_bounds_check": """
// Add bounds checking
if (threadIdx.x < width && threadIdx.y < height) {
    // Original code here
    {original_code}
}""",
            "cuda_shared_memory": """
__shared__ float tile[{tile_size}][{tile_size}];
int tx = threadIdx.x, ty = threadIdx.y;

// Load data into shared memory
if (tx < {tile_size} && ty < {tile_size}) {
    tile[ty][tx] = data[ty * width + tx];
}
__syncthreads();

// Use shared memory for computation
{original_code}""",
            "cuda_vectorization": """
// Vectorized memory operations
float4 *vec_ptr = (float4*)&data[idx * 4];
float4 vec_data = *vec_ptr;  // Load 4 floats at once

// Process vectorized data
vec_data.x *= 2.0f;
vec_data.y *= 2.0f;
vec_data.z *= 2.0f;
vec_data.w *= 2.0f;

*vec_ptr = vec_data;  // Store 4 floats at once""",
            "triton_optimization": """
@triton.jit
def optimized_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data with bounds checking
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute
    output = x + y
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)"""
        }
    
    async def generate_patch(self, suggestion: Suggestion, 
                           original_code: str, 
                           backend: str) -> Dict[str, Any]:
        """Generate a code patch for a suggestion"""
        
        patch = {
            "suggestion_id": suggestion.title,
            "original_code": original_code,
            "patched_code": original_code,
            "changes": [],
            "confidence": 0.0,
            "applied": False
        }
        
        # Parse the original code
        parsed_kernel = await self.parser_factory.parse(original_code)
        if not parsed_kernel:
            patch["error"] = "Failed to parse original code"
            return patch
        
        # Generate patch based on suggestion type
        if "bounds" in suggestion.title.lower():
            patch = await self._generate_bounds_patch(suggestion, original_code, backend, parsed_kernel)
        elif "memory" in suggestion.title.lower():
            patch = await self._generate_memory_patch(suggestion, original_code, backend, parsed_kernel)
        elif "vectorization" in suggestion.title.lower():
            patch = await self._generate_vectorization_patch(suggestion, original_code, backend, parsed_kernel)
        elif "tiling" in suggestion.title.lower():
            patch = await self._generate_tiling_patch(suggestion, original_code, backend, parsed_kernel)
        elif "synchronization" in suggestion.title.lower():
            patch = await self._generate_sync_patch(suggestion, original_code, backend, parsed_kernel)
        else:
            patch = await self._generate_generic_patch(suggestion, original_code, backend, parsed_kernel)
        
        return patch
    
    async def apply_patch(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a patch to the original code"""
        
        if patch.get("applied", False):
            return {"success": True, "message": "Patch already applied"}
        
        try:
            # Validate the patched code
            validation_result = await self._validate_patch(patch)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "message": f"Patch validation failed: {validation_result['error']}"
                }
            
            # Apply the patch
            patch["applied"] = True
            patch["confidence"] = validation_result["confidence"]
            
            return {
                "success": True,
                "message": "Patch applied successfully",
                "confidence": validation_result["confidence"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to apply patch: {str(e)}"
            }
    
    async def _generate_bounds_patch(self, suggestion: Suggestion, 
                                   original_code: str, 
                                   backend: str, 
                                   parsed_kernel) -> Dict[str, Any]:
        """Generate bounds checking patch"""
        
        patch = {
            "suggestion_id": suggestion.title,
            "original_code": original_code,
            "patched_code": original_code,
            "changes": [],
            "confidence": 0.0,
            "applied": False
        }
        
        # Find array accesses that need bounds checking
        array_accesses = []
        for access in parsed_kernel.memory_accesses:
            if access.is_global and not self._has_bounds_check(access, parsed_kernel):
                array_accesses.append(access)
        
        if not array_accesses:
            patch["error"] = "No array accesses found that need bounds checking"
            return patch
        
        # Generate bounds checking code
        patched_code = original_code
        changes = []
        
        for access in array_accesses:
            line_num = access.line_number
            if line_num <= len(parsed_kernel.lines):
                original_line = parsed_kernel.lines[line_num - 1]
                
                # Generate bounds check
                if backend.upper() == "CUDA":
                    bounds_check = self._generate_cuda_bounds_check(access, parsed_kernel)
                elif backend.upper() == "OPENCL":
                    bounds_check = self._generate_opencl_bounds_check(access, parsed_kernel)
                else:
                    bounds_check = self._generate_generic_bounds_check(access, parsed_kernel)
                
                # Insert bounds check before the access
                new_line = f"{bounds_check}\n{original_line}"
                patched_code = patched_code.replace(original_line, new_line)
                
                changes.append({
                    "line": line_num,
                    "type": "bounds_check",
                    "original": original_line,
                    "modified": new_line
                })
        
        patch["patched_code"] = patched_code
        patch["changes"] = changes
        patch["confidence"] = 0.8 if changes else 0.0
        
        return patch
    
    async def _generate_memory_patch(self, suggestion: Suggestion, 
                                   original_code: str, 
                                   backend: str, 
                                   parsed_kernel) -> Dict[str, Any]:
        """Generate memory optimization patch"""
        
        patch = {
            "suggestion_id": suggestion.title,
            "original_code": original_code,
            "patched_code": original_code,
            "changes": [],
            "confidence": 0.0,
            "applied": False
        }
        
        # Check if shared memory is already used
        has_shared_memory = any(var.is_shared for var in parsed_kernel.variables)
        
        if not has_shared_memory and "shared" in suggestion.message.lower():
            # Add shared memory tiling
            if backend.upper() == "CUDA":
                template = self.code_templates["cuda_shared_memory"]
                tile_size = self._determine_optimal_tile_size(parsed_kernel)
                shared_mem_code = template.format(
                    tile_size=tile_size,
                    original_code="// Your computation here"
                )
                
                # Insert shared memory code
                patched_code = self._insert_shared_memory_code(original_code, shared_mem_code, backend)
                
                patch["patched_code"] = patched_code
                patch["changes"].append({
                    "type": "shared_memory",
                    "description": f"Added shared memory tiling with {tile_size}x{tile_size} tiles"
                })
                patch["confidence"] = 0.7
        
        return patch
    
    async def _generate_vectorization_patch(self, suggestion: Suggestion, 
                                          original_code: str, 
                                          backend: str, 
                                          parsed_kernel) -> Dict[str, Any]:
        """Generate vectorization patch"""
        
        patch = {
            "suggestion_id": suggestion.title,
            "original_code": original_code,
            "patched_code": original_code,
            "changes": [],
            "confidence": 0.0,
            "applied": False
        }
        
        # Find scalar operations that can be vectorized
        scalar_ops = self._find_scalar_operations(parsed_kernel)
        
        if scalar_ops:
            if backend.upper() == "CUDA":
                vectorized_code = self.code_templates["cuda_vectorization"]
            else:
                vectorized_code = "// Vectorized operations\n// Replace scalar operations with vectorized equivalents"
            
            # Replace scalar operations
            patched_code = self._replace_scalar_with_vector(original_code, scalar_ops, vectorized_code)
            
            patch["patched_code"] = patched_code
            patch["changes"].append({
                "type": "vectorization",
                "description": f"Vectorized {len(scalar_ops)} scalar operations"
            })
            patch["confidence"] = 0.6
        
        return patch
    
    async def _generate_tiling_patch(self, suggestion: Suggestion, 
                                   original_code: str, 
                                   backend: str, 
                                   parsed_kernel) -> Dict[str, Any]:
        """Generate tiling patch"""
        
        patch = {
            "suggestion_id": suggestion.title,
            "original_code": original_code,
            "patched_code": original_code,
            "changes": [],
            "confidence": 0.0,
            "applied": False
        }
        
        # Find loops that can be tiled
        tiling_opportunities = self._find_tiling_opportunities(parsed_kernel)
        
        if tiling_opportunities:
            patched_code = original_code
            changes = []
            
            for opportunity in tiling_opportunities:
                # Apply tiling to the loop
                tiled_loop = self._apply_loop_tiling(opportunity, backend)
                patched_code = patched_code.replace(opportunity["original"], tiled_loop)
                
                changes.append({
                    "type": "tiling",
                    "line": opportunity["line"],
                    "description": f"Applied tiling to loop with step {opportunity['tile_size']}"
                })
            
            patch["patched_code"] = patched_code
            patch["changes"] = changes
            patch["confidence"] = 0.7
        
        return patch
    
    async def _generate_sync_patch(self, suggestion: Suggestion, 
                                 original_code: str, 
                                 backend: str, 
                                 parsed_kernel) -> Dict[str, Any]:
        """Generate synchronization patch"""
        
        patch = {
            "suggestion_id": suggestion.title,
            "original_code": original_code,
            "patched_code": original_code,
            "changes": [],
            "confidence": 0.0,
            "applied": False
        }
        
        # Find shared memory usage without synchronization
        sync_points = self._find_missing_sync_points(parsed_kernel, backend)
        
        if sync_points:
            patched_code = original_code
            changes = []
            
            for sync_point in sync_points:
                sync_code = self._generate_sync_code(backend)
                # Insert synchronization after shared memory operations
                patched_code = self._insert_sync_code(patched_code, sync_point, sync_code)
                
                changes.append({
                    "type": "synchronization",
                    "line": sync_point["line"],
                    "description": "Added synchronization after shared memory operation"
                })
            
            patch["patched_code"] = patched_code
            patch["changes"] = changes
            patch["confidence"] = 0.9
        
        return patch
    
    async def _generate_generic_patch(self, suggestion: Suggestion, 
                                    original_code: str, 
                                    backend: str, 
                                    parsed_kernel) -> Dict[str, Any]:
        """Generate generic patch based on suggestion"""
        
        patch = {
            "suggestion_id": suggestion.title,
            "original_code": original_code,
            "patched_code": original_code,
            "changes": [],
            "confidence": 0.0,
            "applied": False
        }
        
        # Use the suggestion's code snippet if available
        if suggestion.code_snippet:
            patch["patched_code"] = suggestion.code_snippet
            patch["changes"].append({
                "type": "generic",
                "description": "Applied suggestion code snippet"
            })
            patch["confidence"] = 0.5
        
        return patch
    
    def _has_bounds_check(self, access, parsed_kernel) -> bool:
        """Check if memory access has bounds checking"""
        # Look for bounds checking in nearby lines
        start_line = max(0, access.line_number - 5)
        end_line = min(len(parsed_kernel.lines), access.line_number + 1)
        
        context_lines = parsed_kernel.lines[start_line:end_line]
        context_code = '\n'.join(context_lines)
        
        # Check for common bounds checking patterns
        bounds_patterns = [
            r'if\s*\(\s*[^)]*<\s*[^)]*\)',
            r'if\s*\(\s*[^)]*>=\s*[^)]*\)',
            r'if\s*\(\s*[^)]*<=\s*[^)]*\)'
        ]
        
        return any(re.search(pattern, context_code, re.IGNORECASE) for pattern in bounds_patterns)
    
    def _generate_cuda_bounds_check(self, access, parsed_kernel) -> str:
        """Generate CUDA-specific bounds check"""
        return f"if ({access.indexing_pattern} < size) {{"
    
    def _generate_opencl_bounds_check(self, access, parsed_kernel) -> str:
        """Generate OpenCL-specific bounds check"""
        return f"if ({access.indexing_pattern} < size) {{"
    
    def _generate_generic_bounds_check(self, access, parsed_kernel) -> str:
        """Generate generic bounds check"""
        return f"if ({access.indexing_pattern} < size) {{"
    
    def _determine_optimal_tile_size(self, parsed_kernel) -> int:
        """Determine optimal tile size based on kernel characteristics"""
        # Simple heuristic based on loop patterns
        for loop in parsed_kernel.loops:
            if loop.type == "for" and loop.step:
                try:
                    step_value = int(loop.step)
                    if step_value in [8, 16, 32, 64]:
                        return step_value
                except:
                    pass
        
        return 16  # Default tile size
    
    def _insert_shared_memory_code(self, original_code: str, shared_mem_code: str, backend: str) -> str:
        """Insert shared memory code into the kernel"""
        # Find the kernel function start
        if backend.upper() == "CUDA":
            kernel_pattern = r'(__global__\s+void\s+\w+\s*\([^)]*\)\s*\{)'
        elif backend.upper() == "OPENCL":
            kernel_pattern = r'(__kernel\s+void\s+\w+\s*\([^)]*\)\s*\{)'
        else:
            return original_code
        
        match = re.search(kernel_pattern, original_code)
        if match:
            kernel_start = match.end()
            return original_code[:kernel_start] + "\n" + shared_mem_code + "\n" + original_code[kernel_start:]
        
        return original_code
    
    def _find_scalar_operations(self, parsed_kernel) -> List[Dict[str, Any]]:
        """Find scalar operations that can be vectorized"""
        scalar_ops = []
        
        for i, line in enumerate(parsed_kernel.lines):
            # Look for scalar arithmetic operations
            if re.search(r'(\w+)\s*[+\-*/]\s*(\w+)', line):
                scalar_ops.append({
                    "line": i + 1,
                    "operation": line.strip(),
                    "type": "arithmetic"
                })
        
        return scalar_ops
    
    def _replace_scalar_with_vector(self, original_code: str, scalar_ops: List[Dict], vector_code: str) -> str:
        """Replace scalar operations with vectorized equivalents"""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated code transformation
        return original_code + "\n\n// Vectorized version:\n" + vector_code
    
    def _find_tiling_opportunities(self, parsed_kernel) -> List[Dict[str, Any]]:
        """Find loops that can be tiled"""
        opportunities = []
        
        for loop in parsed_kernel.loops:
            if loop.type == "for" and loop.step in ["1", "++", "+1"]:
                opportunities.append({
                    "line": loop.line_number,
                    "original": f"for (int {loop.variable} = {loop.start}; {loop.variable} < {loop.end}; {loop.variable}++)",
                    "tile_size": 16
                })
        
        return opportunities
    
    def _apply_loop_tiling(self, opportunity: Dict, backend: str) -> str:
        """Apply tiling to a loop"""
        tile_size = opportunity["tile_size"]
        return f"for (int {opportunity['original'].split()[2]} = {opportunity['original'].split()[4]}; {opportunity['original'].split()[2]} < {opportunity['original'].split()[6]}; {opportunity['original'].split()[2]} += {tile_size})"
    
    def _find_missing_sync_points(self, parsed_kernel, backend: str) -> List[Dict[str, Any]]:
        """Find places where synchronization is missing"""
        sync_points = []
        
        # Look for shared memory usage without synchronization
        for i, line in enumerate(parsed_kernel.lines):
            if "__shared__" in line or "__local__" in line:
                # Check if next few lines have synchronization
                has_sync = False
                for j in range(i + 1, min(i + 5, len(parsed_kernel.lines))):
                    if any(sync_word in parsed_kernel.lines[j] for sync_word in ["__syncthreads", "barrier"]):
                        has_sync = True
                        break
                
                if not has_sync:
                    sync_points.append({
                        "line": i + 1,
                        "type": "missing_sync"
                    })
        
        return sync_points
    
    def _generate_sync_code(self, backend: str) -> str:
        """Generate synchronization code for backend"""
        if backend.upper() == "CUDA":
            return "__syncthreads();"
        elif backend.upper() == "OPENCL":
            return "barrier(CLK_LOCAL_MEM_FENCE);"
        else:
            return "// Add appropriate synchronization"
    
    def _insert_sync_code(self, original_code: str, sync_point: Dict, sync_code: str) -> str:
        """Insert synchronization code at the specified point"""
        lines = original_code.split('\n')
        if sync_point["line"] < len(lines):
            lines.insert(sync_point["line"], sync_code)
        return '\n'.join(lines)
    
    async def _validate_patch(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a patch before applying it"""
        
        try:
            # Try to parse the patched code
            parsed_kernel = await self.parser_factory.parse(patch["patched_code"])
            
            if parsed_kernel:
                return {
                    "valid": True,
                    "confidence": 0.8,
                    "message": "Patch validation successful"
                }
            else:
                return {
                    "valid": False,
                    "error": "Failed to parse patched code",
                    "confidence": 0.0
                }
        
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "confidence": 0.0
            }

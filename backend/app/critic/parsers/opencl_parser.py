"""
OpenCL kernel parser
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from .base_parser import BaseParser, LanguageType, ParsedKernel, FunctionSignature, VariableDeclaration, LoopStructure, MemoryAccess, SynchronizationPoint


class OpenCLParser(BaseParser):
    """Parser for OpenCL kernel code"""
    
    def __init__(self):
        super().__init__(LanguageType.OPENCL)
        
        # Regex patterns for OpenCL parsing
        self.kernel_pattern = re.compile(r'__kernel\s+void\s+(\w+)\s*\([^)]*\)', re.MULTILINE)
        self.function_pattern = re.compile(r'(?:__kernel\s+)?(\w+)\s+(\w+)\s*\([^)]*\)', re.MULTILINE)
        self.variable_pattern = re.compile(r'(?:__local\s+|__global\s+|__constant\s+|__private\s+)?(?:const\s+)?(\w+(?:\s*\*\s*)*)\s+(\w+)(?:\[([^\]]+)\])?', re.MULTILINE)
        self.loop_pattern = re.compile(r'(for|while)\s*\([^)]*\)\s*{', re.MULTILINE)
        self.sync_pattern = re.compile(r'(barrier|mem_fence|read_mem_fence|write_mem_fence)\s*\([^)]*\)', re.MULTILINE)
        self.memory_access_pattern = re.compile(r'(\w+)\s*\[([^\]]+)\]', re.MULTILINE)
        self.work_item_pattern = re.compile(r'(get_global_id|get_local_id|get_group_id|get_local_size|get_global_size)\s*\([^)]*\)', re.MULTILINE)
    
    def detect_language(self, code: str) -> bool:
        """Detect if this is OpenCL code"""
        opencl_indicators = [
            '__kernel', '__global', '__local', '__constant', '__private',
            'get_global_id', 'get_local_id', 'get_group_id', 'barrier',
            'work_group_barrier', 'mem_fence', 'cl_mem'
        ]
        
        return any(indicator in code for indicator in opencl_indicators)
    
    async def parse(self, code: str) -> ParsedKernel:
        """Parse OpenCL kernel code"""
        lines = code.split('\n')
        
        # Extract basic information
        includes = self.extract_includes(code)
        defines = self.extract_defines(code)
        
        # Parse functions and kernels
        functions = self._parse_functions(code)
        
        # Parse variables
        variables = self._parse_variables(code)
        
        # Parse loops
        loops = self._parse_loops(code)
        
        # Parse memory accesses
        memory_accesses = self._parse_memory_accesses(code)
        
        # Parse synchronization points
        sync_points = self._parse_synchronization(code)
        
        return ParsedKernel(
            language=self.language,
            functions=functions,
            variables=variables,
            loops=loops,
            memory_accesses=memory_accesses,
            synchronization_points=sync_points,
            includes=includes,
            defines=defines,
            raw_code=code,
            lines=lines
        )
    
    def _parse_functions(self, code: str) -> List[FunctionSignature]:
        """Parse function signatures"""
        functions = []
        
        # Find kernel functions
        kernel_matches = self.kernel_pattern.finditer(code)
        for match in kernel_matches:
            func_name = match.group(1)
            line_num = code[:match.start()].count('\n') + 1
            
            # Extract parameters
            func_start = match.end() - 1
            paren_start = code.find('(', func_start)
            paren_end = code.find(')', paren_start)
            param_str = code[paren_start + 1:paren_end]
            
            parameters = self._parse_parameters(param_str)
            
            functions.append(FunctionSignature(
                name=func_name,
                return_type="void",
                parameters=parameters,
                is_kernel=True,
                line_number=line_num
            ))
        
        # Find regular functions
        func_matches = self.function_pattern.finditer(code)
        for match in func_matches:
            return_type = match.group(1)
            func_name = match.group(2)
            line_num = code[:match.start()].count('\n') + 1
            
            # Skip if already found as kernel
            if any(f.name == func_name for f in functions):
                continue
            
            # Extract parameters
            func_start = match.end() - 1
            paren_start = code.find('(', func_start)
            paren_end = code.find(')', paren_start)
            param_str = code[paren_start + 1:paren_end]
            
            parameters = self._parse_parameters(param_str)
            
            functions.append(FunctionSignature(
                name=func_name,
                return_type=return_type,
                parameters=parameters,
                is_kernel=False,
                line_number=line_num
            ))
        
        return functions
    
    def _parse_parameters(self, param_str: str) -> List[VariableDeclaration]:
        """Parse function parameters"""
        parameters = []
        if not param_str.strip():
            return parameters
        
        # Split by comma, but be careful with template parameters
        param_list = self._split_parameters(param_str)
        
        for param in param_list:
            param = param.strip()
            if not param:
                continue
            
            # Parse parameter type and name
            parts = param.split()
            if len(parts) >= 2:
                param_type = ' '.join(parts[:-1])
                param_name = parts[-1]
                
                # Check for pointer and qualifiers
                is_pointer = '*' in param_type
                is_const = 'const' in param_type
                is_global = '__global' in param_type
                is_local = '__local' in param_type
                is_constant = '__constant' in param_type
                is_private = '__private' in param_type
                
                parameters.append(VariableDeclaration(
                    name=param_name,
                    type=param_type,
                    is_pointer=is_pointer,
                    is_const=is_const,
                    is_global=is_global,
                    is_local=is_local,
                    line_number=0
                ))
        
        return parameters
    
    def _split_parameters(self, param_str: str) -> List[str]:
        """Split parameters by comma, respecting template brackets"""
        params = []
        current = ""
        depth = 0
        
        for char in param_str:
            if char == '<':
                depth += 1
            elif char == '>':
                depth -= 1
            elif char == ',' and depth == 0:
                params.append(current.strip())
                current = ""
                continue
            current += char
        
        if current.strip():
            params.append(current.strip())
        
        return params
    
    def _parse_variables(self, code: str) -> List[VariableDeclaration]:
        """Parse variable declarations"""
        variables = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip comments and preprocessor directives
            if line.startswith('//') or line.startswith('/*') or line.startswith('#'):
                continue
            
            # Look for variable declarations
            var_matches = self.variable_pattern.finditer(line)
            for match in var_matches:
                var_type = match.group(1).strip()
                var_name = match.group(2).strip()
                array_size = match.group(3) if match.group(3) else None
                
                # Determine variable qualifiers
                is_local = '__local' in line
                is_global = '__global' in line
                is_constant = '__constant' in line
                is_private = '__private' in line
                is_const = 'const' in line
                is_pointer = '*' in var_type
                
                variables.append(VariableDeclaration(
                    name=var_name,
                    type=var_type,
                    is_pointer=is_pointer,
                    is_const=is_const,
                    is_shared=is_local,  # OpenCL local memory is similar to shared memory
                    is_global=is_global,
                    is_local=is_private,
                    array_size=array_size,
                    line_number=i + 1
                ))
        
        return variables
    
    def _parse_loops(self, code: str) -> List[LoopStructure]:
        """Parse loop structures"""
        loops = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip comments
            if line.startswith('//') or line.startswith('/*'):
                continue
            
            # Check for for loops
            for_match = re.search(r'for\s*\([^)]*\)', line)
            if for_match:
                loop_content = for_match.group(0)
                
                # Extract loop variable and bounds (simplified)
                var_match = re.search(r'(\w+)\s*=\s*([^;]+)', loop_content)
                if var_match:
                    var_name = var_match.group(1)
                    start_expr = var_match.group(2)
                    
                    # Extract end condition
                    end_match = re.search(r';\s*[^<>=!]+([<>=!]+)\s*([^;]+)', loop_content)
                    if end_match:
                        end_expr = end_match.group(2)
                        
                        # Extract step
                        step_match = re.search(r';\s*[^)]*;\s*([^)]+)', loop_content)
                        step_expr = step_match.group(1) if step_match else "1"
                        
                        loops.append(LoopStructure(
                            type="for",
                            variable=var_name,
                            start=start_expr,
                            end=end_expr,
                            step=step_expr,
                            line_number=i + 1
                        ))
            
            # Check for while loops
            while_match = re.search(r'while\s*\([^)]*\)', line)
            if while_match:
                condition = while_match.group(0)[6:-1]  # Remove 'while(' and ')'
                
                loops.append(LoopStructure(
                    type="while",
                    variable="",
                    start="",
                    end=condition,
                    line_number=i + 1
                ))
        
        return loops
    
    def _parse_memory_accesses(self, code: str) -> List[MemoryAccess]:
        """Parse memory access patterns"""
        accesses = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            # Skip comments
            if line.strip().startswith('//') or line.strip().startswith('/*'):
                continue
            
            # Find array accesses
            mem_matches = self.memory_access_pattern.finditer(line)
            for match in mem_matches:
                var_name = match.group(1)
                index_expr = match.group(2)
                
                # Determine access type
                is_write = '=' in line and var_name in line[:line.find('=')]
                access_type = "write" if is_write else "read"
                
                # Determine memory space
                is_local = '__local' in line or 'local' in var_name.lower()
                is_global = '__global' in line or 'global' in var_name.lower()
                
                accesses.append(MemoryAccess(
                    variable=var_name,
                    access_type=access_type,
                    is_global=is_global,
                    is_shared=is_local,
                    is_local=not (is_local or is_global),
                    indexing_pattern=index_expr,
                    line_number=i + 1
                ))
        
        return accesses
    
    def _parse_synchronization(self, code: str) -> List[SynchronizationPoint]:
        """Parse synchronization points"""
        sync_points = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            sync_matches = self.sync_pattern.finditer(line)
            for match in sync_matches:
                sync_type = match.group(1)
                
                sync_points.append(SynchronizationPoint(
                    type=sync_type,
                    line_number=i + 1
                ))
        
        return sync_points

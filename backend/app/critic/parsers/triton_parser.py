"""
Triton kernel parser
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from .base_parser import BaseParser, LanguageType, ParsedKernel, FunctionSignature, VariableDeclaration, LoopStructure, MemoryAccess, SynchronizationPoint


class TritonParser(BaseParser):
    """Parser for Triton kernel code"""
    
    def __init__(self):
        super().__init__(LanguageType.TRITON)
        
        # Regex patterns for Triton parsing
        self.kernel_pattern = re.compile(r'@triton\.jit\s+def\s+(\w+)\s*\([^)]*\):', re.MULTILINE)
        self.function_pattern = re.compile(r'def\s+(\w+)\s*\([^)]*\):', re.MULTILINE)
        self.variable_pattern = re.compile(r'(\w+)\s*=\s*([^=\n]+)', re.MULTILINE)
        self.loop_pattern = re.compile(r'for\s+(\w+)\s+in\s+([^:]+):', re.MULTILINE)
        self.memory_access_pattern = re.compile(r'(\w+)\s*\[([^\]]+)\]', re.MULTILINE)
        self.triton_ops = ['tl.load', 'tl.store', 'tl.dot', 'tl.sum', 'tl.max', 'tl.min']
    
    def detect_language(self, code: str) -> bool:
        """Detect if this is Triton code"""
        triton_indicators = [
            '@triton.jit', 'import triton', 'tl.load', 'tl.store',
            'tl.dot', 'tl.sum', 'tl.max', 'tl.min', 'triton.language'
        ]
        
        return any(indicator in code for indicator in triton_indicators)
    
    async def parse(self, code: str) -> ParsedKernel:
        """Parse Triton kernel code"""
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
        
        # Parse synchronization points (Triton handles this automatically)
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
        
        # Find Triton kernels
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
                return_type="None",  # Triton kernels typically return None
                parameters=parameters,
                is_kernel=True,
                line_number=line_num
            ))
        
        # Find regular functions
        func_matches = self.function_pattern.finditer(code)
        for match in func_matches:
            func_name = match.group(1)
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
                return_type="Any",  # Python functions can return anything
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
        
        param_list = param_str.split(',')
        
        for param in param_list:
            param = param.strip()
            if not param:
                continue
            
            # Parse parameter type and name
            if ':' in param:
                name, param_type = param.split(':', 1)
                name = name.strip()
                param_type = param_type.strip()
            else:
                name = param
                param_type = "Any"
            
            parameters.append(VariableDeclaration(
                name=name,
                type=param_type,
                is_pointer=False,
                is_const=False,
                line_number=0
            ))
        
        return parameters
    
    def _parse_variables(self, code: str) -> List[VariableDeclaration]:
        """Parse variable declarations"""
        variables = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip comments and imports
            if line.startswith('#') or line.startswith('import') or line.startswith('from'):
                continue
            
            # Look for variable assignments
            var_matches = self.variable_pattern.finditer(line)
            for match in var_matches:
                var_name = match.group(1)
                var_value = match.group(2)
                
                # Determine variable type from value
                var_type = self._infer_type(var_value)
                
                variables.append(VariableDeclaration(
                    name=var_name,
                    type=var_type,
                    is_pointer=False,
                    is_const=False,
                    is_shared=False,
                    is_global=False,
                    is_local=True,
                    line_number=i + 1
                ))
        
        return variables
    
    def _infer_type(self, value: str) -> str:
        """Infer variable type from assignment value"""
        value = value.strip()
        
        if value.startswith('tl.'):
            return "triton.language"
        elif value.startswith('[') and value.endswith(']'):
            return "List"
        elif value.startswith('(') and value.endswith(')'):
            return "Tuple"
        elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return "int"
        elif '.' in value and value.replace('.', '').isdigit():
            return "float"
        elif value.startswith('"') or value.startswith("'"):
            return "str"
        elif value in ['True', 'False']:
            return "bool"
        else:
            return "Any"
    
    def _parse_loops(self, code: str) -> List[LoopStructure]:
        """Parse loop structures"""
        loops = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip comments
            if line.startswith('#'):
                continue
            
            # Check for for loops
            for_match = re.search(r'for\s+(\w+)\s+in\s+([^:]+):', line)
            if for_match:
                var_name = for_match.group(1)
                iterable = for_match.group(2)
                
                loops.append(LoopStructure(
                    type="for",
                    variable=var_name,
                    start="0",
                    end=iterable,
                    line_number=i + 1
                ))
        
        return loops
    
    def _parse_memory_accesses(self, code: str) -> List[MemoryAccess]:
        """Parse memory access patterns"""
        accesses = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            # Skip comments
            if line.strip().startswith('#'):
                continue
            
            # Find array accesses
            mem_matches = self.memory_access_pattern.finditer(line)
            for match in mem_matches:
                var_name = match.group(1)
                index_expr = match.group(2)
                
                # Determine access type
                is_write = '=' in line and var_name in line[:line.find('=')]
                access_type = "write" if is_write else "read"
                
                # Triton memory accesses are typically global
                is_global = 'tl.load' in line or 'tl.store' in line
                
                accesses.append(MemoryAccess(
                    variable=var_name,
                    access_type=access_type,
                    is_global=is_global,
                    is_shared=False,
                    is_local=not is_global,
                    indexing_pattern=index_expr,
                    line_number=i + 1
                ))
        
        return accesses
    
    def _parse_synchronization(self, code: str) -> List[SynchronizationPoint]:
        """Parse synchronization points (Triton handles this automatically)"""
        # Triton kernels are automatically synchronized
        # No explicit synchronization points needed
        return []

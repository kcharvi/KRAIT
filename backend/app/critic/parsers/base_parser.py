"""
Base parser class for kernel code analysis
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class LanguageType(str, Enum):
    """Supported kernel languages"""
    CUDA = "cuda"
    CXX = "cxx"
    TRITON = "triton"
    OPENCL = "opencl"
    PYTHON = "python"


@dataclass
class VariableDeclaration:
    """Variable declaration information"""
    name: str
    type: str
    is_pointer: bool = False
    is_const: bool = False
    is_shared: bool = False
    is_global: bool = False
    is_local: bool = False
    array_size: Optional[str] = None
    line_number: int = 0


@dataclass
class FunctionSignature:
    """Function signature information"""
    name: str
    return_type: str
    parameters: List[VariableDeclaration]
    is_kernel: bool = False
    line_number: int = 0


@dataclass
class LoopStructure:
    """Loop structure information"""
    type: str  # "for", "while", "do_while"
    variable: str
    start: str
    end: str
    step: Optional[str] = None
    is_nested: bool = False
    parent_loop: Optional[int] = None
    line_number: int = 0


@dataclass
class MemoryAccess:
    """Memory access pattern"""
    variable: str
    access_type: str  # "read", "write", "read_write"
    is_global: bool = False
    is_shared: bool = False
    is_local: bool = False
    indexing_pattern: Optional[str] = None
    line_number: int = 0


@dataclass
class SynchronizationPoint:
    """Synchronization primitive usage"""
    type: str  # "__syncthreads", "__syncwarp", "barrier"
    line_number: int = 0
    scope: Optional[str] = None


@dataclass
class ParsedKernel:
    """Complete parsed kernel information"""
    language: LanguageType
    functions: List[FunctionSignature]
    variables: List[VariableDeclaration]
    loops: List[LoopStructure]
    memory_accesses: List[MemoryAccess]
    synchronization_points: List[SynchronizationPoint]
    includes: List[str]
    defines: List[Tuple[str, str]]
    raw_code: str
    lines: List[str]


class BaseParser(ABC):
    """Base class for kernel code parsers"""
    
    def __init__(self, language: LanguageType):
        self.language = language
    
    @abstractmethod
    async def parse(self, code: str) -> ParsedKernel:
        """Parse kernel code and return structured information"""
        pass
    
    @abstractmethod
    def detect_language(self, code: str) -> bool:
        """Detect if this parser can handle the given code"""
        pass
    
    def extract_comments(self, code: str) -> List[Tuple[int, str]]:
        """Extract comments from code with line numbers"""
        comments = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('//') or line.startswith('#'):
                comments.append((i + 1, line))
            elif '/*' in line and '*/' in line:
                comments.append((i + 1, line))
        
        return comments
    
    def extract_includes(self, code: str) -> List[str]:
        """Extract include statements"""
        includes = []
        lines = code.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('#include'):
                includes.append(line)
        
        return includes
    
    def extract_defines(self, code: str) -> List[Tuple[str, str]]:
        """Extract #define statements"""
        defines = []
        lines = code.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('#define'):
                parts = line.split(None, 2)
                if len(parts) >= 2:
                    name = parts[1]
                    value = parts[2] if len(parts) > 2 else ""
                    defines.append((name, value))
        
        return defines
    
    def count_lines_of_code(self, code: str) -> int:
        """Count non-empty, non-comment lines"""
        lines = code.split('\n')
        loc = 0
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('//') and not line.startswith('/*') and not line.startswith('#'):
                loc += 1
        
        return loc

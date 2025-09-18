"""
Parser factory for automatic language detection and parsing
"""

from typing import Optional
from .base_parser import BaseParser, ParsedKernel
from .cuda_parser import CUDAParser
from .triton_parser import TritonParser
from .opencl_parser import OpenCLParser


class ParserFactory:
    """Factory for creating appropriate parsers based on code content"""
    
    def __init__(self):
        self.parsers = [
            CUDAParser(),
            TritonParser(),
            OpenCLParser(),
        ]
    
    def get_parser(self, code: str) -> Optional[BaseParser]:
        """Get the appropriate parser for the given code"""
        for parser in self.parsers:
            if parser.detect_language(code):
                return parser
        return None
    
    async def parse(self, code: str) -> Optional[ParsedKernel]:
        """Parse code using the appropriate parser"""
        parser = self.get_parser(code)
        if parser:
            return await parser.parse(code)
        return None
    
    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages"""
        return [parser.language.value for parser in self.parsers]

"""
LLM-based advanced correctness analysis for kernel code
"""

import asyncio
from typing import Dict, Any, List, Optional
from ..llm.gemini_client import GeminiClient
from ..config import settings
from .critic_models import Suggestion, SeverityLevel


class LLMCorrectnessAnalyzer:
    """Advanced LLM-based correctness analysis"""
    
    def __init__(self):
        self.gemini_client = GeminiClient()
    
    async def analyze_control_flow(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Analyze control flow patterns and potential issues"""
        prompt = f"""
Analyze the control flow patterns in this {backend} kernel code for {hardware}:

```{backend.lower()}
{kernel_code}
```

Focus on:
1. Loop structures and their termination conditions
2. Conditional branches and their coverage
3. Early returns and their impact on execution
4. Nested control structures and complexity
5. Potential infinite loops or unreachable code
6. Thread divergence patterns in parallel execution

Provide a concise analysis with specific line numbers and actionable recommendations.
Format as JSON with: {{"issues": [], "recommendations": [], "complexity_score": 0-10}}
"""
        
        try:
            response = await self.gemini_client.generate_content_async(prompt)
            return self._parse_llm_response(response, "control_flow")
        except Exception as e:
            return {"error": f"Control flow analysis failed: {str(e)}"}
    
    async def analyze_context_dependent_bounds(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Analyze context-dependent bounds checking"""
        prompt = f"""
Analyze bounds checking in this {backend} kernel code for {hardware}:

```{backend.lower()}
{kernel_code}
```

Focus on:
1. Array access patterns and their bounds validation
2. Dynamic index calculations and their safety
3. Thread/block dimension usage in array indexing
4. Conditional bounds checking based on runtime values
5. Potential out-of-bounds access scenarios
6. Hardware-specific memory layout considerations

Provide specific line numbers and suggest improvements.
Format as JSON with: {{"unsafe_accesses": [], "missing_checks": [], "suggestions": []}}
"""
        
        try:
            response = await self.gemini_client.generate_content_async(prompt)
            return self._parse_llm_response(response, "context_bounds")
        except Exception as e:
            return {"error": f"Context-dependent bounds analysis failed: {str(e)}"}
    
    async def analyze_dynamic_memory_allocation(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Analyze dynamic memory allocation patterns"""
        prompt = f"""
Analyze dynamic memory allocation in this {backend} kernel code for {hardware}:

```{backend.lower()}
{kernel_code}
```

Focus on:
1. Memory allocation patterns and their lifecycle
2. Pointer arithmetic and potential overflows
3. Memory deallocation and leak prevention
4. Shared memory usage and synchronization
5. Stack vs heap allocation decisions
6. Hardware-specific memory constraints

Provide specific recommendations for memory safety.
Format as JSON with: {{"allocation_patterns": [], "potential_leaks": [], "optimizations": []}}
"""
        
        try:
            response = await self.gemini_client.generate_content_async(prompt)
            return self._parse_llm_response(response, "dynamic_memory")
        except Exception as e:
            return {"error": f"Dynamic memory analysis failed: {str(e)}"}
    
    async def analyze_cross_function_dependencies(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Analyze cross-function dependencies and interactions"""
        prompt = f"""
Analyze cross-function dependencies in this {backend} kernel code for {hardware}:

```{backend.lower()}
{kernel_code}
```

Focus on:
1. Function call patterns and their side effects
2. Global variable usage and thread safety
3. Parameter passing and data flow
4. Function call overhead and optimization opportunities
5. Recursive patterns and stack usage
6. Library function dependencies and their safety

Provide analysis of function interactions and optimization suggestions.
Format as JSON with: {{"function_calls": [], "dependencies": [], "optimizations": []}}
"""
        
        try:
            response = await self.gemini_client.generate_content_async(prompt)
            return self._parse_llm_response(response, "cross_function")
        except Exception as e:
            return {"error": f"Cross-function analysis failed: {str(e)}"}
    
    def _parse_llm_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Parse LLM response and extract structured data"""
        try:
            # For type safety analysis, we want a clean summary format
            if analysis_type in ["hardware_specific_types", "backend_specific_types", "cross_function_types"]:
                return self._parse_type_safety_response(response, analysis_type)
            
            # Try to extract JSON from the response for other analyses
            import json
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # Fallback: return the raw response
                return {
                    "raw_response": response,
                    "analysis_type": analysis_type,
                    "parsed": False
                }
        except Exception as e:
            return {
                "raw_response": response,
                "analysis_type": analysis_type,
                "parsed": False,
                "error": f"Failed to parse response: {str(e)}"
            }
    
    def _parse_type_safety_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Parse type safety LLM response into clean summary format"""
        try:
            # Extract key information from the verbose response
            summary = {
                "analysis_type": analysis_type,
                "summary": "",
                "recommendations": [],
                "critical_issues": [],
                "warnings": [],
                "optimizations": []
            }
            
            # Extract summary (first meaningful sentence)
            lines = response.split('\n')
            summary_text = ""
            for line in lines:
                if line.strip() and not line.startswith('#') and not line.startswith('*') and len(line.strip()) > 20:
                    summary_text = line.strip()
                    break
            summary["summary"] = summary_text[:150] + "..." if len(summary_text) > 150 else summary_text
            
            # Extract actionable recommendations (look for specific patterns)
            recommendations = []
            warnings = []
            optimizations = []
            
            # Look for specific actionable patterns
            for line in lines:
                line_clean = line.strip()
                if not line_clean or line_clean.startswith('#') or line_clean.startswith('*'):
                    continue
                    
                line_lower = line_clean.lower()
                
                # Extract actionable recommendations
                if any(phrase in line_lower for phrase in [
                    "use ", "switch to", "replace", "implement", "add", "enable", "consider using",
                    "change", "modify", "update", "apply", "install", "configure"
                ]):
                    # Clean up the recommendation
                    rec = line_clean
                    if rec.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                        rec = rec[3:].strip()
                    elif rec.startswith(('-', '*', '•')):
                        rec = rec[1:].strip()
                    
                    # Remove common prefixes
                    prefixes_to_remove = [
                        "**", "*", "•", "- ", "1. ", "2. ", "3. ", "4. ", "5. ",
                        "Key Changes:", "Recommendations:", "Suggestions:", "Actions:"
                    ]
                    for prefix in prefixes_to_remove:
                        if rec.startswith(prefix):
                            rec = rec[len(prefix):].strip()
                    
                    if len(rec) > 10 and len(rec) < 200:
                        recommendations.append(rec)
                
                # Extract warnings
                elif any(phrase in line_lower for phrase in [
                    "warning", "caution", "be careful", "avoid", "don't", "should not",
                    "potential issue", "risk", "problem", "concern"
                ]):
                    if len(line_clean) > 10 and len(line_clean) < 200:
                        warnings.append(line_clean)
                
                # Extract optimizations
                elif any(phrase in line_lower for phrase in [
                    "optimize", "performance", "faster", "efficient", "vector", "tensor",
                    "fp16", "fp32", "simd", "parallel", "coalesced", "alignment"
                ]):
                    if len(line_clean) > 10 and len(line_clean) < 200:
                        optimizations.append(line_clean)
            
            # Limit results to avoid overwhelming the user
            summary["recommendations"] = recommendations[:8]  # Top 8 actionable items
            summary["warnings"] = warnings[:5]  # Top 5 warnings
            summary["optimizations"] = optimizations[:6]  # Top 6 optimizations
            
            return summary
            
        except Exception as e:
            return {
                "analysis_type": analysis_type,
                "summary": "Analysis completed with parsing issues",
                "recommendations": [],
                "critical_issues": [],
                "warnings": [],
                "optimizations": [],
                "error": f"Failed to parse response: {str(e)}"
            }
    
    async def generate_llm_suggestions(self, analysis_results: Dict[str, Any]) -> List[Suggestion]:
        """Convert LLM analysis results to suggestions"""
        suggestions = []
        
        for analysis_type, result in analysis_results.items():
            # Handle case where result is a string (error message)
            if isinstance(result, str):
                suggestions.append(Suggestion(
                    severity=SeverityLevel.MEDIUM,
                    category="llm_analysis",
                    title=f"LLM {analysis_type.replace('_', ' ').title()} Analysis",
                    message=f"Analysis completed with issues: {result}"
                ))
                continue
            
            # Handle case where result is a dict with error
            if isinstance(result, dict) and "error" in result:
                suggestions.append(Suggestion(
                    severity=SeverityLevel.MEDIUM,
                    category="llm_analysis",
                    title=f"LLM {analysis_type.replace('_', ' ').title()} Analysis",
                    message=f"Analysis completed with issues: {result['error']}"
                ))
                continue
            
            # Generate suggestions based on analysis type
            if analysis_type == "control_flow":
                if "issues" in result and result["issues"]:
                    suggestions.append(Suggestion(
                        severity=SeverityLevel.HIGH,
                        category="control_flow",
                        title="Control Flow Issues",
                        message=f"Found {len(result['issues'])} control flow issues: {', '.join(result['issues'][:3])}"
                    ))
            
            elif analysis_type == "context_bounds":
                if "unsafe_accesses" in result and result["unsafe_accesses"]:
                    suggestions.append(Suggestion(
                        severity=SeverityLevel.CRITICAL,
                        category="memory_safety",
                        title="Unsafe Memory Accesses",
                        message=f"Found {len(result['unsafe_accesses'])} potentially unsafe memory accesses"
                    ))
            
            elif analysis_type == "dynamic_memory":
                if "potential_leaks" in result and result["potential_leaks"]:
                    suggestions.append(Suggestion(
                        severity=SeverityLevel.HIGH,
                        category="memory_management",
                        title="Memory Leak Risks",
                        message=f"Identified {len(result['potential_leaks'])} potential memory leak scenarios"
                    ))
            
            elif analysis_type == "cross_function":
                if "dependencies" in result and result["dependencies"]:
                    suggestions.append(Suggestion(
                        severity=SeverityLevel.MEDIUM,
                        category="architecture",
                        title="Function Dependencies",
                        message=f"Complex function dependencies detected: {len(result['dependencies'])} interactions"
                    ))
        
        return suggestions
    
    async def analyze_hardware_specific_types(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Analyze hardware-specific type usage and optimizations"""
        print(f"🎯 Hardware-specific type analysis - Starting for {hardware} {backend}")
        print(f"📝 Kernel code length: {len(kernel_code)}")
        
        prompt = f"""
        Analyze the type usage in this {backend} kernel code for {hardware} hardware:
        
        ```{backend.lower()}
        {kernel_code}
        ```
        
        Focus on hardware-specific type optimizations:
        1. **Precision requirements** for {hardware}
        2. **Memory alignment** requirements
        3. **Tensor Core usage** (if applicable)
        4. **Vector types** and SIMD operations
        5. **Mixed precision** patterns
        6. **Performance implications** of type choices
        
        Provide specific recommendations for {hardware} optimization.
        """
        
        try:
            print(f"🤖 Calling Gemini API for hardware analysis...")
            response = await self.gemini_client.generate_content_async(prompt)
            print(f"✅ Gemini response received, length: {len(response) if response else 0}")
            print(f"📄 Response preview: {response[:200] if response else 'None'}...")
            
            parsed = self._parse_llm_response(response, "hardware_specific_types")
            print(f"✅ Hardware analysis parsed successfully: {type(parsed)}")
            return parsed
        except Exception as e:
            print(f"❌ Hardware analysis failed: {str(e)}")
            return {"error": f"Hardware analysis failed: {str(e)}"}
    
    async def analyze_backend_specific_types(self, kernel_code: str, hardware: str, backend: str) -> Dict[str, Any]:
        """Analyze backend-specific type usage and API compliance"""
        print(f"🔧 Backend-specific type analysis - Starting for {hardware} {backend}")
        print(f"📝 Kernel code length: {len(kernel_code)}")
        
        prompt = f"""
        Analyze the type usage in this {backend} kernel code for API compliance:
        
        ```{backend.lower()}
        {kernel_code}
        ```
        
        Focus on backend-specific type requirements:
        1. **API type signatures** for {backend}
        2. **Template instantiation** correctness
        3. **Memory management** types
        4. **Error handling** types
        5. **Platform-specific** type requirements
        6. **Performance implications** of type choices
        
        Provide specific recommendations for {backend} optimization.
        """
        
        try:
            print(f"🤖 Calling Gemini API for backend analysis...")
            response = await self.gemini_client.generate_content_async(prompt)
            print(f"✅ Gemini response received, length: {len(response) if response else 0}")
            print(f"📄 Response preview: {response[:200] if response else 'None'}...")
            
            parsed = self._parse_llm_response(response, "backend_specific_types")
            print(f"✅ Backend analysis parsed successfully: {type(parsed)}")
            return parsed
        except Exception as e:
            print(f"❌ Backend analysis failed: {str(e)}")
            return {"error": f"Backend analysis failed: {str(e)}"}

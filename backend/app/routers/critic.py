"""
Critic Agent API endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
import asyncio
import time

from ..critic.critic_models import (
    CriticAnalysisRequest, CriticAnalysis, BatchAnalysisRequest, 
    BatchAnalysisResponse, CheckType, CriticHealthResponse, SeverityLevel
)
from ..critic.analyzer import KernelAnalyzer
from ..critic.checks import (
    BoundsChecker, SynchronizationChecker, MemorySafetyChecker, 
    TypeSafetyChecker, PerformanceChecker
)
from ..critic.llm_analysis import LLMCorrectnessAnalyzer

router = APIRouter()

# Initialize analyzer
analyzer = KernelAnalyzer()

# Initialize checkers
checkers = {
    "bounds": BoundsChecker(),
    "synchronization": SynchronizationChecker(),
    "memory_safety": MemorySafetyChecker(),
    "type_safety": TypeSafetyChecker(),
    "performance": PerformanceChecker(),
}


@router.post("/analyze", response_model=CriticAnalysis)
async def analyze_kernel(request: CriticAnalysisRequest):
    """
    Analyze a single kernel for correctness, performance, and optimization opportunities
    """
    try:
        analysis = await analyzer.analyze_kernel(request)
        return analysis
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/batch-analyze", response_model=BatchAnalysisResponse)
async def batch_analyze_kernels(request: BatchAnalysisRequest):
    """
    Analyze multiple kernels in batch
    """
    start_time = time.time()
    results = []
    errors = []
    
    try:
        if request.parallel:
            # Run analyses in parallel
            tasks = [analyzer.analyze_kernel(kernel_req) for kernel_req in request.kernels]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Separate successful results from errors
            successful_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    errors.append({
                        "index": i,
                        "error": str(result)
                    })
                else:
                    successful_results.append(result)
            
            results = successful_results
        else:
            # Run analyses sequentially
            for i, kernel_req in enumerate(request.kernels):
                try:
                    result = await analyzer.analyze_kernel(kernel_req)
                    results.append(result)
                except Exception as e:
                    errors.append({
                        "index": i,
                        "error": str(e)
                    })
        
        total_time_ms = (time.time() - start_time) * 1000
        
        return BatchAnalysisResponse(
            results=results,
            total_time_ms=total_time_ms,
            success_count=len(results),
            error_count=len(errors),
            errors=errors
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )


@router.get("/checks", response_model=List[CheckType])
async def get_available_checks():
    """
    Get list of available check types and their configurations
    """
    try:
        check_types = []
        
        for checker_name, checker in checkers.items():
            check_types.append(CheckType(
                name=checker.name,
                category=checker_name,
                description=checker.description,
                severity_levels=[SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH, SeverityLevel.CRITICAL],
                enabled=True
            ))
        
        return check_types
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get available checks: {str(e)}"
        )


@router.get("/health", response_model=CriticHealthResponse)
async def health_check():
    """
    Health check for the critic service
    """
    try:
        return CriticHealthResponse(
            status="healthy",
            version="1.0.0",
            available_checks=[
                CheckType(
                    name=checker.name,
                    category=name,
                    description=checker.description,
                    severity_levels=[SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH, SeverityLevel.CRITICAL],
                    enabled=True
                )
                for name, checker in checkers.items()
            ],
            supported_hardware=analyzer.supported_hardware,
            supported_backends=analyzer.supported_backends
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )


@router.post("/analyze/quick", response_model=CriticAnalysis)
async def quick_analyze(request: CriticAnalysisRequest):
    """
    Quick analysis with only essential checks for fast feedback
    """
    try:
        # Create a simplified analyzer for quick analysis
        quick_analyzer = KernelAnalyzer()
        
        # Override the analysis to run only essential checks
        start_time = time.time()
        analysis_id = f"quick_{int(time.time())}"
        
        # Run only bounds checking and basic performance analysis
        correctness = await quick_analyzer._analyze_correctness(request.kernel_code, request.backend)
        performance = await quick_analyzer._analyze_performance(request.kernel_code, request.hardware, request.backend)
        
        # Generate only high-priority suggestions
        suggestions = []
        if correctness.issues:
            suggestions.append({
                "severity": "high",
                "category": "correctness",
                "title": "Critical Issues Found",
                "message": f"Found {len(correctness.issues)} critical issues",
                "impact": "High - may cause crashes or incorrect results"
            })
        
        analysis_time_ms = (time.time() - start_time) * 1000
        
        return CriticAnalysis(
            analysis_id=analysis_id,
            kernel_code=request.kernel_code,
            hardware_config=request.hardware,
            backend=request.backend,
            correctness=correctness,
            performance=performance,
            suggestions=suggestions,
            overall_score=quick_analyzer._calculate_overall_score(correctness, performance),
            analysis_time_ms=analysis_time_ms,
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%S")
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Quick analysis failed: {str(e)}"
        )


@router.get("/stats")
async def get_analysis_stats():
    """
    Get statistics about the critic service
    """
    try:
        return {
            "total_checks": len(checkers),
            "supported_hardware": len(analyzer.supported_hardware),
            "supported_backends": len(analyzer.supported_backends),
            "service_uptime": "N/A",  # Could be implemented with proper tracking
            "version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


@router.post("/llm-analyze")
async def llm_analyze(request: CriticAnalysisRequest):
    """
    Perform specific LLM-based analysis on kernel code
    """
    try:
        llm_analyzer = LLMCorrectnessAnalyzer()
        
        # Run all LLM analyses
        analysis_results = {}
        
        # Control flow analysis
        control_flow = await llm_analyzer.analyze_control_flow(
            request.kernel_code, request.hardware, request.backend
        )
        analysis_results["control_flow"] = control_flow
        
        # Context-dependent bounds analysis
        context_bounds = await llm_analyzer.analyze_context_dependent_bounds(
            request.kernel_code, request.hardware, request.backend
        )
        analysis_results["context_bounds"] = context_bounds
        
        # Dynamic memory allocation analysis
        dynamic_memory = await llm_analyzer.analyze_dynamic_memory_allocation(
            request.kernel_code, request.hardware, request.backend
        )
        analysis_results["dynamic_memory"] = dynamic_memory
        
        # Cross-function analysis
        cross_function = await llm_analyzer.analyze_cross_function_dependencies(
            request.kernel_code, request.hardware, request.backend
        )
        analysis_results["cross_function"] = cross_function
        
        # Generate suggestions
        suggestions = await llm_analyzer.generate_llm_suggestions(analysis_results)
        
        return {
            "analysis_results": analysis_results,
            "suggestions": suggestions,
            "analysis_type": "llm_advanced",
            "hardware": request.hardware,
            "backend": request.backend
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM analysis failed: {str(e)}"
        )


@router.post("/llm-memory-analysis")
async def llm_memory_analysis(request: CriticAnalysisRequest):
    """
    Perform LLM-based analysis focused specifically on memory safety
    """
    try:
        llm_analyzer = LLMCorrectnessAnalyzer()
        
        # Run only memory-related LLM analyses
        analysis_results = {}
        
        # Context-dependent bounds analysis (memory-focused)
        context_bounds = await llm_analyzer.analyze_context_dependent_bounds(
            request.kernel_code, request.hardware, request.backend
        )
        analysis_results["context_bounds"] = context_bounds
        
        # Dynamic memory allocation analysis
        dynamic_memory = await llm_analyzer.analyze_dynamic_memory_allocation(
            request.kernel_code, request.hardware, request.backend
        )
        analysis_results["dynamic_memory"] = dynamic_memory
        
        # Generate memory-focused suggestions
        suggestions = await llm_analyzer.generate_llm_suggestions(analysis_results)
        
        return {
            "analysis_results": analysis_results,
            "suggestions": suggestions,
            "analysis_type": "llm_memory_safety",
            "hardware": request.hardware,
            "backend": request.backend
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM memory analysis failed: {str(e)}"
        )


@router.post("/llm-type-safety-analysis")
async def llm_type_safety_analysis(request: CriticAnalysisRequest):
    """
    Perform LLM-based type safety analysis
    """
    print(f"🔍 Type Safety LLM Analysis - Starting")
    print(f"📝 Request: kernel_code length={len(request.kernel_code)}, hardware={request.hardware}, backend={request.backend}")
    
    try:
        print(f"📦 Importing LLMCorrectnessAnalyzer...")
        from ..critic.llm_analysis import LLMCorrectnessAnalyzer
        print(f"✅ LLMCorrectnessAnalyzer imported successfully")
        
        print(f"🏗️ Initializing LLM analyzer...")
        llm_analyzer = LLMCorrectnessAnalyzer()
        print(f"✅ LLM analyzer initialized successfully")
        
        # Type safety specific LLM analysis
        analysis_results = {}
        
        print(f"🎯 Starting hardware-specific type analysis...")
        try:
            hardware_analysis = await llm_analyzer.analyze_hardware_specific_types(
                request.kernel_code, request.hardware, request.backend
            )
            print(f"✅ Hardware analysis completed: {type(hardware_analysis)}")
            print(f"📊 Hardware analysis keys: {hardware_analysis.keys() if isinstance(hardware_analysis, dict) else 'Not a dict'}")
            analysis_results["hardware_types"] = hardware_analysis
        except Exception as e:
            print(f"❌ Hardware analysis failed: {str(e)}")
            analysis_results["hardware_types"] = {"error": str(e)}
        
        print(f"🔧 Starting backend-specific type analysis...")
        try:
            backend_analysis = await llm_analyzer.analyze_backend_specific_types(
                request.kernel_code, request.hardware, request.backend
            )
            print(f"✅ Backend analysis completed: {type(backend_analysis)}")
            print(f"📊 Backend analysis keys: {backend_analysis.keys() if isinstance(backend_analysis, dict) else 'Not a dict'}")
            analysis_results["backend_types"] = backend_analysis
        except Exception as e:
            print(f"❌ Backend analysis failed: {str(e)}")
            analysis_results["backend_types"] = {"error": str(e)}
        
        print(f"🔗 Starting cross-function type analysis...")
        try:
            cross_function = await llm_analyzer.analyze_cross_function_dependencies(
                request.kernel_code, request.hardware, request.backend
            )
            print(f"✅ Cross-function analysis completed: {type(cross_function)}")
            print(f"📊 Cross-function analysis keys: {cross_function.keys() if isinstance(cross_function, dict) else 'Not a dict'}")
            analysis_results["cross_function_types"] = cross_function
        except Exception as e:
            print(f"❌ Cross-function analysis failed: {str(e)}")
            analysis_results["cross_function_types"] = {"error": str(e)}
        
        print(f"💡 Generating type safety suggestions...")
        try:
            suggestions = await llm_analyzer.generate_llm_suggestions(analysis_results)
            print(f"✅ Suggestions generated: {len(suggestions) if isinstance(suggestions, list) else 'Not a list'}")
            print(f"📊 Suggestions type: {type(suggestions)}")
        except Exception as e:
            print(f"❌ Suggestions generation failed: {str(e)}")
            suggestions = []
        
        result = {
            "analysis_results": analysis_results,
            "suggestions": suggestions,
            "analysis_type": "llm_type_safety",
            "hardware": request.hardware,
            "backend": request.backend
        }
        
        print(f"🎉 Type Safety LLM Analysis completed successfully!")
        print(f"📊 Final result keys: {result.keys()}")
        return result
        
    except Exception as e:
        print(f"💥 Type Safety LLM Analysis failed with exception: {str(e)}")
        print(f"🔍 Exception type: {type(e)}")
        import traceback
        print(f"📋 Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"LLM type safety analysis failed: {str(e)}"
        )

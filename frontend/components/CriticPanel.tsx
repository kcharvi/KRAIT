"use client";

import { useState, useEffect } from "react";
import {
    CheckCircle,
    AlertTriangle,
    XCircle,
    Info,
    Loader2,
    ChevronDown,
    ChevronRight,
    Copy,
    Download,
    RefreshCw,
} from "lucide-react";

interface AnalysisResult {
    analysis_id: string;
    status: "analyzed" | "error";
    correctness: {
        status: "likely_correct" | "potential_issues" | "failed_checks";
        score: number;
        issues: Array<{
            name: string;
            status: "pass" | "fail" | "warning" | "skip";
            message: string;
            details?: any;
        }>;
        checks: Array<{
            name: string;
            status: "pass" | "fail" | "warning" | "skip";
            message: string;
            details?: any;
        }>;
    };
    performance: {
        flops_total: number;
        estimated_runtime_ms: number;
        bound: "memory" | "compute" | "unknown";
        shared_mem_per_block_bytes: number;
        tiling_detected: boolean;
        vectorization_detected: boolean;
        tensor_core_usage_detected: boolean;
        loop_unrolling_detected: boolean;
    };
    suggestions: Array<{
        severity: "low" | "medium" | "high" | "critical";
        category: string;
        title: string;
        message: string;
        code_snippet?: string;
    }>;
    score: number;
    analysis_time_ms: number;
    generated_at: string;
}

interface CriticPanelProps {
    kernelCode: string;
    hardware: string;
    backend: string;
    isAnalyzing: boolean;
    onAnalysisComplete?: (result: AnalysisResult) => void;
}

export default function CriticPanel({
    kernelCode,
    hardware,
    backend,
    isAnalyzing,
    onAnalysisComplete,
}: CriticPanelProps) {
    const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
    const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(["overview"]));
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [llmAnalysisResults, setLlmAnalysisResults] = useState<any>(null);
    const [isLlmAnalyzing, setIsLlmAnalyzing] = useState(false);
    const [llmAnalysisError, setLlmAnalysisError] = useState<string | null>(null);
    const [typeSafetyLlmResults, setTypeSafetyLlmResults] = useState<any>(null);
    const [isTypeSafetyLlmAnalyzing, setIsTypeSafetyLlmAnalyzing] = useState(false);
    const [typeSafetyLlmError, setTypeSafetyLlmError] = useState<string | null>(null);

    // GPU execution state
    const [realMetrics, setRealMetrics] = useState<any>(null);
    const [isExecuting, setIsExecuting] = useState(false);
    const [executionProvider, setExecutionProvider] = useState("github_colab");
    const [executionError, setExecutionError] = useState<string | null>(null);

    // Debug: Track analysisResult changes
    useEffect(() => {
        console.log("analysisResult state changed:", analysisResult);
    }, [analysisResult]);

    const analyzeKernel = async () => {
        if (!kernelCode.trim()) return;

        setIsLoading(true);
        setError(null);

        try {
            const response = await fetch(
                `${process.env.BACKEND_URL || "http://localhost:8000"}/api/v1/critic/analyze`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        kernel_code: kernelCode,
                        hardware: hardware,
                        backend: backend,
                        analysis_mode: "full",
                        use_llm_review: false,
                    }),
                }
            );

            if (!response.ok) {
                throw new Error(`Analysis failed: ${response.status}`);
            }

            const result = await response.json();
            console.log("Analysis result received:", result);
            console.log("Analysis result structure:", {
                hasCorrectness: !!result.correctness,
                hasPerformance: !!result.performance,
                correctnessChecks: result.correctness?.checks?.length || 0,
                correctnessIssues: result.correctness?.issues?.length || 0,
                overallScore: result.overall_score,
            });
            console.log("Correctness checks:", result.correctness?.checks);
            console.log("Correctness issues:", result.correctness?.issues);
            console.log("Setting analysis result...");
            setAnalysisResult(result);
            console.log("Analysis result set, calling onAnalysisComplete");
            onAnalysisComplete?.(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Analysis failed");
        } finally {
            setIsLoading(false);
        }
    };

    const toggleSection = (section: string) => {
        const newExpanded = new Set(expandedSections);
        if (newExpanded.has(section)) {
            newExpanded.delete(section);
        } else {
            newExpanded.add(section);
        }
        setExpandedSections(newExpanded);
    };

    const getStatusIcon = (status: string) => {
        switch (status) {
            case "pass":
                return <CheckCircle className="w-4 h-4 text-green-500" />;
            case "fail":
                return <XCircle className="w-4 h-4 text-red-500" />;
            case "warning":
                return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
            case "skip":
                return <Info className="w-4 h-4 text-gray-400" />;
            default:
                return null; // Don't show icon for unknown status
        }
    };

    const getSeverityIcon = (severity: string) => {
        switch (severity) {
            case "critical":
                return <XCircle className="w-4 h-4 text-red-500" />;
            case "high":
                return <AlertTriangle className="w-4 h-4 text-orange-500" />;
            case "medium":
                return <Info className="w-4 h-4 text-yellow-500" />;
            case "low":
                return <CheckCircle className="w-4 h-4 text-green-500" />;
            default:
                return null; // Don't show icon for unknown severity
        }
    };

    const getSeverityColor = (severity: string) => {
        switch (severity) {
            case "critical":
                return "bg-red-50 border-red-200 text-red-800";
            case "high":
                return "bg-orange-50 border-orange-200 text-orange-800";
            case "medium":
                return "bg-yellow-50 border-yellow-200 text-yellow-800";
            case "low":
                return "bg-green-50 border-green-200 text-green-800";
            default:
                return "bg-gray-50 border-gray-200 text-gray-800";
        }
    };

    const getCorrectnessStatusColor = (status: string) => {
        switch (status) {
            case "likely_correct":
                return "text-green-600";
            case "potential_issues":
                return "text-yellow-600";
            case "failed_checks":
                return "text-red-600";
            default:
                return "text-gray-600";
        }
    };

    const copyToClipboard = (text: string) => {
        navigator.clipboard.writeText(text);
    };

    const runLlmAnalysis = async () => {
        if (!kernelCode.trim()) return;

        setIsLlmAnalyzing(true);
        setLlmAnalysisError(null);

        try {
            const response = await fetch(
                `${
                    process.env.BACKEND_URL || "http://localhost:8000"
                }/api/v1/critic/llm-memory-analysis`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        kernel_code: kernelCode,
                        hardware: hardware,
                        backend: backend,
                    }),
                }
            );

            if (!response.ok) {
                throw new Error(`LLM analysis failed: ${response.status}`);
            }

            const result = await response.json();
            console.log("LLM Analysis result received:", result);
            console.log("LLM Analysis results:", result.analysis_results);
            console.log("LLM Suggestions:", result.suggestions);
            setLlmAnalysisResults(result);
            setLlmAnalysisError(null);
        } catch (err) {
            setLlmAnalysisError(err instanceof Error ? err.message : "LLM analysis failed");
        } finally {
            setIsLlmAnalyzing(false);
        }
    };

    const runTypeSafetyLlmAnalysis = async () => {
        if (!kernelCode.trim()) return;

        setIsTypeSafetyLlmAnalyzing(true);
        setTypeSafetyLlmError(null);

        try {
            const response = await fetch(
                `${
                    process.env.BACKEND_URL || "http://localhost:8000"
                }/api/v1/critic/llm-type-safety-analysis`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        kernel_code: kernelCode,
                        hardware: hardware,
                        backend: backend,
                    }),
                }
            );

            if (!response.ok) {
                throw new Error(`Type safety LLM analysis failed: ${response.status}`);
            }

            const result = await response.json();
            console.log("Type Safety LLM Analysis result received:", result);
            setTypeSafetyLlmResults(result);
            setTypeSafetyLlmError(null);
        } catch (err) {
            setTypeSafetyLlmError(
                err instanceof Error ? err.message : "Type safety LLM analysis failed"
            );
        } finally {
            setIsTypeSafetyLlmAnalyzing(false);
        }
    };

    const executeOnGPU = async () => {
        if (!kernelCode.trim()) {
            alert("No kernel code to execute");
            return;
        }

        console.log("🚀 Starting GPU execution...", {
            kernelLength: kernelCode.length,
            hardware,
            provider: executionProvider,
            timestamp: new Date().toISOString(),
        });

        setIsExecuting(true);
        setExecutionError(null);
        setRealMetrics(null);

        try {
            const requestBody = {
                kernel_code: kernelCode,
                hardware: hardware,
                provider: executionProvider,
                timeout: 600, // Increased to 10 minutes for GitHub-Colab cycle
            };

            // Call backend directly instead of through Next.js API route
            const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
            const endpoint = `${backendUrl}/api/v1/gpu/execute-kernel`;

            console.log("📤 Sending request to:", endpoint);
            console.log("📤 Request body:", {
                ...requestBody,
                kernel_code: kernelCode.substring(0, 100) + "...",
            });

            const response = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(requestBody),
            });

            console.log("📊 Response status:", response.status);
            console.log("📊 Response headers:", Object.fromEntries(response.headers.entries()));
            console.log("📊 Response ok:", response.ok);

            if (!response.ok) {
                let errorData;
                try {
                    const responseText = await response.text();
                    console.log("❌ Error response text:", responseText);

                    // Check if it's HTML (like a 404 page)
                    if (responseText.includes("<!DOCTYPE") || responseText.includes("<html")) {
                        throw new Error(
                            `Backend server not running or endpoint not found (${response.status}). Please ensure the backend is running on port 8000.`
                        );
                    }

                    errorData = JSON.parse(responseText);
                } catch (parseError) {
                    console.error("❌ Failed to parse error response:", parseError);
                    throw new Error(
                        `HTTP error! status: ${response.status}. Response was not valid JSON.`
                    );
                }

                throw new Error(
                    errorData.error || errorData.detail || `HTTP error! status: ${response.status}`
                );
            }

            const result = await response.json();
            console.log("✅ GPU execution result:", result);
            console.log("📊 Result success:", result.success);
            console.log("📊 Result metrics:", result.metrics);
            console.log("📊 Result error:", result.error);
            console.log("📊 Result provider:", result.provider);

            if (result.success) {
                setRealMetrics(result.metrics);
                setExecutionError(null);
                console.log("🎉 GPU execution successful!");
            } else {
                const errorMsg = result.error || "GPU execution failed";
                console.error("❌ GPU execution failed:", errorMsg);
                setExecutionError(errorMsg);
            }
        } catch (error) {
            console.error("💥 GPU execution error:", error);
            const errorMessage = error instanceof Error ? error.message : "GPU execution failed";
            setExecutionError(errorMessage);
        } finally {
            setIsExecuting(false);
            console.log("🏁 GPU execution finished");
        }
    };

    const exportResults = () => {
        if (!analysisResult) return;

        const dataStr = JSON.stringify(analysisResult, null, 2);
        const dataBlob = new Blob([dataStr], { type: "application/json" });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `critic-analysis-${analysisResult.analysis_id}.json`;
        link.click();
        URL.revokeObjectURL(url);
    };

    if (isLoading) {
        return (
            <div className="h-full flex flex-col bg-white overflow-hidden">
                <div className="flex-1 flex items-center justify-center overflow-y-auto scrollbar-hide">
                    <div className="text-center">
                        <Loader2 className="w-8 h-8 animate-spin text-primary-500 mx-auto mb-2" />
                        <p className="text-sm text-gray-600">Analyzing kernel...</p>
                    </div>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="h-full flex flex-col bg-white overflow-hidden">
                <div className="flex-1 flex items-center justify-center overflow-y-auto scrollbar-hide">
                    <div className="text-center">
                        <XCircle className="w-8 h-8 text-red-500 mx-auto mb-2" />
                        <p className="text-sm text-red-600 mb-2">Analysis failed</p>
                        <p className="text-xs text-gray-500 mb-4">{error}</p>
                        <button
                            onClick={analyzeKernel}
                            className="px-3 py-1 text-xs bg-primary-500 text-white rounded hover:bg-primary-600"
                        >
                            Retry
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    if (!analysisResult) {
        return (
            <div className="h-full flex flex-col bg-white overflow-hidden">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
                    <h3 className="text-sm font-semibold text-gray-900">Critic and Analysis </h3>
                </div>

                {/* Content */}
                <div className="flex-1 flex items-center justify-center overflow-y-auto scrollbar-hide">
                    <div className="text-center">
                        <div className="mx-auto mb-4 h-10 w-10 rounded-full bg-gray-50 text-gray-600 flex items-center justify-center">
                            👁️
                        </div>
                        <h2 className="text-gray-700 mb-2">Ready to Analyze</h2>
                        <p className="text-sm text-gray-500 mb-4">
                            {kernelCode.trim()
                                ? "Click the analyze button to start analysis"
                                : "Generate a kernel first to enable analysis"}
                        </p>
                        {kernelCode.trim() && (
                            <button
                                onClick={analyzeKernel}
                                className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed"
                                disabled={isLoading}
                            >
                                {isLoading ? (
                                    <div className="flex items-center gap-2">
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        Analyzing...
                                    </div>
                                ) : (
                                    "Analyze Kernel"
                                )}
                            </button>
                        )}
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col bg-white overflow-hidden min-h-0">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
                <div className="flex items-center gap-2">
                    <h3 className="text-lg font-semibold text-gray-900">Critic and Analysis </h3>
                    <div
                        className={`px-2 py-1 text-xs rounded-full ${getCorrectnessStatusColor(
                            analysisResult.correctness.status
                        )}`}
                    >
                        {analysisResult.correctness.status.replace("_", " ").toUpperCase()}
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={analyzeKernel}
                        className="p-1 text-gray-400 hover:text-gray-600"
                        title="Refresh Analysis"
                    >
                        <RefreshCw className="w-4 h-4" />
                    </button>
                    <button
                        onClick={exportResults}
                        className="p-1 text-gray-400 hover:text-gray-600"
                        title="Export Results"
                    >
                        <Download className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto scrollbar-hide min-h-0">
                {/* Overview Section */}
                <div className="border-b border-gray-200">
                    <button
                        onClick={() => toggleSection("overview")}
                        className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50"
                    >
                        <span className="font-medium text-gray-900">Overview</span>
                        {expandedSections.has("overview") ? (
                            <ChevronDown className="w-4 h-4 text-gray-400" />
                        ) : (
                            <ChevronRight className="w-4 h-4 text-gray-400" />
                        )}
                    </button>
                    {expandedSections.has("overview") && (
                        <div className="px-4 pb-4 space-y-3">
                            <div className="grid grid-cols-2 gap-4">
                                <div className="text-center p-3 bg-gray-50 rounded-lg">
                                    <div className="text-2xl font-bold text-primary-600">
                                        {analysisResult.score}
                                    </div>
                                    <div className="text-xs text-gray-600">Overall Score</div>
                                </div>
                                <div className="text-center p-3 bg-gray-50 rounded-lg">
                                    <div className="text-2xl font-bold text-blue-600">
                                        {analysisResult.performance.flops_total?.toFixed(0) ||
                                            "N/A"}
                                    </div>
                                    <div className="text-xs text-gray-600">Total FLOPs</div>
                                </div>
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="text-center p-3 bg-gray-50 rounded-lg">
                                    <div className="text-lg font-semibold text-gray-700">
                                        {analysisResult.performance.estimated_runtime_ms?.toFixed(
                                            2
                                        ) || "N/A"}
                                        ms
                                    </div>
                                    <div className="text-xs text-gray-600">Est. Runtime</div>
                                </div>
                                <div className="text-center p-3 bg-gray-50 rounded-lg">
                                    <div className="text-lg font-semibold text-gray-700">
                                        {analysisResult.performance.bound || "N/A"}
                                    </div>
                                    <div className="text-xs text-gray-600">Bound Type</div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Correctness Section */}
                <div className="border-b border-gray-200">
                    <button
                        onClick={() => toggleSection("correctness")}
                        className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50"
                    >
                        <span className="font-medium text-gray-900">
                            Correctness ({analysisResult.correctness.issues.length} issues)
                        </span>
                        {expandedSections.has("correctness") ? (
                            <ChevronDown className="w-4 h-4 text-gray-400" />
                        ) : (
                            <ChevronRight className="w-4 h-4 text-gray-400" />
                        )}
                    </button>
                    {expandedSections.has("correctness") && (
                        <div className="px-4 pb-4 space-y-3">
                            {analysisResult.correctness.issues.length === 0 ? (
                                <div className="text-center py-4 text-gray-500">
                                    <CheckCircle className="w-8 h-8 mx-auto mb-2 text-green-500" />
                                    <p className="text-sm">No correctness issues found</p>
                                </div>
                            ) : (
                                (() => {
                                    console.log(
                                        "Rendering correctness issues:",
                                        analysisResult.correctness.issues
                                    );
                                    console.log(
                                        "Issues count:",
                                        analysisResult.correctness.issues.length
                                    );
                                    console.log(
                                        "Issues content:",
                                        analysisResult.correctness.issues
                                    );
                                    return analysisResult.correctness.issues
                                        .filter(
                                            (issue) =>
                                                issue.name &&
                                                issue.message &&
                                                issue.name.trim() !== "" &&
                                                issue.message.trim() !== ""
                                        )
                                        .map((issue, index) => {
                                            return (
                                                <div
                                                    key={index}
                                                    className="p-3 border border-gray-200 rounded-lg"
                                                >
                                                    <div className="flex items-start gap-2">
                                                        {getStatusIcon(issue.status)}
                                                        <div className="flex-1">
                                                            <div className="font-medium text-sm text-gray-900">
                                                                {issue.name}
                                                            </div>
                                                            <div className="text-sm text-gray-600 mt-1">
                                                                {issue.message}
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            );
                                        });
                                })()
                            )}

                            {/* Detailed per-check info */}
                            <div className="space-y-2">
                                {analysisResult.correctness.checks.filter(
                                    (check) =>
                                        check.details && Object.keys(check.details).length > 0
                                ).length === 0 ? (
                                    <div className="text-sm text-gray-500 italic p-3 border border-gray-200 rounded-lg">
                                        No detailed check information available. All checks
                                        completed without detailed results.
                                    </div>
                                ) : null}
                                {analysisResult.correctness.checks.map((check, idx) => {
                                    // Skip rendering if check has no meaningful content
                                    if (!check.details || Object.keys(check.details).length === 0) {
                                        return null;
                                    }

                                    return (
                                        <div
                                            key={idx}
                                            className="border border-gray-200 rounded-lg"
                                        >
                                            <button
                                                onClick={() => toggleSection(`check-${idx}`)}
                                                className="w-full flex items-center justify-between p-3 text-left hover:bg-gray-50"
                                            >
                                                <div className="flex items-center gap-2">
                                                    <span className="text-sm font-medium text-gray-900">
                                                        {check.name.replace(/_/g, " ")}
                                                    </span>
                                                    <span
                                                        className={`px-2 py-0.5 text-xs rounded-full ${
                                                            check.status === "pass"
                                                                ? "bg-green-100 text-green-700"
                                                                : check.status === "warning"
                                                                ? "bg-yellow-100 text-yellow-700"
                                                                : check.status === "fail"
                                                                ? "bg-red-100 text-red-700"
                                                                : "bg-gray-100 text-gray-700"
                                                        }`}
                                                    >
                                                        {check.status.toUpperCase()}
                                                    </span>
                                                </div>
                                                {expandedSections.has(`check-${idx}`) ? (
                                                    <ChevronDown className="w-4 h-4 text-gray-400" />
                                                ) : (
                                                    <ChevronRight className="w-4 h-4 text-gray-400" />
                                                )}
                                            </button>
                                            {expandedSections.has(`check-${idx}`) && (
                                                <div className="px-3 pb-3 text-sm text-gray-700 space-y-2">
                                                    <div className="text-gray-800">
                                                        {check.message}
                                                    </div>
                                                    {(() => {
                                                        return check.details ? (
                                                            <div className="grid grid-cols-1 gap-2">
                                                                {/* Special handling for bounds and synchronization checkers */}
                                                                {check.name.includes("bounds") ? (
                                                                    <div className="space-y-2">
                                                                        <div className="flex flex-wrap gap-2">
                                                                            {Array.isArray(
                                                                                (check as any)
                                                                                    .details
                                                                                    ?.patterns_found
                                                                            ) &&
                                                                                (
                                                                                    check as any
                                                                                ).details.patterns_found.map(
                                                                                    (
                                                                                        p: any,
                                                                                        i: number
                                                                                    ) => (
                                                                                        <span
                                                                                            key={i}
                                                                                            className="px-2 py-0.5 text-xs rounded bg-blue-50 text-blue-700 border border-blue-200"
                                                                                        >
                                                                                            {String(
                                                                                                p
                                                                                            )}
                                                                                        </span>
                                                                                    )
                                                                                )}
                                                                        </div>
                                                                        <div className="text-xs text-gray-600">
                                                                            Unsafe accesses:{" "}
                                                                            {(check as any).details
                                                                                ?.unsafe_accesses ??
                                                                                0}{" "}
                                                                            /{" "}
                                                                            {(check as any).details
                                                                                ?.total_memory_accesses ??
                                                                                0}
                                                                        </div>
                                                                        {Array.isArray(
                                                                            (check as any).details
                                                                                ?.issues
                                                                        ) &&
                                                                            (check as any).details
                                                                                .issues.length >
                                                                                0 && (
                                                                                <div className="bg-red-50 text-red-800 border border-red-200 rounded p-2 text-xs">
                                                                                    <div className="font-medium mb-1">
                                                                                        Detected
                                                                                        Issues
                                                                                    </div>
                                                                                    <ul className="list-disc pl-5 space-y-0.5">
                                                                                        {(
                                                                                            check as any
                                                                                        ).details.issues.map(
                                                                                            (
                                                                                                it: any,
                                                                                                ii: number
                                                                                            ) => (
                                                                                                <li
                                                                                                    key={
                                                                                                        ii
                                                                                                    }
                                                                                                >
                                                                                                    {String(
                                                                                                        it
                                                                                                    )}
                                                                                                </li>
                                                                                            )
                                                                                        )}
                                                                                    </ul>
                                                                                </div>
                                                                            )}
                                                                        {Array.isArray(
                                                                            (check as any).details
                                                                                ?.suggestions
                                                                        ) &&
                                                                            (check as any).details
                                                                                .suggestions
                                                                                .length > 0 && (
                                                                                <div className="bg-yellow-50 text-yellow-800 border border-yellow-200 rounded p-2 text-xs">
                                                                                    <div className="font-medium mb-1">
                                                                                        Suggestions
                                                                                    </div>
                                                                                    <ul className="list-disc pl-5 space-y-0.5">
                                                                                        {(
                                                                                            check as any
                                                                                        ).details.suggestions.map(
                                                                                            (
                                                                                                s: any,
                                                                                                si: number
                                                                                            ) => (
                                                                                                <li
                                                                                                    key={
                                                                                                        si
                                                                                                    }
                                                                                                >
                                                                                                    {String(
                                                                                                        s
                                                                                                    )}
                                                                                                </li>
                                                                                            )
                                                                                        )}
                                                                                    </ul>
                                                                                </div>
                                                                            )}
                                                                    </div>
                                                                ) : check.name.includes(
                                                                      "synchronization"
                                                                  ) ? (
                                                                    <div className="space-y-2">
                                                                        <div className="flex flex-wrap gap-2">
                                                                            {Array.isArray(
                                                                                (check as any)
                                                                                    .details
                                                                                    ?.sync_types
                                                                            ) &&
                                                                                (
                                                                                    check as any
                                                                                ).details.sync_types.map(
                                                                                    (
                                                                                        sync: any,
                                                                                        i: number
                                                                                    ) => (
                                                                                        <span
                                                                                            key={i}
                                                                                            className="px-2 py-0.5 text-xs rounded bg-green-50 text-green-700 border border-green-200"
                                                                                        >
                                                                                            {String(
                                                                                                sync
                                                                                            )}
                                                                                        </span>
                                                                                    )
                                                                                )}
                                                                        </div>
                                                                        <div className="text-xs text-gray-600">
                                                                            Shared memory usage:{" "}
                                                                            {(check as any).details
                                                                                ?.shared_memory_usage
                                                                                ? "Yes"
                                                                                : "No"}
                                                                        </div>
                                                                        <div className="text-xs text-gray-600">
                                                                            Synchronization
                                                                            barriers:{" "}
                                                                            {(check as any).details
                                                                                ?.sync_types
                                                                                ?.length ?? 0}
                                                                        </div>
                                                                        {Array.isArray(
                                                                            (check as any).details
                                                                                ?.sync_locations
                                                                        ) &&
                                                                            (check as any).details
                                                                                .sync_locations
                                                                                .length > 0 && (
                                                                                <div className="bg-blue-50 text-blue-800 border border-blue-200 rounded p-2 text-xs">
                                                                                    <div className="font-medium mb-1">
                                                                                        Sync
                                                                                        Locations
                                                                                    </div>
                                                                                    <ul className="list-disc pl-5 space-y-0.5">
                                                                                        {(
                                                                                            check as any
                                                                                        ).details.sync_locations.map(
                                                                                            (
                                                                                                loc: any,
                                                                                                li: number
                                                                                            ) => (
                                                                                                <li
                                                                                                    key={
                                                                                                        li
                                                                                                    }
                                                                                                >
                                                                                                    {String(
                                                                                                        loc.type
                                                                                                    )}{" "}
                                                                                                    at
                                                                                                    line{" "}
                                                                                                    {String(
                                                                                                        loc.line
                                                                                                    )}
                                                                                                </li>
                                                                                            )
                                                                                        )}
                                                                                    </ul>
                                                                                </div>
                                                                            )}
                                                                        {Array.isArray(
                                                                            (check as any).details
                                                                                ?.potential_race_conditions
                                                                        ) &&
                                                                            (check as any).details
                                                                                .potential_race_conditions
                                                                                .length > 0 && (
                                                                                <div className="bg-red-50 text-red-800 border border-red-200 rounded p-2 text-xs">
                                                                                    <div className="font-medium mb-1">
                                                                                        Race
                                                                                        Conditions
                                                                                    </div>
                                                                                    <ul className="list-disc pl-5 space-y-0.5">
                                                                                        {(
                                                                                            check as any
                                                                                        ).details.potential_race_conditions.map(
                                                                                            (
                                                                                                race: any,
                                                                                                ri: number
                                                                                            ) => (
                                                                                                <li
                                                                                                    key={
                                                                                                        ri
                                                                                                    }
                                                                                                >
                                                                                                    {String(
                                                                                                        race
                                                                                                    )}
                                                                                                </li>
                                                                                            )
                                                                                        )}
                                                                                    </ul>
                                                                                </div>
                                                                            )}
                                                                    </div>
                                                                ) : check.name.includes(
                                                                      "memory_safety"
                                                                  ) ? (
                                                                    <div className="space-y-2">
                                                                        {Array.isArray(
                                                                            (check as any).details
                                                                                ?.unsafe_patterns
                                                                        ) &&
                                                                            (check as any).details
                                                                                .unsafe_patterns
                                                                                .length > 0 && (
                                                                                <div className="bg-red-50 text-red-800 border border-red-200 rounded p-2 text-xs">
                                                                                    <div className="font-medium mb-1">
                                                                                        Unsafe
                                                                                        Patterns
                                                                                    </div>
                                                                                    <ul className="list-disc pl-5 space-y-0.5">
                                                                                        {(
                                                                                            check as any
                                                                                        ).details.unsafe_patterns.map(
                                                                                            (
                                                                                                p: any,
                                                                                                pi: number
                                                                                            ) => (
                                                                                                <li
                                                                                                    key={
                                                                                                        pi
                                                                                                    }
                                                                                                >
                                                                                                    {String(
                                                                                                        p.description ||
                                                                                                            p.type ||
                                                                                                            p
                                                                                                    )}
                                                                                                </li>
                                                                                            )
                                                                                        )}
                                                                                    </ul>
                                                                                </div>
                                                                            )}
                                                                        {Array.isArray(
                                                                            (check as any).details
                                                                                ?.warnings
                                                                        ) &&
                                                                            (check as any).details
                                                                                .warnings.length >
                                                                                0 && (
                                                                                <div className="bg-yellow-50 text-yellow-800 border border-yellow-200 rounded p-2 text-xs">
                                                                                    <div className="font-medium mb-1">
                                                                                        Warnings
                                                                                    </div>
                                                                                    <ul className="list-disc pl-5 space-y-0.5">
                                                                                        {(
                                                                                            check as any
                                                                                        ).details.warnings.map(
                                                                                            (
                                                                                                w: any,
                                                                                                wi: number
                                                                                            ) => (
                                                                                                <li
                                                                                                    key={
                                                                                                        wi
                                                                                                    }
                                                                                                >
                                                                                                    {String(
                                                                                                        w
                                                                                                    )}
                                                                                                </li>
                                                                                            )
                                                                                        )}
                                                                                    </ul>
                                                                                </div>
                                                                            )}
                                                                        {Array.isArray(
                                                                            (check as any).details
                                                                                ?.safe_patterns
                                                                        ) &&
                                                                            (check as any).details
                                                                                .safe_patterns
                                                                                .length > 0 && (
                                                                                <div className="bg-green-50 text-green-800 border border-green-200 rounded p-2 text-xs">
                                                                                    <div className="font-medium mb-1">
                                                                                        Safe
                                                                                        Patterns
                                                                                    </div>
                                                                                    <ul className="list-disc pl-5 space-y-0.5">
                                                                                        {(
                                                                                            check as any
                                                                                        ).details.safe_patterns.map(
                                                                                            (
                                                                                                sp: any,
                                                                                                spi: number
                                                                                            ) => (
                                                                                                <li
                                                                                                    key={
                                                                                                        spi
                                                                                                    }
                                                                                                >
                                                                                                    {String(
                                                                                                        sp.description ||
                                                                                                            sp.type ||
                                                                                                            sp
                                                                                                    )}
                                                                                                </li>
                                                                                            )
                                                                                        )}
                                                                                    </ul>
                                                                                </div>
                                                                            )}
                                                                        {Array.isArray(
                                                                            (check as any).details
                                                                                ?.array_access
                                                                        ) &&
                                                                            (check as any).details
                                                                                .array_access
                                                                                .length > 0 && (
                                                                                <div className="bg-gray-50 text-gray-800 border border-gray-200 rounded p-2 text-xs">
                                                                                    <div className="font-medium mb-1">
                                                                                        Array
                                                                                        Accesses
                                                                                    </div>
                                                                                    <ul className="list-disc pl-5 space-y-0.5">
                                                                                        {(
                                                                                            check as any
                                                                                        ).details.array_access.map(
                                                                                            (
                                                                                                aa: any,
                                                                                                ai: number
                                                                                            ) => (
                                                                                                <li
                                                                                                    key={
                                                                                                        ai
                                                                                                    }
                                                                                                >
                                                                                                    {String(
                                                                                                        aa.array
                                                                                                    )}

                                                                                                    [
                                                                                                    {String(
                                                                                                        aa.index
                                                                                                    )}

                                                                                                    ]
                                                                                                    at
                                                                                                    line{" "}
                                                                                                    {String(
                                                                                                        aa.line
                                                                                                    )}{" "}
                                                                                                    —{" "}
                                                                                                    {aa.is_safe
                                                                                                        ? "safe"
                                                                                                        : "potentially unsafe"}
                                                                                                </li>
                                                                                            )
                                                                                        )}
                                                                                    </ul>
                                                                                </div>
                                                                            )}
                                                                        {/* LLM Analysis Button for Memory Safety */}
                                                                        <div className="mt-3 pt-3 border-t border-gray-200">
                                                                            {llmAnalysisError && (
                                                                                <div className="mb-3 p-2 bg-red-50 border border-red-200 rounded text-red-800 text-xs">
                                                                                    <div className="font-medium mb-1">
                                                                                        LLM Analysis
                                                                                        Error:
                                                                                    </div>
                                                                                    <div className="mb-2">
                                                                                        {
                                                                                            llmAnalysisError
                                                                                        }
                                                                                    </div>
                                                                                    <button
                                                                                        onClick={
                                                                                            runLlmAnalysis
                                                                                        }
                                                                                        className="px-2 py-1 bg-red-500 text-white text-xs rounded hover:bg-red-600"
                                                                                    >
                                                                                        Retry LLM
                                                                                        Analysis
                                                                                    </button>
                                                                                </div>
                                                                            )}
                                                                            <button
                                                                                onClick={
                                                                                    runLlmAnalysis
                                                                                }
                                                                                className="w-full px-3 py-2 bg-purple-500 text-white text-sm rounded-lg hover:bg-purple-600 disabled:opacity-50 disabled:cursor-not-allowed"
                                                                                disabled={
                                                                                    isLlmAnalyzing
                                                                                }
                                                                            >
                                                                                {isLlmAnalyzing ? (
                                                                                    <div className="flex items-center justify-center gap-2">
                                                                                        <Loader2 className="w-3 h-3 animate-spin" />
                                                                                        Running
                                                                                        Advanced LLM
                                                                                        Analysis...
                                                                                    </div>
                                                                                ) : (
                                                                                    "🔍 Advanced LLM Memory Analysis"
                                                                                )}
                                                                            </button>

                                                                            {/* LLM Analysis Results - Display directly below button */}
                                                                            {llmAnalysisResults && (
                                                                                <div className="mt-4 space-y-3">
                                                                                    <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
                                                                                        <h4 className="font-medium text-purple-900 mb-2">
                                                                                            🧠
                                                                                            Memory
                                                                                            Safety
                                                                                            LLM
                                                                                            Analysis
                                                                                        </h4>
                                                                                        <p className="text-sm text-purple-800 mb-3">
                                                                                            Advanced
                                                                                            AI
                                                                                            analysis
                                                                                            focused
                                                                                            specifically
                                                                                            on
                                                                                            memory
                                                                                            safety
                                                                                            patterns,
                                                                                            bounds
                                                                                            checking,
                                                                                            and
                                                                                            allocation
                                                                                            strategies.
                                                                                        </p>

                                                                                        {/* Context-Dependent Bounds Analysis */}
                                                                                        {llmAnalysisResults
                                                                                            .analysis_results
                                                                                            ?.context_bounds && (
                                                                                            <div className="bg-orange-50 border border-orange-200 rounded-lg p-2 mb-2">
                                                                                                <h5 className="font-medium text-orange-900 mb-1 text-sm">
                                                                                                    🎯
                                                                                                    Context-Dependent
                                                                                                    Bounds
                                                                                                </h5>
                                                                                                <div className="text-xs text-orange-800">
                                                                                                    {llmAnalysisResults
                                                                                                        .analysis_results
                                                                                                        .context_bounds
                                                                                                        .raw_response ? (
                                                                                                        <pre className="whitespace-pre-wrap">
                                                                                                            {
                                                                                                                llmAnalysisResults
                                                                                                                    .analysis_results
                                                                                                                    .context_bounds
                                                                                                                    .raw_response
                                                                                                            }
                                                                                                        </pre>
                                                                                                    ) : (
                                                                                                        <div>
                                                                                                            {llmAnalysisResults
                                                                                                                .analysis_results
                                                                                                                .context_bounds
                                                                                                                .unsafe_accesses && (
                                                                                                                <div className="mb-1">
                                                                                                                    <strong>
                                                                                                                        Unsafe
                                                                                                                        Accesses:
                                                                                                                    </strong>
                                                                                                                    <ul className="list-disc pl-4 mt-1">
                                                                                                                        {llmAnalysisResults.analysis_results.context_bounds.unsafe_accesses.map(
                                                                                                                            (
                                                                                                                                access: any,
                                                                                                                                i: number
                                                                                                                            ) => (
                                                                                                                                <li
                                                                                                                                    key={
                                                                                                                                        i
                                                                                                                                    }
                                                                                                                                >
                                                                                                                                    {typeof access ===
                                                                                                                                    "string" ? (
                                                                                                                                        access
                                                                                                                                    ) : (
                                                                                                                                        <div>
                                                                                                                                            <div className="font-medium">
                                                                                                                                                Line{" "}
                                                                                                                                                {
                                                                                                                                                    access.line
                                                                                                                                                }

                                                                                                                                                :
                                                                                                                                            </div>
                                                                                                                                            <div className="text-sm">
                                                                                                                                                {
                                                                                                                                                    access.description
                                                                                                                                                }
                                                                                                                                            </div>
                                                                                                                                            {access.potential_causes && (
                                                                                                                                                <div className="text-xs text-gray-600 mt-1">
                                                                                                                                                    <strong>
                                                                                                                                                        Potential
                                                                                                                                                        causes:
                                                                                                                                                    </strong>{" "}
                                                                                                                                                    {
                                                                                                                                                        access.potential_causes
                                                                                                                                                    }
                                                                                                                                                </div>
                                                                                                                                            )}
                                                                                                                                        </div>
                                                                                                                                    )}
                                                                                                                                </li>
                                                                                                                            )
                                                                                                                        )}
                                                                                                                    </ul>
                                                                                                                </div>
                                                                                                            )}
                                                                                                            {llmAnalysisResults
                                                                                                                .analysis_results
                                                                                                                .context_bounds
                                                                                                                .suggestions && (
                                                                                                                <div className="mb-1">
                                                                                                                    <strong>
                                                                                                                        Suggestions:
                                                                                                                    </strong>
                                                                                                                    <ul className="list-disc pl-4 mt-1">
                                                                                                                        {llmAnalysisResults.analysis_results.context_bounds.suggestions.map(
                                                                                                                            (
                                                                                                                                suggestion: any,
                                                                                                                                i: number
                                                                                                                            ) => (
                                                                                                                                <li
                                                                                                                                    key={
                                                                                                                                        i
                                                                                                                                    }
                                                                                                                                >
                                                                                                                                    {typeof suggestion ===
                                                                                                                                    "string" ? (
                                                                                                                                        suggestion
                                                                                                                                    ) : (
                                                                                                                                        <div>
                                                                                                                                            <div className="font-medium">
                                                                                                                                                {suggestion.title ||
                                                                                                                                                    "Suggestion"}

                                                                                                                                                :
                                                                                                                                            </div>
                                                                                                                                            <div className="text-sm">
                                                                                                                                                {suggestion.description ||
                                                                                                                                                    suggestion}
                                                                                                                                            </div>
                                                                                                                                        </div>
                                                                                                                                    )}
                                                                                                                                </li>
                                                                                                                            )
                                                                                                                        )}
                                                                                                                    </ul>
                                                                                                                </div>
                                                                                                            )}
                                                                                                        </div>
                                                                                                    )}
                                                                                                </div>
                                                                                            </div>
                                                                                        )}

                                                                                        {/* Dynamic Memory Allocation Analysis */}
                                                                                        {llmAnalysisResults
                                                                                            .analysis_results
                                                                                            ?.dynamic_memory && (
                                                                                            <div className="bg-red-50 border border-red-200 rounded-lg p-2 mb-2">
                                                                                                <h5 className="font-medium text-red-900 mb-1 text-sm">
                                                                                                    💾
                                                                                                    Dynamic
                                                                                                    Memory
                                                                                                    Allocation
                                                                                                </h5>
                                                                                                <div className="text-xs text-red-800">
                                                                                                    {llmAnalysisResults
                                                                                                        .analysis_results
                                                                                                        .dynamic_memory
                                                                                                        .raw_response ? (
                                                                                                        <pre className="whitespace-pre-wrap">
                                                                                                            {
                                                                                                                llmAnalysisResults
                                                                                                                    .analysis_results
                                                                                                                    .dynamic_memory
                                                                                                                    .raw_response
                                                                                                            }
                                                                                                        </pre>
                                                                                                    ) : (
                                                                                                        <div>
                                                                                                            {llmAnalysisResults
                                                                                                                .analysis_results
                                                                                                                .dynamic_memory
                                                                                                                .allocation_patterns && (
                                                                                                                <div className="mb-1">
                                                                                                                    <strong>
                                                                                                                        Allocation
                                                                                                                        Patterns:
                                                                                                                    </strong>
                                                                                                                    <ul className="list-disc pl-4 mt-1">
                                                                                                                        {llmAnalysisResults.analysis_results.dynamic_memory.allocation_patterns.map(
                                                                                                                            (
                                                                                                                                pattern: any,
                                                                                                                                i: number
                                                                                                                            ) => (
                                                                                                                                <li
                                                                                                                                    key={
                                                                                                                                        i
                                                                                                                                    }
                                                                                                                                >
                                                                                                                                    {typeof pattern ===
                                                                                                                                    "string" ? (
                                                                                                                                        pattern
                                                                                                                                    ) : (
                                                                                                                                        <div>
                                                                                                                                            <div className="font-medium">
                                                                                                                                                {pattern.type ||
                                                                                                                                                    "Pattern"}

                                                                                                                                                :
                                                                                                                                            </div>
                                                                                                                                            <div className="text-sm">
                                                                                                                                                {pattern.description ||
                                                                                                                                                    pattern}
                                                                                                                                            </div>
                                                                                                                                        </div>
                                                                                                                                    )}
                                                                                                                                </li>
                                                                                                                            )
                                                                                                                        )}
                                                                                                                    </ul>
                                                                                                                </div>
                                                                                                            )}
                                                                                                            {llmAnalysisResults
                                                                                                                .analysis_results
                                                                                                                .dynamic_memory
                                                                                                                .potential_leaks && (
                                                                                                                <div className="mb-1">
                                                                                                                    <strong>
                                                                                                                        Potential
                                                                                                                        Leaks:
                                                                                                                    </strong>
                                                                                                                    <ul className="list-disc pl-4 mt-1">
                                                                                                                        {llmAnalysisResults.analysis_results.dynamic_memory.potential_leaks.map(
                                                                                                                            (
                                                                                                                                leak: any,
                                                                                                                                i: number
                                                                                                                            ) => (
                                                                                                                                <li
                                                                                                                                    key={
                                                                                                                                        i
                                                                                                                                    }
                                                                                                                                >
                                                                                                                                    {typeof leak ===
                                                                                                                                    "string" ? (
                                                                                                                                        leak
                                                                                                                                    ) : (
                                                                                                                                        <div>
                                                                                                                                            <div className="font-medium">
                                                                                                                                                {leak.type ||
                                                                                                                                                    "Leak"}

                                                                                                                                                :
                                                                                                                                            </div>
                                                                                                                                            <div className="text-sm">
                                                                                                                                                {leak.description ||
                                                                                                                                                    leak}
                                                                                                                                            </div>
                                                                                                                                        </div>
                                                                                                                                    )}
                                                                                                                                </li>
                                                                                                                            )
                                                                                                                        )}
                                                                                                                    </ul>
                                                                                                                </div>
                                                                                                            )}
                                                                                                        </div>
                                                                                                    )}
                                                                                                </div>
                                                                                            </div>
                                                                                        )}

                                                                                        {/* LLM Suggestions */}
                                                                                        {llmAnalysisResults.suggestions &&
                                                                                            llmAnalysisResults
                                                                                                .suggestions
                                                                                                .length >
                                                                                                0 && (
                                                                                                <div className="bg-blue-50 border border-blue-200 rounded-lg p-2">
                                                                                                    <h5 className="font-medium text-blue-900 mb-1 text-sm">
                                                                                                        💡
                                                                                                        LLM
                                                                                                        Suggestions
                                                                                                    </h5>
                                                                                                    <div className="space-y-1">
                                                                                                        {llmAnalysisResults.suggestions.map(
                                                                                                            (
                                                                                                                suggestion: any,
                                                                                                                i: number
                                                                                                            ) => (
                                                                                                                <div
                                                                                                                    key={
                                                                                                                        i
                                                                                                                    }
                                                                                                                    className="text-xs text-blue-800"
                                                                                                                >
                                                                                                                    <div className="font-medium">
                                                                                                                        {
                                                                                                                            suggestion.title
                                                                                                                        }
                                                                                                                    </div>
                                                                                                                    <div className="text-xs mt-1">
                                                                                                                        {
                                                                                                                            suggestion.message
                                                                                                                        }
                                                                                                                    </div>
                                                                                                                </div>
                                                                                                            )
                                                                                                        )}
                                                                                                    </div>
                                                                                                </div>
                                                                                            )}
                                                                                    </div>
                                                                                </div>
                                                                            )}
                                                                        </div>
                                                                    </div>
                                                                ) : check.name.includes(
                                                                      "type_safety"
                                                                  ) ? (
                                                                    <div className="space-y-2">
                                                                        {/* Type Safety Check Details */}
                                                                        {Array.isArray(
                                                                            (check as any).details
                                                                                ?.type_casts
                                                                        ) &&
                                                                            (check as any).details
                                                                                .type_casts.length >
                                                                                0 && (
                                                                                <div className="bg-yellow-50 text-yellow-800 border border-yellow-200 rounded p-2 text-xs">
                                                                                    <div className="font-medium mb-1">
                                                                                        Type Casts
                                                                                        Detected
                                                                                    </div>
                                                                                    <ul className="list-disc pl-5 space-y-0.5">
                                                                                        {(
                                                                                            check as any
                                                                                        ).details.type_casts.map(
                                                                                            (
                                                                                                cast: any,
                                                                                                i: number
                                                                                            ) => (
                                                                                                <li
                                                                                                    key={
                                                                                                        i
                                                                                                    }
                                                                                                >
                                                                                                    {
                                                                                                        cast.type
                                                                                                    }

                                                                                                    :{" "}
                                                                                                    {
                                                                                                        cast.expression
                                                                                                    }{" "}
                                                                                                    (line{" "}
                                                                                                    {
                                                                                                        cast.line
                                                                                                    }

                                                                                                    )
                                                                                                </li>
                                                                                            )
                                                                                        )}
                                                                                    </ul>
                                                                                </div>
                                                                            )}

                                                                        {Array.isArray(
                                                                            (check as any).details
                                                                                ?.implicit_conversions
                                                                        ) &&
                                                                            (check as any).details
                                                                                .implicit_conversions
                                                                                .length > 0 && (
                                                                                <div className="bg-orange-50 text-orange-800 border border-orange-200 rounded p-2 text-xs">
                                                                                    <div className="font-medium mb-1">
                                                                                        Implicit
                                                                                        Conversions
                                                                                    </div>
                                                                                    <ul className="list-disc pl-5 space-y-0.5">
                                                                                        {(
                                                                                            check as any
                                                                                        ).details.implicit_conversions.map(
                                                                                            (
                                                                                                conv: any,
                                                                                                i: number
                                                                                            ) => (
                                                                                                <li
                                                                                                    key={
                                                                                                        i
                                                                                                    }
                                                                                                >
                                                                                                    {
                                                                                                        conv.type
                                                                                                    }

                                                                                                    :{" "}
                                                                                                    {
                                                                                                        conv.variable
                                                                                                    }{" "}
                                                                                                    (line{" "}
                                                                                                    {
                                                                                                        conv.line
                                                                                                    }

                                                                                                    )
                                                                                                </li>
                                                                                            )
                                                                                        )}
                                                                                    </ul>
                                                                                </div>
                                                                            )}

                                                                        {/* LLM Analysis Button for Type Safety */}
                                                                        <div className="mt-3 pt-3 border-t border-gray-200">
                                                                            {typeSafetyLlmError && (
                                                                                <div className="mb-3 p-2 bg-red-50 border border-red-200 rounded text-red-800 text-xs">
                                                                                    <div className="font-medium mb-1">
                                                                                        Type Safety
                                                                                        LLM Analysis
                                                                                        Error:
                                                                                    </div>
                                                                                    <div className="mb-2">
                                                                                        {
                                                                                            typeSafetyLlmError
                                                                                        }
                                                                                    </div>
                                                                                    <button
                                                                                        onClick={
                                                                                            runTypeSafetyLlmAnalysis
                                                                                        }
                                                                                        className="px-2 py-1 bg-red-500 text-white text-xs rounded hover:bg-red-600"
                                                                                    >
                                                                                        Retry Type
                                                                                        Safety LLM
                                                                                        Analysis
                                                                                    </button>
                                                                                </div>
                                                                            )}
                                                                            <button
                                                                                onClick={
                                                                                    runTypeSafetyLlmAnalysis
                                                                                }
                                                                                className="w-full px-3 py-2 bg-purple-500 text-white text-sm rounded-lg hover:bg-purple-600 disabled:opacity-50 disabled:cursor-not-allowed"
                                                                                disabled={
                                                                                    isTypeSafetyLlmAnalyzing
                                                                                }
                                                                            >
                                                                                {isTypeSafetyLlmAnalyzing ? (
                                                                                    <div className="flex items-center justify-center gap-2">
                                                                                        <Loader2 className="w-3 h-3 animate-spin" />
                                                                                        Running
                                                                                        Advanced
                                                                                        Type Safety
                                                                                        Analysis...
                                                                                    </div>
                                                                                ) : (
                                                                                    "🔍 Advanced Type Safety LLM Analysis"
                                                                                )}
                                                                            </button>

                                                                            {/* Type Safety LLM Analysis Results - Clean Summary Format */}
                                                                            {typeSafetyLlmResults && (
                                                                                <div className="mt-4 space-y-3">
                                                                                    <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                                                                                        <h4 className="font-semibold text-purple-900 mb-3">
                                                                                            🧠
                                                                                            Advanced
                                                                                            Type
                                                                                            Safety
                                                                                            Analysis
                                                                                        </h4>

                                                                                        {/* Hardware-Specific Types */}
                                                                                        {typeSafetyLlmResults
                                                                                            .analysis_results
                                                                                            ?.hardware_types && (
                                                                                            <div className="mb-4">
                                                                                                <h5 className="font-medium text-purple-800 mb-2 flex items-center">
                                                                                                    🎯
                                                                                                    Hardware-Specific
                                                                                                    Types
                                                                                                    <span className="ml-2 text-xs bg-purple-200 text-purple-800 px-2 py-1 rounded">
                                                                                                        {
                                                                                                            hardware
                                                                                                        }
                                                                                                    </span>
                                                                                                </h5>

                                                                                                {typeSafetyLlmResults
                                                                                                    .analysis_results
                                                                                                    .hardware_types
                                                                                                    .summary && (
                                                                                                    <p className="text-sm text-purple-700 mb-3">
                                                                                                        {
                                                                                                            typeSafetyLlmResults
                                                                                                                .analysis_results
                                                                                                                .hardware_types
                                                                                                                .summary
                                                                                                        }
                                                                                                    </p>
                                                                                                )}

                                                                                                {typeSafetyLlmResults
                                                                                                    .analysis_results
                                                                                                    .hardware_types
                                                                                                    .critical_issues &&
                                                                                                    typeSafetyLlmResults
                                                                                                        .analysis_results
                                                                                                        .hardware_types
                                                                                                        .critical_issues
                                                                                                        .length >
                                                                                                        0 && (
                                                                                                        <div className="mb-3">
                                                                                                            <h6 className="font-medium text-red-700 mb-2">
                                                                                                                🚨
                                                                                                                Critical
                                                                                                                Issues
                                                                                                            </h6>
                                                                                                            <ul className="space-y-1">
                                                                                                                {typeSafetyLlmResults.analysis_results.hardware_types.critical_issues.map(
                                                                                                                    (
                                                                                                                        issue: string,
                                                                                                                        index: number
                                                                                                                    ) => (
                                                                                                                        <li
                                                                                                                            key={
                                                                                                                                index
                                                                                                                            }
                                                                                                                            className="text-sm text-red-600 flex items-start"
                                                                                                                        >
                                                                                                                            <span className="mr-2">
                                                                                                                                ❌
                                                                                                                            </span>
                                                                                                                            <span>
                                                                                                                                {
                                                                                                                                    issue
                                                                                                                                }
                                                                                                                            </span>
                                                                                                                        </li>
                                                                                                                    )
                                                                                                                )}
                                                                                                            </ul>
                                                                                                        </div>
                                                                                                    )}

                                                                                                {typeSafetyLlmResults
                                                                                                    .analysis_results
                                                                                                    .hardware_types
                                                                                                    .recommendations &&
                                                                                                    typeSafetyLlmResults
                                                                                                        .analysis_results
                                                                                                        .hardware_types
                                                                                                        .recommendations
                                                                                                        .length >
                                                                                                        0 && (
                                                                                                        <div className="mb-3">
                                                                                                            <h6 className="font-medium text-green-700 mb-2">
                                                                                                                ✅
                                                                                                                Action
                                                                                                                Items
                                                                                                            </h6>
                                                                                                            <div className="space-y-2">
                                                                                                                {typeSafetyLlmResults.analysis_results.hardware_types.recommendations.map(
                                                                                                                    (
                                                                                                                        rec: string,
                                                                                                                        index: number
                                                                                                                    ) => (
                                                                                                                        <label
                                                                                                                            key={
                                                                                                                                index
                                                                                                                            }
                                                                                                                            className="flex items-start cursor-pointer hover:bg-green-50 p-2 rounded"
                                                                                                                        >
                                                                                                                            <input
                                                                                                                                type="checkbox"
                                                                                                                                className="mt-1 mr-3 h-4 w-4 text-green-600 border-gray-300 rounded focus:ring-green-500"
                                                                                                                                onChange={() => {}} // Add state management if needed
                                                                                                                            />
                                                                                                                            <span className="text-sm text-green-700 flex-1">
                                                                                                                                {
                                                                                                                                    rec
                                                                                                                                }
                                                                                                                            </span>
                                                                                                                        </label>
                                                                                                                    )
                                                                                                                )}
                                                                                                            </div>
                                                                                                        </div>
                                                                                                    )}
                                                                                            </div>
                                                                                        )}

                                                                                        {/* Backend-Specific Types */}
                                                                                        {typeSafetyLlmResults
                                                                                            .analysis_results
                                                                                            ?.backend_types && (
                                                                                            <div className="mb-4">
                                                                                                <h5 className="font-medium text-purple-800 mb-2 flex items-center">
                                                                                                    🔧
                                                                                                    Backend-Specific
                                                                                                    Types
                                                                                                    <span className="ml-2 text-xs bg-purple-200 text-purple-800 px-2 py-1 rounded">
                                                                                                        {
                                                                                                            backend
                                                                                                        }
                                                                                                    </span>
                                                                                                </h5>

                                                                                                {typeSafetyLlmResults
                                                                                                    .analysis_results
                                                                                                    .backend_types
                                                                                                    .summary && (
                                                                                                    <p className="text-sm text-purple-700 mb-3">
                                                                                                        {
                                                                                                            typeSafetyLlmResults
                                                                                                                .analysis_results
                                                                                                                .backend_types
                                                                                                                .summary
                                                                                                        }
                                                                                                    </p>
                                                                                                )}

                                                                                                {typeSafetyLlmResults
                                                                                                    .analysis_results
                                                                                                    .backend_types
                                                                                                    .warnings &&
                                                                                                    typeSafetyLlmResults
                                                                                                        .analysis_results
                                                                                                        .backend_types
                                                                                                        .warnings
                                                                                                        .length >
                                                                                                        0 && (
                                                                                                        <div className="mb-3">
                                                                                                            <h6 className="font-medium text-yellow-700 mb-2">
                                                                                                                ⚠️
                                                                                                                Warnings
                                                                                                            </h6>
                                                                                                            <div className="space-y-2">
                                                                                                                {typeSafetyLlmResults.analysis_results.backend_types.warnings.map(
                                                                                                                    (
                                                                                                                        warning: string,
                                                                                                                        index: number
                                                                                                                    ) => (
                                                                                                                        <div
                                                                                                                            key={
                                                                                                                                index
                                                                                                                            }
                                                                                                                            className="flex items-start p-2 bg-yellow-50 rounded border-l-4 border-yellow-400"
                                                                                                                        >
                                                                                                                            <span className="mr-2 text-yellow-600">
                                                                                                                                ⚠️
                                                                                                                            </span>
                                                                                                                            <span className="text-sm text-yellow-800">
                                                                                                                                {
                                                                                                                                    warning
                                                                                                                                }
                                                                                                                            </span>
                                                                                                                        </div>
                                                                                                                    )
                                                                                                                )}
                                                                                                            </div>
                                                                                                        </div>
                                                                                                    )}

                                                                                                {typeSafetyLlmResults
                                                                                                    .analysis_results
                                                                                                    .backend_types
                                                                                                    .optimizations &&
                                                                                                    typeSafetyLlmResults
                                                                                                        .analysis_results
                                                                                                        .backend_types
                                                                                                        .optimizations
                                                                                                        .length >
                                                                                                        0 && (
                                                                                                        <div className="mb-3">
                                                                                                            <h6 className="font-medium text-blue-700 mb-2">
                                                                                                                ⚡
                                                                                                                Optimizations
                                                                                                            </h6>
                                                                                                            <div className="space-y-2">
                                                                                                                {typeSafetyLlmResults.analysis_results.backend_types.optimizations.map(
                                                                                                                    (
                                                                                                                        opt: string,
                                                                                                                        index: number
                                                                                                                    ) => (
                                                                                                                        <div
                                                                                                                            key={
                                                                                                                                index
                                                                                                                            }
                                                                                                                            className="flex items-start p-2 bg-blue-50 rounded border-l-4 border-blue-400"
                                                                                                                        >
                                                                                                                            <span className="mr-2 text-blue-600">
                                                                                                                                ⚡
                                                                                                                            </span>
                                                                                                                            <span className="text-sm text-blue-800">
                                                                                                                                {
                                                                                                                                    opt
                                                                                                                                }
                                                                                                                            </span>
                                                                                                                        </div>
                                                                                                                    )
                                                                                                                )}
                                                                                                            </div>
                                                                                                        </div>
                                                                                                    )}
                                                                                            </div>
                                                                                        )}

                                                                                        {/* Cross-Function Type Consistency */}
                                                                                        {typeSafetyLlmResults
                                                                                            .analysis_results
                                                                                            ?.cross_function_types && (
                                                                                            <div className="mb-4">
                                                                                                <h5 className="font-medium text-purple-800 mb-2">
                                                                                                    🔗
                                                                                                    Cross-Function
                                                                                                    Type
                                                                                                    Consistency
                                                                                                </h5>

                                                                                                {typeSafetyLlmResults
                                                                                                    .analysis_results
                                                                                                    .cross_function_types
                                                                                                    .summary && (
                                                                                                    <p className="text-sm text-purple-700 mb-3">
                                                                                                        {
                                                                                                            typeSafetyLlmResults
                                                                                                                .analysis_results
                                                                                                                .cross_function_types
                                                                                                                .summary
                                                                                                        }
                                                                                                    </p>
                                                                                                )}

                                                                                                {typeSafetyLlmResults
                                                                                                    .analysis_results
                                                                                                    .cross_function_types
                                                                                                    .recommendations &&
                                                                                                    typeSafetyLlmResults
                                                                                                        .analysis_results
                                                                                                        .cross_function_types
                                                                                                        .recommendations
                                                                                                        .length >
                                                                                                        0 && (
                                                                                                        <div className="mb-3">
                                                                                                            <h6 className="font-medium text-green-700 mb-2">
                                                                                                                ✅
                                                                                                                Consistency
                                                                                                                Checks
                                                                                                            </h6>
                                                                                                            <div className="space-y-2">
                                                                                                                {typeSafetyLlmResults.analysis_results.cross_function_types.recommendations.map(
                                                                                                                    (
                                                                                                                        rec: string,
                                                                                                                        index: number
                                                                                                                    ) => (
                                                                                                                        <label
                                                                                                                            key={
                                                                                                                                index
                                                                                                                            }
                                                                                                                            className="flex items-start cursor-pointer hover:bg-green-50 p-2 rounded"
                                                                                                                        >
                                                                                                                            <input
                                                                                                                                type="checkbox"
                                                                                                                                className="mt-1 mr-3 h-4 w-4 text-green-600 border-gray-300 rounded focus:ring-green-500"
                                                                                                                                onChange={() => {}} // Add state management if needed
                                                                                                                            />
                                                                                                                            <span className="text-sm text-green-700 flex-1">
                                                                                                                                {
                                                                                                                                    rec
                                                                                                                                }
                                                                                                                            </span>
                                                                                                                        </label>
                                                                                                                    )
                                                                                                                )}
                                                                                                            </div>
                                                                                                        </div>
                                                                                                    )}
                                                                                            </div>
                                                                                        )}
                                                                                    </div>
                                                                                </div>
                                                                            )}
                                                                        </div>
                                                                    </div>
                                                                ) : (
                                                                    // Generic dump for other checks
                                                                    <pre className="bg-gray-50 border border-gray-200 rounded p-2 overflow-auto text-xs text-gray-700">
                                                                        <code>
                                                                            {JSON.stringify(
                                                                                (check as any)
                                                                                    .details,
                                                                                null,
                                                                                2
                                                                            )}
                                                                        </code>
                                                                    </pre>
                                                                )}
                                                            </div>
                                                        ) : (
                                                            <div className="text-sm text-gray-500 italic">
                                                                No detailed information available
                                                                for this check.
                                                            </div>
                                                        );
                                                    })()}
                                                </div>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}
                </div>

                {/* Performance Section */}
                <div className="border-b border-gray-200">
                    <button
                        onClick={() => toggleSection("performance")}
                        className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50"
                    >
                        <span className="font-medium text-gray-900">Performance</span>
                        {expandedSections.has("performance") ? (
                            <ChevronDown className="w-4 h-4 text-gray-400" />
                        ) : (
                            <ChevronRight className="w-4 h-4 text-gray-400" />
                        )}
                    </button>
                    {expandedSections.has("performance") && (
                        <div className="px-4 pb-4 space-y-3">
                            <div className="grid grid-cols-2 gap-4">
                                <div className="p-3 bg-gray-50 rounded-lg">
                                    <div className="text-sm font-medium text-gray-700">
                                        Memory Usage
                                    </div>
                                    <div className="text-lg font-semibold text-gray-900">
                                        {analysisResult.performance.shared_mem_per_block_bytes
                                            ? (
                                                  analysisResult.performance
                                                      .shared_mem_per_block_bytes / 1024
                                              ).toFixed(1)
                                            : "N/A"}{" "}
                                        KB
                                    </div>
                                </div>
                                <div className="p-3 bg-gray-50 rounded-lg">
                                    <div className="text-sm font-medium text-gray-700">
                                        Optimizations
                                    </div>
                                    <div className="text-sm text-gray-600">
                                        {[
                                            analysisResult.performance.tiling_detected && "Tiling",
                                            analysisResult.performance.vectorization_detected &&
                                                "Vectorization",
                                            analysisResult.performance.tensor_core_usage_detected &&
                                                "Tensor Cores",
                                            analysisResult.performance.loop_unrolling_detected &&
                                                "Loop Unrolling",
                                        ]
                                            .filter(Boolean)
                                            .join(", ") || "None detected"}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* GPU Execution Section */}
                <div className="border-b border-gray-200">
                    <button
                        onClick={() => toggleSection("gpu-execution")}
                        className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50"
                    >
                        <span className="font-medium text-gray-900">Real GPU Execution</span>
                        {expandedSections.has("gpu-execution") ? (
                            <ChevronDown className="w-4 h-4 text-gray-400" />
                        ) : (
                            <ChevronRight className="w-4 h-4 text-gray-400" />
                        )}
                    </button>
                    {expandedSections.has("gpu-execution") && (
                        <div className="px-4 pb-4 space-y-4">
                            <div className="p-4 bg-blue-50 rounded-lg">
                                <h4 className="font-medium text-blue-900 mb-3">
                                    Execute on Real GPU
                                </h4>
                                <div className="flex items-center gap-2 mb-3">
                                    <select
                                        value={executionProvider}
                                        onChange={(e) => setExecutionProvider(e.target.value)}
                                        className="text-sm border rounded px-2 py-1"
                                    >
                                        <option value="github_colab">Google Colab (GitHub)</option>
                                    </select>
                                    <button
                                        onClick={executeOnGPU}
                                        disabled={isExecuting || !kernelCode.trim()}
                                        className="px-3 py-1 bg-blue-500 text-white rounded text-sm disabled:opacity-50 flex items-center gap-1"
                                    >
                                        {isExecuting ? (
                                            <>
                                                <Loader2 className="w-3 h-3 animate-spin" />
                                                Executing...
                                            </>
                                        ) : (
                                            "Run on GPU"
                                        )}
                                    </button>
                                </div>

                                {executionError && (
                                    <div className="mt-2 p-2 bg-red-100 border border-red-300 rounded text-sm text-red-700">
                                        <div className="flex items-center gap-1">
                                            <XCircle className="w-4 h-4" />
                                            {executionError}
                                        </div>
                                    </div>
                                )}

                                {realMetrics && (
                                    <div className="mt-3 space-y-2">
                                        <h5 className="font-medium text-blue-900 text-sm">
                                            Execution Results:
                                        </h5>
                                        <div className="grid grid-cols-2 gap-2 text-sm">
                                            <div className="flex justify-between">
                                                <span className="text-gray-600">Provider:</span>
                                                <span className="font-medium">
                                                    {realMetrics.provider || "N/A"}
                                                </span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-gray-600">Status:</span>
                                                <span className="font-medium text-green-600">
                                                    {realMetrics.status || "Completed"}
                                                </span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-gray-600">
                                                    Execution Time:
                                                </span>
                                                <span className="font-medium">
                                                    {realMetrics.execution_time?.toFixed(2) ||
                                                        "N/A"}
                                                    ms
                                                </span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-gray-600">
                                                    GPU Utilization:
                                                </span>
                                                <span className="font-medium">
                                                    {realMetrics.gpu_utilization?.toFixed(1) ||
                                                        "N/A"}
                                                    %
                                                </span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-gray-600">Memory Usage:</span>
                                                <span className="font-medium">
                                                    {realMetrics.memory_usage?.toFixed(0) || "N/A"}
                                                    MB
                                                </span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-gray-600">Throughput:</span>
                                                <span className="font-medium">
                                                    {realMetrics.throughput
                                                        ? realMetrics.throughput.toExponential(2)
                                                        : "N/A"}{" "}
                                                    ops/s
                                                </span>
                                            </div>
                                        </div>
                                        {realMetrics.hardware && (
                                            <div className="text-xs text-gray-500 mt-2">
                                                Hardware: {realMetrics.hardware}
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>

                {/* Suggestions Section */}
                <div className="border-b border-gray-200">
                    <button
                        onClick={() => toggleSection("suggestions")}
                        className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50"
                    >
                        <span className="font-medium text-gray-900">
                            Suggestions ({analysisResult.suggestions.length})
                        </span>
                        {expandedSections.has("suggestions") ? (
                            <ChevronDown className="w-4 h-4 text-gray-400" />
                        ) : (
                            <ChevronRight className="w-4 h-4 text-gray-400" />
                        )}
                    </button>
                    {expandedSections.has("suggestions") && (
                        <div className="px-4 pb-4 space-y-3">
                            {analysisResult.suggestions.length === 0 ? (
                                <div className="text-center py-4 text-gray-500">
                                    <CheckCircle className="w-8 h-8 mx-auto mb-2 text-green-500" />
                                    <p className="text-sm">No suggestions available</p>
                                </div>
                            ) : (
                                analysisResult.suggestions.map((suggestion, index) => (
                                    <div
                                        key={index}
                                        className={`p-4 border rounded-lg ${getSeverityColor(
                                            suggestion.severity
                                        )}`}
                                    >
                                        <div className="flex items-start gap-2">
                                            {getSeverityIcon(suggestion.severity)}
                                            <div className="flex-1">
                                                <div className="font-medium text-sm">
                                                    {suggestion.title}
                                                </div>
                                                <div className="text-sm mt-1">
                                                    {suggestion.message}
                                                </div>
                                                {suggestion.code_snippet && (
                                                    <div className="mt-3">
                                                        <div className="flex items-center justify-between mb-2">
                                                            <span className="text-xs font-medium text-gray-600">
                                                                Code Suggestion
                                                            </span>
                                                            <button
                                                                onClick={() =>
                                                                    copyToClipboard(
                                                                        suggestion.code_snippet!
                                                                    )
                                                                }
                                                                className="p-1 text-gray-400 hover:text-gray-600"
                                                                title="Copy code"
                                                            >
                                                                <Copy className="w-3 h-3" />
                                                            </button>
                                                        </div>
                                                        <pre className="text-xs bg-gray-900 text-green-400 p-2 rounded overflow-x-auto">
                                                            <code>{suggestion.code_snippet}</code>
                                                        </pre>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

"use client";

import { useState, useEffect } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { Problem, parseProblemMDX } from "../utils/parseMDX";
import CriticPanel from "./CriticPanel";

export default function KernelWorkbench() {
    const [backend, setBackend] = useState("CUDA");
    const [hardware, setHardware] = useState("NVIDIA T4");

    // Define which backends support compilation and execution
    const supportsCompilation = backend === "CUDA";
    const supportsExecution = backend === "CUDA";

    // Define supported hardware for each backend
    const getSupportedHardware = (backend: string) => {
        switch (backend) {
            case "CUDA":
                return ["NVIDIA T4", "CPU"];
            case "Triton":
            case "OpenCL":
                return ["NVIDIA T4", "CPU"]; // Same options but compilation blocked
            default:
                return ["NVIDIA T4", "CPU"];
        }
    };

    const supportedHardware = getSupportedHardware(backend);

    // Auto-switch hardware when backend changes if current hardware is not supported
    useEffect(() => {
        if (!supportedHardware.includes(hardware)) {
            setHardware(supportedHardware[0]);
        }
    }, [backend, supportedHardware, hardware]);

    const [attempts, setAttempts] = useState(1);
    const [prompt, setPrompt] = useState("");
    const [selectedProblem, setSelectedProblem] = useState<string>("");
    const [problems, setProblems] = useState<Problem[]>([]);
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);
    const [isLoadingProblems, setIsLoadingProblems] = useState(true);
    const [isGenerating, setIsGenerating] = useState(false);
    const [generatedCode, setGeneratedCode] = useState<string>("");
    const [isCompiling, setIsCompiling] = useState(false);
    const [compilationStatus, setCompilationStatus] = useState<
        "idle" | "compiling" | "success" | "error"
    >("idle");
    const [compilationError, setCompilationError] = useState<string>("");
    const [compilationAttempts, setCompilationAttempts] = useState(0);
    const [maxCompilationAttempts] = useState(5);

    // Load sample problems from markdown files
    useEffect(() => {
        const loadProblems = async () => {
            const problemFiles = [
                { id: "matrix_multiplication", file: "matrix_multiplication.mdx" },
                { id: "convolution", file: "convolution.mdx" },
                { id: "reduction", file: "reduction.mdx" },
            ];

            const loadedProblems: Problem[] = [];

            for (const { id, file } of problemFiles) {
                try {
                    console.log(`Fetching: /samples/problems-mdx/${file}`);
                    const response = await fetch(`/samples/problems-mdx/${file}`);
                    console.log(`Response status: ${response.status}`);

                    if (response.ok) {
                        const content = await response.text();
                        const problem = parseProblemMDX(id, content);
                        loadedProblems.push(problem);
                    } else {
                        console.error(`Failed to fetch ${file}: ${response.status}`);
                    }
                } catch (error) {
                    console.error(`Failed to load ${file}:`, error);
                }
            }

            setProblems(loadedProblems);
            setIsLoadingProblems(false);
        };

        loadProblems();
    }, []);

    const selectedProblemData = problems.find((p) => p.id === selectedProblem);
    const currentCode = selectedProblemData?.code || "";

    // Reset compilation state when problem changes (new problem = new compilation needed)
    useEffect(() => {
        if (selectedProblem) {
            setCompilationStatus("idle");
            setCompilationError("");
            setCompilationAttempts(0);
            setIsCompiling(false);
            // Clear generated code when problem changes
            setGeneratedCode("");
        }
    }, [selectedProblem]);

    const generateKernel = async () => {
        if (!selectedProblem || !currentCode.trim() || !prompt.trim()) {
            alert("Please select a problem, ensure code is present, and provide a prompt.");
            return;
        }

        console.log(
            "Rendering - generatedCode:",
            generatedCode?.substring(0, 50),
            "length:",
            generatedCode?.length
        );

        setIsGenerating(true);
        setGeneratedCode("");

        // Reset compilation state when generating new code
        setCompilationStatus("idle");
        setCompilationError("");
        setCompilationAttempts(0);
        setIsCompiling(false);

        try {
            const requestBody = {
                backend,
                hardware,
                code: currentCode,
                user_prompt: prompt,
                problem_name: selectedProblemData?.name,
                provider: "gemini",
            };

            console.log("Sending to backend:", requestBody);

            const response = await fetch(
                `${process.env.BACKEND_URL || "http://localhost:8000"}/api/v1/kernel/generate`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "Cache-Control": "no-cache",
                    },
                    body: JSON.stringify(requestBody),
                }
            );

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(
                    `Backend error: ${response.status} - ${errorData.detail || "Unknown error"}`
                );
            }

            const data = await response.json();
            console.log("Backend response:", data);
            console.log("Optimized code length:", data.optimized_code?.length);
            console.log("Optimized code preview:", data.optimized_code?.substring(0, 100));

            // Set the generated code from the response
            if (data.optimized_code) {
                console.log("Code received, updating display...");
                setGeneratedCode(data.optimized_code);
            } else {
                console.error("No optimized_code in response:", data);
                throw new Error("No optimized code in response");
            }
        } catch (error) {
            console.error("Error generating kernel:", error);
            const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
            alert(`Failed to generate kernel: ${errorMessage}`);
        } finally {
            setIsGenerating(false);
        }
    };

    const handleCompile = async () => {
        if (!generatedCode.trim()) {
            alert("No code to compile");
            return;
        }

        // Reset attempts when starting fresh compilation
        setCompilationAttempts(0);
        setIsCompiling(true);
        setCompilationStatus("compiling");
        setCompilationError("");

        await attemptCompilation();
    };

    const attemptCompilation = async () => {
        try {
            const response = await fetch(
                `${process.env.BACKEND_URL || "http://localhost:8000"}/api/v1/gpu/compile-kernel`,
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        kernel_code: generatedCode,
                        hardware,
                        backend,
                        problem_name: selectedProblemData?.name || "Unknown",
                        user_prompt: prompt,
                    }),
                }
            );

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log("🔍 Compilation result:", result);

            if (result.success) {
                setCompilationStatus("success");
                setCompilationError("");
                setCompilationAttempts(0);
                console.log("✅ Compilation successful!");

                // Update the generated code with the corrected version
                if (result.corrected_code) {
                    setGeneratedCode(result.corrected_code);
                    console.log("✅ Code updated with corrected version");
                }

                setIsCompiling(false);
            } else {
                setCompilationStatus("error");
                setCompilationError(result.error || "Compilation failed");
                console.log("❌ Compilation failed:", result.error);

                // Try to fix the kernel with LLM if under limit
                if (compilationAttempts < maxCompilationAttempts) {
                    await handleFixKernel(result.error);
                } else {
                    console.log("❌ Max attempts reached, stopping");
                    setIsCompiling(false);
                }
            }
        } catch (error) {
            console.error("Compilation failed:", error);
            setCompilationStatus("error");
            setCompilationError("Failed to compile kernel");
            setIsCompiling(false);
        }
    };

    const handleFixKernel = async (error: string) => {
        const currentAttempt = compilationAttempts + 1;
        console.log(
            `🔧 Attempting to fix kernel (attempt ${currentAttempt}/${maxCompilationAttempts})`
        );

        try {
            const response = await fetch(
                `${process.env.BACKEND_URL || "http://localhost:8000"}/api/v1/gpu/fix-kernel`,
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        kernel_code: generatedCode,
                        compilation_error: error,
                        hardware,
                        backend,
                        problem_name: selectedProblemData?.name || "Unknown",
                        user_prompt: prompt,
                    }),
                }
            );

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log("🔍 Fix result:", result);

            if (result.success) {
                setGeneratedCode(result.fixed_code);
                setCompilationAttempts(currentAttempt);
                console.log("🔧 Kernel fixed, retrying compilation...");

                // Retry compilation with fixed code
                setTimeout(() => {
                    attemptCompilation();
                }, 1000);
            } else {
                setCompilationError(result.error || "Failed to fix kernel");
                console.log("❌ Failed to fix kernel:", result.error);
                setIsCompiling(false);
            }
        } catch (error) {
            console.error("Kernel fixing failed:", error);
            setCompilationError("Failed to fix kernel");
            setIsCompiling(false);
        }
    };

    return (
        <div className="h-full grid grid-cols-1 lg:grid-cols-2 gap-1.5 p-1.5">
            {/* Left column */}
            <div className="flex flex-col bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden h-full min-h-0">
                {/* Sticky header with dropdowns */}
                <div className="sticky top-0 z-10 bg-white border-b border-gray-200">
                    <div className="flex items-center justify-between px-4 py-3">
                        <h3 className="text-sm font-semibold text-gray-900">Kernel Builder</h3>
                        <div className="flex items-center gap-2">
                            <select
                                className="border border-gray-300 rounded-md px-2 py-1 text-sm"
                                value={backend}
                                onChange={(e) => {
                                    setBackend(e.target.value);
                                    // Reset compilation state when backend changes
                                    setCompilationStatus("idle");
                                    setCompilationError("");
                                    setCompilationAttempts(0);
                                    setIsCompiling(false);
                                    // Clear generated code when backend changes
                                    setGeneratedCode("");
                                }}
                            >
                                <option>CUDA</option>
                                <option>Triton</option>
                                <option>OpenCL</option>
                            </select>
                            <select
                                className="border border-gray-300 rounded-md px-2 py-1 text-sm"
                                value={hardware}
                                onChange={(e) => {
                                    setHardware(e.target.value);
                                    // Reset compilation state when hardware changes
                                    setCompilationStatus("idle");
                                    setCompilationError("");
                                    setCompilationAttempts(0);
                                    setIsCompiling(false);
                                    // Clear generated code when hardware changes
                                    setGeneratedCode("");
                                }}
                            >
                                {supportedHardware.map((hw) => (
                                    <option key={hw} value={hw}>
                                        {hw}
                                    </option>
                                ))}
                                <option disabled>NVIDIA H100</option>
                                <option disabled>AMD</option>
                            </select>
                        </div>
                    </div>
                </div>

                {/* Top prompt composer */}
                <div className="border-b border-gray-200 p-4 space-y-4 flex-shrink-0">
                    {/* Problem Selector Dropdown */}
                    <div className="relative">
                        <button
                            type="button"
                            onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                            className="w-full flex items-center justify-between px-3 py-2 border border-gray-300 rounded-lg bg-white text-left focus:outline-none focus:ring-2 focus:ring-primary-500"
                        >
                            <span className="text-sm text-gray-700">
                                {isLoadingProblems
                                    ? "Loading problems..."
                                    : selectedProblemData
                                    ? selectedProblemData.name
                                    : "Select a problem..."}
                            </span>
                            {isDropdownOpen ? (
                                <ChevronUp className="h-4 w-4 text-gray-400" />
                            ) : (
                                <ChevronDown className="h-4 w-4 text-gray-400" />
                            )}
                        </button>

                        {isDropdownOpen && !isLoadingProblems && (
                            <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-300 rounded-lg shadow-lg z-20 max-h-48 overflow-y-auto">
                                {problems.map((problem) => (
                                    <button
                                        key={problem.id}
                                        type="button"
                                        onClick={() => {
                                            setSelectedProblem(problem.id);
                                            setIsDropdownOpen(false);
                                        }}
                                        className="w-full text-left px-3 py-2 text-sm hover:bg-gray-50 first:rounded-t-lg last:rounded-b-lg"
                                    >
                                        {problem.name}
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Code Block Display */}
                    {selectedProblem && currentCode && (
                        <div className="bg-gray-900 rounded-lg p-3 h-24 overflow-y-auto">
                            <pre className="text-xs text-green-400 font-mono leading-tight whitespace-pre">
                                <code>{currentCode}</code>
                            </pre>
                        </div>
                    )}
                    {selectedProblem && !currentCode && (
                        <div className="bg-gray-100 rounded-lg p-3 h-24 flex items-center justify-center text-sm text-gray-500">
                            No code available for this problem
                        </div>
                    )}
                    {!selectedProblem && (
                        <div className="bg-gray-100 rounded-lg p-3 h-24 flex items-center justify-center text-sm text-gray-500">
                            Select a problem to view code
                        </div>
                    )}

                    {/* Prompt Input */}
                    <div className="flex items-center gap-2">
                        <input
                            type="text"
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            placeholder="Describe the kernel you want to generate…"
                            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                        />
                        <button
                            className="px-4 py-2 flex items-center gap-2 rounded-lg bg-primary-500 hover:bg-primary-600 text-white disabled:opacity-50 disabled:cursor-not-allowed"
                            title="Generate Kernel"
                            onClick={generateKernel}
                            disabled={
                                isGenerating ||
                                !selectedProblem ||
                                !currentCode.trim() ||
                                !prompt.trim()
                            }
                        >
                            {isGenerating ? (
                                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                            ) : (
                                <span className="text-sm">Generate</span>
                            )}
                        </button>
                    </div>
                </div>

                {/* Generated Code Display */}
                <div className="flex-1 p-4 min-h-0">
                    {generatedCode ? (
                        <div className="h-full flex flex-col">
                            <div className="flex items-center justify-between mb-3 flex-shrink-0">
                                <h3 className="text-sm font-semibold text-gray-900">
                                    Generated Kernel
                                </h3>
                                <div className="flex items-center gap-2">
                                    {isCompiling && (
                                        <div className="flex items-center text-xs text-blue-600">
                                            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse mr-2"></div>
                                            Compiling... ({compilationAttempts}/
                                            {maxCompilationAttempts})
                                        </div>
                                    )}
                                    {compilationStatus === "success" && (
                                        <div className="flex items-center text-xs text-green-600">
                                            <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                                            ✅ Compiled Successfully
                                        </div>
                                    )}
                                    {compilationStatus === "error" &&
                                        compilationAttempts >= maxCompilationAttempts && (
                                            <div className="flex items-center text-xs text-red-600">
                                                <div className="w-2 h-2 bg-red-500 rounded-full mr-2"></div>
                                                ❌ Compilation Failed
                                            </div>
                                        )}
                                    <button
                                        onClick={() => {
                                            navigator.clipboard.writeText(generatedCode);
                                            alert("Code copied to clipboard!");
                                        }}
                                        className="px-2 py-1 text-xs bg-gray-200 hover:bg-gray-300 rounded text-gray-700"
                                    >
                                        Copy
                                    </button>
                                    <button
                                        onClick={handleCompile}
                                        disabled={
                                            !generatedCode.trim() ||
                                            isCompiling ||
                                            isGenerating ||
                                            !supportsCompilation
                                        }
                                        className={`px-2 py-1 text-xs rounded text-white ${
                                            !supportsCompilation
                                                ? "bg-gray-400 cursor-not-allowed"
                                                : compilationStatus === "success"
                                                ? "bg-green-500 hover:bg-green-600"
                                                : compilationStatus === "error" &&
                                                  compilationAttempts >= maxCompilationAttempts
                                                ? "bg-red-500 hover:bg-red-600"
                                                : "bg-blue-500 hover:bg-blue-600"
                                        } disabled:bg-gray-400 disabled:cursor-not-allowed`}
                                        title={
                                            !supportsCompilation
                                                ? "Currently unavailable for this backend"
                                                : ""
                                        }
                                    >
                                        {!supportsCompilation
                                            ? "🚫 Blocked"
                                            : compilationStatus === "success"
                                            ? "✅ Compiled"
                                            : compilationStatus === "error" &&
                                              compilationAttempts >= maxCompilationAttempts
                                            ? "❌ Failed"
                                            : isCompiling
                                            ? "Compiling..."
                                            : "Compile"}
                                    </button>
                                </div>
                            </div>
                            <div className="flex-1 bg-gray-900 rounded-lg overflow-hidden border border-gray-700 min-h-0">
                                <div className="h-full flex">
                                    {/* Fixed line numbers */}
                                    <div className="bg-gray-800 text-gray-500 text-sm font-mono leading-relaxed p-4 pr-2 flex-shrink-0">
                                        {generatedCode.split("\n").map((_, index) => (
                                            <div key={index} className="text-right w-8">
                                                {index + 1}
                                            </div>
                                        ))}
                                    </div>
                                    {/* Scrollable code content */}
                                    <div className="flex-1 overflow-auto">
                                        <pre className="text-sm text-green-400 font-mono leading-relaxed p-4 pl-2 whitespace-pre min-w-full">
                                            <code>
                                                {generatedCode.split("\n").map((line, index) => (
                                                    <div key={index} className="hover:bg-gray-800">
                                                        {line}
                                                    </div>
                                                ))}
                                            </code>
                                        </pre>
                                    </div>
                                </div>
                            </div>
                            {!supportsCompilation && (
                                <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded-lg">
                                    <div className="flex items-center">
                                        <div className="w-4 h-4 text-yellow-500 mr-2">⚠️</div>
                                        <span className="text-sm text-yellow-700">
                                            {backend} compilation unavailable - Generate button
                                            still works
                                        </span>
                                    </div>
                                </div>
                            )}
                            {compilationError && (
                                <div className="mt-2 p-3 bg-red-50 border border-red-200 rounded-lg">
                                    <div className="flex items-start">
                                        <div className="flex-shrink-0">
                                            <div className="w-4 h-4 text-red-500">❌</div>
                                        </div>
                                        <div className="ml-3 flex-1">
                                            <div className="flex items-center gap-2 mb-1">
                                                <h4 className="text-sm font-medium text-red-800">
                                                    Compilation Error
                                                </h4>
                                                {compilationAttempts < maxCompilationAttempts && (
                                                    <span className="text-xs text-red-600">
                                                        🔧 Attempting to fix automatically... (
                                                        {compilationAttempts}/
                                                        {maxCompilationAttempts})
                                                    </span>
                                                )}
                                            </div>
                                            <div className="text-sm text-red-700">
                                                <div className="bg-red-100 border border-red-300 rounded p-2 max-h-20 overflow-y-auto">
                                                    <pre className="whitespace-pre-wrap font-mono text-xs leading-tight">
                                                        {compilationError}
                                                    </pre>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="h-full flex items-center justify-center">
                            <div className="text-center">
                                <div className="mx-auto mb-4 h-10 w-10 rounded-full bg-primary-50 text-primary-600 flex items-center justify-center">
                                    ⚙️
                                </div>
                                <h2 className="text-gray-700">Generate a kernel in 30 seconds</h2>
                                <p className="text-sm text-gray-500 mt-2">
                                    Select a problem, configure settings, and click generate
                                </p>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Right column - Critic Agent */}
            <div className="h-full min-h-0">
                <CriticPanel
                    kernelCode={generatedCode}
                    hardware={hardware}
                    backend={backend}
                    isAnalyzing={false}
                    compilationStatus={compilationStatus}
                    onAnalysisComplete={(result) => {
                        console.log("Critic analysis completed:", result);
                    }}
                />
            </div>
        </div>
    );
}

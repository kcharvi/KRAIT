"use client";

import { useState, useEffect } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { Problem, parseProblemMDX } from "../utils/parseMDX";
import CriticPanel from "./CriticPanel";

export default function KernelWorkbench() {
    const [backend, setBackend] = useState("CUDA");
    const [hardware, setHardware] = useState("NVIDIA T4");

    // Tab system state
    const [activeTab, setActiveTab] = useState<"cuda" | "pytorch">("cuda");

    // Define which backends support compilation and execution
    // Block PyTorch CUDA extensions until compilation issues are resolved
    const supportsCompilation = backend === "CUDA" && activeTab === "cuda";
    const supportsExecution = backend === "CUDA" && activeTab === "cuda";

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

    // Separate code storage for each tab
    const [generatedCudaCode, setGeneratedCudaCode] = useState<string>("");
    const [generatedPytorchCode, setGeneratedPytorchCode] = useState<string>("");

    // Backward compatibility - keep generatedCode for CriticPanel
    const [generatedCode, setGeneratedCode] = useState<string>("");

    // Separate compilation states for each tab
    const [cudaCompilationStatus, setCudaCompilationStatus] = useState<
        "idle" | "compiling" | "success" | "error"
    >("idle");
    const [pytorchCompilationStatus, setPytorchCompilationStatus] = useState<
        "idle" | "compiling" | "success" | "error"
    >("idle");

    const [cudaCompilationError, setCudaCompilationError] = useState<string>("");
    const [pytorchCompilationError, setPytorchCompilationError] = useState<string>("");

    const [cudaCompilationAttempts, setCudaCompilationAttempts] = useState(0);
    const [pytorchCompilationAttempts, setPytorchCompilationAttempts] = useState(0);

    // Legacy states for backward compatibility
    const [isCompiling, setIsCompiling] = useState(false);
    const [compilationStatus, setCompilationStatus] = useState<
        "idle" | "compiling" | "success" | "error"
    >("idle");
    const [compilationError, setCompilationError] = useState<string>("");
    const [compilationAttempts, setCompilationAttempts] = useState(0);

    const [maxCompilationAttempts] = useState(5);

    // Custom problem state
    const [customProblemCode, setCustomProblemCode] = useState<string>(`# Custom PyTorch Model Code
# Replace this with your own PyTorch model code
# Example: Matrix multiplication, convolution, reduction, etc.

        import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Initialize your model layers here
        self.layer1 = nn.Linear(10, 64)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x):
        # Put your model implementation here
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# This is how functions below will be used during
# validation and benchmarking
# model = Model(*get_init_inputs())
# output = model(*get_inputs())

def get_inputs():
    return [torch.randn(64, 10)]

def get_init_inputs():
    return []`);

    // Create custom problem object
    const customProblem: Problem = {
        id: "custom",
        name: "Custom Problem",
        code: customProblemCode,
        description: "Define your own custom PyTorch model problem",
    };

    // Load sample problems from markdown files
    useEffect(() => {
        const loadProblems = async () => {
            const problemFiles = [
                { id: "matrix_multiplication", file: "matrix_multiplication.mdx" },
                { id: "convolution", file: "convolution.mdx" },
                { id: "reduction", file: "reduction.mdx" },
            ];

            const loadedProblems: Problem[] = [customProblem]; // Start with custom problem

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

    // Update custom problem in problems array when customProblemCode changes
    useEffect(() => {
        setProblems((prevProblems) => {
            const updatedProblems = prevProblems.map((problem) =>
                problem.id === "custom" ? { ...problem, code: customProblemCode } : problem
            );
            return updatedProblems;
        });
    }, [customProblemCode]);

    const selectedProblemData = problems.find((p) => p.id === selectedProblem);
    const currentCode =
        selectedProblem === "custom" ? customProblemCode : selectedProblemData?.code || "";

    // Helper functions for tab management
    const getCurrentCode = () => {
        return activeTab === "cuda" ? generatedCudaCode : generatedPytorchCode;
    };

    const getCurrentCompilationStatus = () => {
        return activeTab === "cuda" ? cudaCompilationStatus : pytorchCompilationStatus;
    };

    const getCurrentCompilationError = () => {
        return activeTab === "cuda" ? cudaCompilationError : pytorchCompilationError;
    };

    const getCurrentCompilationAttempts = () => {
        return activeTab === "cuda" ? cudaCompilationAttempts : pytorchCompilationAttempts;
    };

    const setCurrentCode = (code: string) => {
        if (activeTab === "cuda") {
            setGeneratedCudaCode(code);
        } else {
            setGeneratedPytorchCode(code);
        }
        // Also update the legacy generatedCode for CriticPanel
        setGeneratedCode(code);
    };

    const resetCurrentCompilationState = () => {
        if (activeTab === "cuda") {
            setCudaCompilationStatus("idle");
            setCudaCompilationError("");
            setCudaCompilationAttempts(0);
        } else {
            setPytorchCompilationStatus("idle");
            setPytorchCompilationError("");
            setPytorchCompilationAttempts(0);
        }
    };

    // Reset compilation state when problem changes (new problem = new compilation needed)
    useEffect(() => {
        if (selectedProblem) {
            // Reset both tabs
            setCudaCompilationStatus("idle");
            setCudaCompilationError("");
            setCudaCompilationAttempts(0);
            setPytorchCompilationStatus("idle");
            setPytorchCompilationError("");
            setPytorchCompilationAttempts(0);

            // Reset legacy states
            setCompilationStatus("idle");
            setCompilationError("");
            setCompilationAttempts(0);
            setIsCompiling(false);

            // Clear generated code when problem changes
            setGeneratedCudaCode("");
            setGeneratedPytorchCode("");
            setGeneratedCode("");
        }
    }, [selectedProblem]);

    const generateKernel = async () => {
        if (!selectedProblem || !currentCode.trim() || !prompt.trim()) {
            alert("Please select a problem, ensure code is present, and provide a prompt.");
            return;
        }

        console.log("Starting kernel generation for both CUDA and PyTorch...");

        setIsGenerating(true);

        // Clear all generated code
        setGeneratedCudaCode("");
        setGeneratedPytorchCode("");
        setGeneratedCode("");

        // Reset compilation state when generating new code
        resetCurrentCompilationState();
        setCompilationStatus("idle");
        setCompilationError("");
        setCompilationAttempts(0);
        setIsCompiling(false);

        try {
            // Generate CUDA code
            console.log("Generating CUDA code...");
            const cudaResponse = await fetch(
                `${process.env.BACKEND_URL || "http://localhost:8000"}/api/v1/kernel/generate`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "Cache-Control": "no-cache",
                    },
                    body: JSON.stringify({
                        backend: "CUDA",
                        hardware,
                        code: currentCode,
                        user_prompt: prompt,
                        problem_name: selectedProblemData?.name,
                        provider: "gemini",
                    }),
                }
            );

            if (!cudaResponse.ok) {
                const errorData = await cudaResponse.json();
                throw new Error(
                    `CUDA generation failed: ${cudaResponse.status} - ${
                        errorData.detail || "Unknown error"
                    }`
                );
            }

            const cudaData = await cudaResponse.json();
            console.log("CUDA response:", cudaData);

            // Generate PyTorch code
            console.log("Generating PyTorch code...");
            const pytorchResponse = await fetch(
                `${process.env.BACKEND_URL || "http://localhost:8000"}/api/v1/kernel/generate`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "Cache-Control": "no-cache",
                    },
                    body: JSON.stringify({
                        backend: "PYTORCH_CUDA_EXTENSION",
                        hardware,
                        code: currentCode,
                        user_prompt: prompt,
                        problem_name: selectedProblemData?.name,
                        provider: "gemini",
                    }),
                }
            );

            if (!pytorchResponse.ok) {
                const errorData = await pytorchResponse.json();
                throw new Error(
                    `PyTorch generation failed: ${pytorchResponse.status} - ${
                        errorData.detail || "Unknown error"
                    }`
                );
            }

            const pytorchData = await pytorchResponse.json();
            console.log("PyTorch response:", pytorchData);

            // Set the generated codes
            if (cudaData.optimized_code) {
                console.log("CUDA code received, updating display...");
                setGeneratedCudaCode(cudaData.optimized_code);
                // Set as default for CriticPanel
                setGeneratedCode(cudaData.optimized_code);
            } else {
                console.error("No CUDA code in response:", cudaData);
                throw new Error("No CUDA code in response");
            }

            if (pytorchData.optimized_code) {
                console.log("PyTorch code received, updating display...");
                setGeneratedPytorchCode(pytorchData.optimized_code);
            } else {
                console.error("No PyTorch code in response:", pytorchData);
                throw new Error("No PyTorch code in response");
            }
        } catch (error) {
            console.error("Error generating kernels:", error);
            const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
            alert(`Failed to generate kernels: ${errorMessage}`);
        } finally {
            setIsGenerating(false);
        }
    };

    const handleCompile = async () => {
        const currentCode = getCurrentCode();
        if (!currentCode.trim()) {
            alert("No code to compile");
            return;
        }

        // Reset attempts when starting fresh compilation
        if (activeTab === "cuda") {
            setCudaCompilationAttempts(0);
            setCudaCompilationStatus("compiling");
            setCudaCompilationError("");
        } else {
            setPytorchCompilationAttempts(0);
            setPytorchCompilationStatus("compiling");
            setPytorchCompilationError("");
        }

        // Also update legacy states for backward compatibility
        setCompilationAttempts(0);
        setIsCompiling(true);
        setCompilationStatus("compiling");
        setCompilationError("");

        await attemptCompilation();
    };

    const handleStopCompilation = () => {
        setIsCompiling(false);
        setCompilationStatus("idle");
        setCompilationError("Compilation cancelled by user");
        setCompilationAttempts(0);
    };

    const attemptCompilation = async () => {
        const currentCode = getCurrentCode();
        const currentBackend = activeTab === "cuda" ? "CUDA" : "PYTORCH_CUDA_EXTENSION";

        try {
            console.log(`🔨 Starting ${activeTab} compilation...`);
            const response = await fetch(
                `${process.env.BACKEND_URL || "http://localhost:8000"}/api/v1/gpu/compile-kernel`,
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        kernel_code: currentCode,
                        hardware,
                        backend: currentBackend,
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
                // Update current tab's compilation status
                if (activeTab === "cuda") {
                    setCudaCompilationStatus("success");
                    setCudaCompilationError("");
                    setCudaCompilationAttempts(0);
                } else {
                    setPytorchCompilationStatus("success");
                    setPytorchCompilationError("");
                    setPytorchCompilationAttempts(0);
                }

                // Update legacy states
                setCompilationStatus("success");
                setCompilationError("");
                setCompilationAttempts(0);
                console.log("✅ Compilation successful!");

                // Update the generated code with the corrected version
                if (result.corrected_code) {
                    setCurrentCode(result.corrected_code);
                    console.log("✅ Code updated with corrected version");
                }

                setIsCompiling(false);
            } else {
                // Update current tab's compilation status
                if (activeTab === "cuda") {
                    setCudaCompilationStatus("error");
                    setCudaCompilationError(result.error || "Compilation failed");
                } else {
                    setPytorchCompilationStatus("error");
                    setPytorchCompilationError(result.error || "Compilation failed");
                }

                // Update legacy states
                setCompilationStatus("error");
                setCompilationError(result.error || "Compilation failed");
                console.log("❌ Compilation failed:", result.error);

                // Try to fix the kernel with LLM if under limit
                const currentAttempts = getCurrentCompilationAttempts();
                if (currentAttempts < maxCompilationAttempts) {
                    await handleFixKernel(result.error);
                } else {
                    console.log("❌ Max attempts reached, stopping");
                    setIsCompiling(false);
                }
            }
        } catch (error) {
            console.error("Compilation failed:", error);

            // Update current tab's compilation status
            if (activeTab === "cuda") {
                setCudaCompilationStatus("error");
                setCudaCompilationError("Failed to compile kernel");
            } else {
                setPytorchCompilationStatus("error");
                setPytorchCompilationError("Failed to compile kernel");
            }

            // Update legacy states
            setCompilationStatus("error");
            setCompilationError("Failed to compile kernel");
            setIsCompiling(false);
        }
    };

    const handleFixKernel = async (error: string) => {
        const currentCode = getCurrentCode();
        const currentBackend = activeTab === "cuda" ? "CUDA" : "PYTORCH_CUDA_EXTENSION";
        const currentAttempt = getCurrentCompilationAttempts() + 1;

        console.log(
            `🔧 Attempting to fix ${activeTab} kernel (attempt ${currentAttempt}/${maxCompilationAttempts})`
        );

        try {
            const response = await fetch(
                `${process.env.BACKEND_URL || "http://localhost:8000"}/api/v1/gpu/fix-kernel`,
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        kernel_code: currentCode,
                        compilation_error: error,
                        hardware,
                        backend: currentBackend,
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
                setCurrentCode(result.fixed_code);

                // Update current tab's compilation attempts
                if (activeTab === "cuda") {
                    setCudaCompilationAttempts(currentAttempt);
                } else {
                    setPytorchCompilationAttempts(currentAttempt);
                }

                // Update legacy state
                setCompilationAttempts(currentAttempt);
                console.log("🔧 Kernel fixed, retrying compilation...");

                // Retry compilation with fixed code
                setTimeout(() => {
                    attemptCompilation();
                }, 1000);
            } else {
                // Update current tab's compilation error
                if (activeTab === "cuda") {
                    setCudaCompilationError(result.error || "Failed to fix kernel");
                } else {
                    setPytorchCompilationError(result.error || "Failed to fix kernel");
                }

                // Update legacy state
                setCompilationError(result.error || "Failed to fix kernel");
                console.log("❌ Failed to fix kernel:", result.error);
                setIsCompiling(false);
            }
        } catch (error) {
            console.error("Kernel fixing failed:", error);

            // Update current tab's compilation error
            if (activeTab === "cuda") {
                setCudaCompilationError("Failed to fix kernel");
            } else {
                setPytorchCompilationError("Failed to fix kernel");
            }

            // Update legacy state
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
                        <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
                            {selectedProblem === "custom" ? (
                                <div className="relative">
                                    <textarea
                                        value={customProblemCode}
                                        onChange={(e) => setCustomProblemCode(e.target.value)}
                                        className="w-full h-24 bg-gray-900 text-green-400 font-mono text-xs p-3 resize-none overflow-y-auto focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
                                        style={{
                                            fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
                                            lineHeight: "1.4",
                                            tabSize: 4,
                                        }}
                                        placeholder="Paste your custom PyTorch model code here..."
                                    />
                                    <div className="absolute top-1 right-1">
                                        <span className="text-xs text-gray-500 bg-gray-800 px-2 py-1 rounded">
                                            Editable
                                        </span>
                                    </div>
                                </div>
                            ) : (
                                <div className="h-24 overflow-y-auto">
                                    <pre className="text-xs text-green-400 font-mono leading-tight whitespace-pre p-3">
                                        <code>{currentCode}</code>
                                    </pre>
                                </div>
                            )}
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
                            className="w-24 px-4 py-2 flex items-center justify-center gap-2 rounded-lg bg-primary-500 hover:bg-primary-600 text-white disabled:opacity-50 disabled:cursor-not-allowed"
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
                    {generatedCudaCode || generatedPytorchCode ? (
                        <div className="h-full flex flex-col">
                            <div className="flex items-center justify-between mb-3 flex-shrink-0">
                                {/* Left side: Title + Tab Switches */}
                                <div className="flex items-center gap-4">
                                    <h3 className="text-sm font-semibold text-gray-900">
                                        Generated Kernel
                                    </h3>

                                    {/* Tab Switches - Fixed on left */}
                                    <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
                                        <button
                                            onClick={() => setActiveTab("cuda")}
                                            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                                                activeTab === "cuda"
                                                    ? "bg-white text-gray-900 shadow-sm"
                                                    : "text-gray-600 hover:text-gray-900"
                                            }`}
                                        >
                                            CUDA
                                        </button>
                                        <button
                                            onClick={() => setActiveTab("pytorch")}
                                            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                                                activeTab === "pytorch"
                                                    ? "bg-white text-gray-900 shadow-sm"
                                                    : "text-gray-600 hover:text-gray-900"
                                            }`}
                                        >
                                            PyTorch
                                        </button>
                                    </div>
                                </div>

                                {/* Right side: Action Buttons */}
                                <div className="flex items-center gap-2">
                                    {isCompiling && (
                                        <div className="flex items-center text-xs text-blue-600">
                                            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse mr-2"></div>
                                            Compiling... ({getCurrentCompilationAttempts()}/
                                            {maxCompilationAttempts})
                                        </div>
                                    )}
                                    {getCurrentCompilationStatus() === "success" && (
                                        <div className="flex items-center text-xs text-green-600">
                                            <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                                            ✅ Compiled Successfully
                                        </div>
                                    )}
                                    {getCurrentCompilationStatus() === "error" &&
                                        getCurrentCompilationAttempts() >=
                                            maxCompilationAttempts && (
                                            <div className="flex items-center text-xs text-red-600">
                                                <div className="w-2 h-2 bg-red-500 rounded-full mr-2"></div>
                                                ❌ Compilation Failed
                                            </div>
                                        )}
                                    {isCompiling ? (
                                        <button
                                            onClick={handleStopCompilation}
                                            className="w-16 px-2 py-1 text-xs rounded text-white bg-red-500 hover:bg-red-600 flex items-center justify-center"
                                            title="Stop compilation"
                                        >
                                            Stop
                                        </button>
                                    ) : (
                                        <button
                                            onClick={handleCompile}
                                            disabled={
                                                !getCurrentCode().trim() ||
                                                isGenerating ||
                                                !supportsCompilation
                                            }
                                            className={`w-16 px-2 py-1 text-xs rounded text-white flex items-center justify-center ${
                                                !supportsCompilation
                                                    ? "bg-gray-400 cursor-not-allowed"
                                                    : getCurrentCompilationStatus() === "success"
                                                    ? "bg-green-500 hover:bg-green-600"
                                                    : getCurrentCompilationStatus() === "error" &&
                                                      getCurrentCompilationAttempts() >=
                                                          maxCompilationAttempts
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
                                                ? "Blocked"
                                                : getCurrentCompilationStatus() === "success"
                                                ? "Compiled"
                                                : getCurrentCompilationStatus() === "error" &&
                                                  getCurrentCompilationAttempts() >=
                                                      maxCompilationAttempts
                                                ? "Failed"
                                                : "Compile"}
                                        </button>
                                    )}
                                    <button
                                        onClick={() => {
                                            navigator.clipboard.writeText(getCurrentCode());
                                            alert("Code copied to clipboard!");
                                        }}
                                        className="px-2 py-1 text-xs bg-gray-200 hover:bg-gray-300 rounded text-gray-700"
                                    >
                                        Copy Code
                                    </button>
                                </div>
                            </div>
                            <div className="flex-1 bg-gray-900 rounded-lg overflow-hidden border border-gray-700 min-h-0 relative">
                                <textarea
                                    value={getCurrentCode()}
                                    onChange={(e) => setCurrentCode(e.target.value)}
                                    className="w-full h-full text-sm text-green-400 font-mono leading-relaxed p-4 bg-transparent border-none outline-none resize-none overflow-y-auto"
                                    style={{
                                        fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
                                        lineHeight: "1.6",
                                        tabSize: 4,
                                    }}
                                    spellCheck={false}
                                />
                                <div className="absolute top-2 right-2">
                                    <span className="text-xs text-gray-400 bg-gray-800 px-2 py-1 rounded border border-gray-600">
                                        {activeTab === "cuda" ? "CUDA C++" : "PyTorch Python"}
                                    </span>
                                </div>
                            </div>
                            {!supportsCompilation && (
                                <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded-lg">
                                    <div className="flex items-center">
                                        <div className="w-4 h-4 text-yellow-500 mr-2">⚠️</div>
                                        <span className="text-sm text-yellow-700">
                                            {activeTab === "pytorch"
                                                ? "PyTorch CUDA extension compilation is temporarily disabled while we fix compilation issues. You can still generate and view code."
                                                : "Compilation is currently unavailable for this backend - Generate button still works"}
                                        </span>
                                    </div>
                                </div>
                            )}
                            {getCurrentCompilationError() && (
                                <div className="mt-2 p-3 bg-red-50 border border-red-200 rounded-lg">
                                    <div className="flex items-start">
                                        <div className="flex-shrink-0">
                                            <div className="w-4 h-4 text-red-500">❌</div>
                                        </div>
                                        <div className="ml-3 flex-1">
                                            <div className="flex items-center gap-2 mb-1">
                                                <h4 className="text-sm font-medium text-red-800">
                                                    {activeTab === "cuda" ? "CUDA" : "PyTorch"}{" "}
                                                    Compilation Error
                                                </h4>
                                                {getCurrentCompilationAttempts() <
                                                    maxCompilationAttempts && (
                                                    <span className="text-xs text-red-600">
                                                        🔧 Attempting to fix automatically... (
                                                        {getCurrentCompilationAttempts()}/
                                                        {maxCompilationAttempts})
                                                    </span>
                                                )}
                                            </div>
                                            <div className="text-sm text-red-700">
                                                <div className="bg-red-100 border border-red-300 rounded p-2 max-h-20 overflow-y-auto">
                                                    <pre className="whitespace-pre-wrap font-mono text-xs leading-tight">
                                                        {getCurrentCompilationError()}
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
                    kernelCode={getCurrentCode()}
                    activeTab={activeTab}
                    hardware={hardware}
                    backend={backend}
                    isAnalyzing={false}
                    compilationStatus={getCurrentCompilationStatus()}
                    onAnalysisComplete={(result) => {
                        console.log("Critic analysis completed:", result);
                    }}
                    onCodeFixed={(fixedCode) => {
                        setCurrentCode(fixedCode);
                        // Reset compilation state since we have new code
                        resetCurrentCompilationState();
                        setCompilationStatus("idle");
                        setCompilationError("");
                        setCompilationAttempts(0);
                        setIsCompiling(false);
                    }}
                />
            </div>
        </div>
    );
}

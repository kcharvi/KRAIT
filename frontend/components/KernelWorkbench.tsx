"use client";

import { useState, useEffect } from "react";
import { ChevronDown, Send, ChevronUp } from "lucide-react";
import { Problem, parseProblemMDX } from "../utils/parseMDX";
import CriticPanel from "./CriticPanel";

export default function KernelWorkbench() {
    const [backend, setBackend] = useState("Triton");
    const [hardware, setHardware] = useState("AMD MI300X");
    const [attempts, setAttempts] = useState(1);
    const [prompt, setPrompt] = useState("");
    const [selectedProblem, setSelectedProblem] = useState<string>("");
    const [problems, setProblems] = useState<Problem[]>([]);
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);
    const [isLoadingProblems, setIsLoadingProblems] = useState(true);
    const [customCode, setCustomCode] = useState<string>("");
    const [isCustomProblem, setIsCustomProblem] = useState(false);
    const [isGenerating, setIsGenerating] = useState(false);
    const [generatedCode, setGeneratedCode] = useState<string>("");
    const [isStreaming, setIsStreaming] = useState(false);

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

            // Add custom problem option
            const customProblem: Problem = {
                id: "custom",
                name: "Custom Problem",
                code: "",
                description: "Write your own custom kernel code",
                requirements: [],
            };

            setProblems([...loadedProblems, customProblem]);
            setIsLoadingProblems(false);
        };

        loadProblems();
    }, []);

    const selectedProblemData = problems.find((p) => p.id === selectedProblem);
    const currentCode = isCustomProblem ? customCode : selectedProblemData?.code || "";

    // Update custom problem state when selection changes
    useEffect(() => {
        setIsCustomProblem(selectedProblem === "custom");
        if (selectedProblem === "custom" && !customCode) {
            setCustomCode(
                "// Write your custom kernel code here\nimport torch\nimport torch.nn as nn\n\nclass CustomKernel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        \n    def forward(self, x):\n        return x"
            );
        }
    }, [selectedProblem, customCode]);

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
        setIsStreaming(true);
        setGeneratedCode("");

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
                console.log("Starting streaming simulation...");
                // Simulate streaming by displaying code character by character
                const code = data.optimized_code;
                setGeneratedCode("");
                setIsStreaming(true);

                for (let i = 0; i <= code.length; i++) {
                    await new Promise((resolve) => setTimeout(resolve, 10)); // 10ms delay
                    setGeneratedCode(code.substring(0, i));
                }

                console.log("Streaming simulation complete");
                setIsStreaming(false);
            } else {
                console.error("No optimized_code in response:", data);
                throw new Error("No optimized code in response");
            }
        } catch (error) {
            console.error("Error generating kernel:", error);
            const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
            alert(`Failed to generate kernel: ${errorMessage}`);
            setIsStreaming(false);
        } finally {
            setIsGenerating(false);
        }
    };

    return (
        <div className="h-full grid grid-cols-1 lg:grid-cols-2 gap-4 p-4">
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
                                onChange={(e) => setBackend(e.target.value)}
                            >
                                <option>Triton</option>
                                <option>CUDA</option>
                                <option>OpenCL</option>
                            </select>
                            <select
                                className="border border-gray-300 rounded-md px-2 py-1 text-sm"
                                value={hardware}
                                onChange={(e) => setHardware(e.target.value)}
                            >
                                <option>AMD MI300X</option>
                                <option>NVIDIA A100</option>
                                <option>NVIDIA H100</option>
                                <option>CPU</option>
                            </select>
                        </div>
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
                                    {isStreaming && (
                                        <div className="flex items-center text-xs text-primary-600">
                                            <div className="w-2 h-2 bg-primary-500 rounded-full animate-pulse mr-2"></div>
                                            Generating...
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
                                </div>
                            </div>
                            <div className="flex-1 bg-gray-900 rounded-lg overflow-hidden border border-gray-700 min-h-0">
                                <div className="h-full overflow-auto">
                                    <pre className="text-sm text-green-400 font-mono leading-relaxed p-4 whitespace-pre">
                                        <code>
                                            {generatedCode.split("\n").map((line, index) => (
                                                <div key={index} className="flex hover:bg-gray-800">
                                                    <span className="text-gray-500 mr-4 select-none w-8 text-right flex-shrink-0">
                                                        {index + 1}
                                                    </span>
                                                    <span className="flex-1 text-gray-100">
                                                        {line}
                                                    </span>
                                                </div>
                                            ))}
                                        </code>
                                    </pre>
                                </div>
                            </div>
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

                {/* Bottom prompt composer */}
                <div className="border-t border-gray-200 p-4 space-y-4 flex-shrink-0">
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
                            <div className="absolute bottom-full left-0 right-0 mb-1 bg-white border border-gray-300 rounded-lg shadow-lg z-20 max-h-48 overflow-y-auto">
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
                            {isCustomProblem ? (
                                <textarea
                                    value={customCode}
                                    onChange={(e) => setCustomCode(e.target.value)}
                                    className="w-full h-full bg-transparent text-green-400 font-mono text-xs leading-tight resize-none border-none outline-none"
                                    placeholder="Write your custom kernel code here..."
                                />
                            ) : (
                                <pre className="text-xs text-green-400 font-mono leading-tight whitespace-pre">
                                    <code>{currentCode}</code>
                                </pre>
                            )}
                        </div>
                    )}
                    {selectedProblem && !currentCode && (
                        <div className="bg-gray-100 rounded-lg p-3 h-24 flex items-center justify-center text-sm text-gray-500">
                            {isCustomProblem
                                ? "Start writing your custom code..."
                                : "No code available for this problem"}
                        </div>
                    )}
                    {!selectedProblem && (
                        <div className="bg-gray-100 rounded-lg p-3 h-24 flex items-center justify-center text-sm text-gray-500">
                            Select a problem to view code
                        </div>
                    )}

                    {/* Prompt Input */}
                    <div className="flex items-end gap-2">
                        <textarea
                            rows={2}
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            placeholder="Describe the kernel you want to generate…"
                            className="flex-1 resize-none px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                        />
                        <button
                            className="h-10 w-10 flex items-center justify-center rounded-lg bg-primary-500 hover:bg-primary-600 text-white disabled:opacity-50 disabled:cursor-not-allowed"
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
                                <Send className="h-5 w-5" />
                            )}
                        </button>
                    </div>
                </div>
            </div>

            {/* Right column - Critic Agent */}
            <div className="flex flex-col bg-white rounded-lg shadow-sm border border-gray-200 h-full min-h-0">
                <CriticPanel
                    kernelCode={generatedCode}
                    hardware={hardware}
                    backend={backend}
                    isAnalyzing={false}
                    onAnalysisComplete={(result) => {
                        console.log("Critic analysis completed:", result);
                    }}
                />
            </div>
        </div>
    );
}

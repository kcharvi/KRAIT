"use client";

import { useState } from "react";
import TopNavigation from "@/components/TopNavigation";
import KernelWorkbench from "@/components/KernelWorkbench";
import MetricsDashboard from "@/components/MetricsDashboard";

export default function Home() {
    const [activeTab, setActiveTab] = useState("kernel-workbench");

    const sampleData = [
        {
            id: "1",
            kernel_name: "matrix_multiply",
            hardware: "NVIDIA A100",
            backend: "CUDA",
            score: 85.5,
            flops_total: 1024000000,
            runtime_ms: 12.5,
            memory_usage_kb: 2048,
            bound_type: "compute",
            correctness_status: "pass",
            suggestions_count: 3,
            analysis_time_ms: 150,
            optimizations: ["vectorization", "shared_memory", "tiling"],
            timestamp: new Date().toISOString(),
        },
        {
            id: "2",
            kernel_name: "convolution_2d",
            hardware: "AMD MI300X",
            backend: "Triton",
            score: 92.3,
            flops_total: 2048000000,
            runtime_ms: 8.7,
            memory_usage_kb: 4096,
            bound_type: "memory",
            correctness_status: "pass",
            suggestions_count: 1,
            analysis_time_ms: 200,
            optimizations: ["memory_coalescing", "tensor_cores"],
            timestamp: new Date(Date.now() - 3600000).toISOString(),
        },
    ];

    return (
        <div className="h-screen flex flex-col">
            <TopNavigation activeTab={activeTab} onTabChange={setActiveTab} />
            <main className="flex-1 overflow-hidden">
                <div className={`h-full ${activeTab === "kernel-workbench" ? "block" : "hidden"}`}>
                    <KernelWorkbench />
                </div>

                <div className={`h-full ${activeTab === "visualizations" ? "block" : "hidden"}`}>
                    <div className="flex items-center justify-center h-full bg-gray-50">
                        <div className="text-center">
                            <div className="mx-auto mb-4 h-16 w-16 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center">
                                <svg
                                    className="h-8 w-8"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                                    />
                                </svg>
                            </div>
                            <h2 className="text-2xl font-bold text-gray-900 mb-2">
                                Visualizations
                            </h2>
                            <p className="text-gray-600 mb-4">
                                Performance charts and interactive graphs
                            </p>
                            <p className="text-sm text-gray-500">Coming soon...</p>
                        </div>
                    </div>
                </div>

                <div className={`h-full ${activeTab === "metrics" ? "block" : "hidden"}`}>
                    <MetricsDashboard data={sampleData} />
                </div>
            </main>
        </div>
    );
}

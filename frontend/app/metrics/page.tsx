"use client";

import { useState, useEffect } from "react";
import MetricsDashboard from "../../components/MetricsDashboard";

interface MetricsData {
    id: string;
    timestamp: string;
    kernel_name: string;
    hardware: string;
    backend: string;
    score: number;
    flops_total: number;
    runtime_ms: number;
    memory_usage_kb: number;
    bound_type: string;
    correctness_status: string;
    suggestions_count: number;
    analysis_time_ms: number;
    optimizations: string[];
}

export default function MetricsPage() {
    const [data, setData] = useState<MetricsData[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadMetricsData();
    }, []);

    const loadMetricsData = async () => {
        try {
            setLoading(true);
            setError(null);

            // For now, generate sample data
            // In a real implementation, this would fetch from your backend
            const sampleData: MetricsData[] = [
                {
                    id: "1",
                    timestamp: new Date(Date.now() - 3600000).toISOString(),
                    kernel_name: "matrix_multiply",
                    hardware: "NVIDIA H100",
                    backend: "CUDA",
                    score: 85,
                    flops_total: 1000000000,
                    runtime_ms: 2.5,
                    memory_usage_kb: 1024,
                    bound_type: "memory",
                    correctness_status: "likely_correct",
                    suggestions_count: 3,
                    analysis_time_ms: 150,
                    optimizations: ["tiling", "vectorization"],
                },
                {
                    id: "2",
                    timestamp: new Date(Date.now() - 7200000).toISOString(),
                    kernel_name: "convolution",
                    hardware: "AMD MI300X",
                    backend: "OpenCL",
                    score: 72,
                    flops_total: 2000000000,
                    runtime_ms: 4.2,
                    memory_usage_kb: 2048,
                    bound_type: "compute",
                    correctness_status: "potential_issues",
                    suggestions_count: 5,
                    analysis_time_ms: 200,
                    optimizations: ["shared_memory"],
                },
                {
                    id: "3",
                    timestamp: new Date(Date.now() - 10800000).toISOString(),
                    kernel_name: "reduction",
                    hardware: "NVIDIA A100",
                    backend: "CUDA",
                    score: 91,
                    flops_total: 500000000,
                    runtime_ms: 1.8,
                    memory_usage_kb: 512,
                    bound_type: "memory",
                    correctness_status: "likely_correct",
                    suggestions_count: 1,
                    analysis_time_ms: 120,
                    optimizations: ["tiling", "vectorization", "tensor_cores"],
                },
                {
                    id: "4",
                    timestamp: new Date(Date.now() - 14400000).toISOString(),
                    kernel_name: "matrix_multiply",
                    hardware: "NVIDIA H100",
                    backend: "Triton",
                    score: 78,
                    flops_total: 1500000000,
                    runtime_ms: 3.1,
                    memory_usage_kb: 1536,
                    bound_type: "memory",
                    correctness_status: "potential_issues",
                    suggestions_count: 4,
                    analysis_time_ms: 180,
                    optimizations: ["tiling"],
                },
                {
                    id: "5",
                    timestamp: new Date(Date.now() - 18000000).toISOString(),
                    kernel_name: "convolution",
                    hardware: "AMD MI300X",
                    backend: "OpenCL",
                    score: 65,
                    flops_total: 3000000000,
                    runtime_ms: 5.5,
                    memory_usage_kb: 3072,
                    bound_type: "compute",
                    correctness_status: "failed_checks",
                    suggestions_count: 8,
                    analysis_time_ms: 250,
                    optimizations: [],
                },
            ];

            setData(sampleData);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load metrics data");
        } finally {
            setLoading(false);
        }
    };

    const handleExport = (format: "csv" | "json") => {
        console.log(`Exporting data as ${format}`);
        // Export functionality is handled by the MetricsTable component
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500 mx-auto mb-4"></div>
                    <p className="text-gray-600">Loading metrics data...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center">
                <div className="text-center">
                    <div className="text-red-500 text-4xl mb-4">⚠️</div>
                    <h2 className="text-xl font-semibold text-gray-900 mb-2">
                        Error Loading Metrics
                    </h2>
                    <p className="text-gray-600 mb-4">{error}</p>
                    <button
                        onClick={loadMetricsData}
                        className="px-4 py-2 bg-primary-500 text-white rounded hover:bg-primary-600"
                    >
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <MetricsDashboard data={data} onExport={handleExport} />
            </div>
        </div>
    );
}

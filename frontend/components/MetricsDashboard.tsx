"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import {
    BarChart3,
    TrendingUp,
    Clock,
    MemoryStick,
    Cpu,
    AlertTriangle,
    CheckCircle,
    XCircle,
    Info,
} from "lucide-react";
import MetricsTable from "./MetricsTable";
const PerformanceChart = dynamic(() => import("./PerformanceChart"), { ssr: false });

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

interface MetricsDashboardProps {
    data: MetricsData[];
    onExport?: (format: "csv" | "json") => void;
}

export default function MetricsDashboard({ data, onExport }: MetricsDashboardProps) {
    const [selectedTab, setSelectedTab] = useState<"overview" | "charts" | "table">("overview");
    const [selectedChartType, setSelectedChartType] = useState<
        "score" | "runtime" | "flops" | "memory" | "suggestions"
    >("score");
    const [selectedChartStyle, setSelectedChartStyle] = useState<"bar" | "line" | "pie">("bar");
    const [selectedGroupBy, setSelectedGroupBy] = useState<"hardware" | "backend" | "kernel_name">(
        "hardware"
    );

    // Calculate summary statistics
    const summaryStats = {
        totalAnalyses: data.length,
        averageScore: data.reduce((sum, item) => sum + item.score, 0) / data.length || 0,
        averageRuntime: data.reduce((sum, item) => sum + item.runtime_ms, 0) / data.length || 0,
        averageFlops: data.reduce((sum, item) => sum + item.flops_total, 0) / data.length || 0,
        averageMemory: data.reduce((sum, item) => sum + item.memory_usage_kb, 0) / data.length || 0,
        totalSuggestions: data.reduce((sum, item) => sum + item.suggestions_count, 0),
        correctnessDistribution: {
            likely_correct: data.filter((item) => item.correctness_status === "likely_correct")
                .length,
            potential_issues: data.filter((item) => item.correctness_status === "potential_issues")
                .length,
            failed_checks: data.filter((item) => item.correctness_status === "failed_checks")
                .length,
        },
        boundTypeDistribution: {
            memory: data.filter((item) => item.bound_type === "memory").length,
            compute: data.filter((item) => item.bound_type === "compute").length,
            unknown: data.filter((item) => item.bound_type === "unknown").length,
        },
        hardwareDistribution: data.reduce((acc, item) => {
            acc[item.hardware] = (acc[item.hardware] || 0) + 1;
            return acc;
        }, {} as Record<string, number>),
        backendDistribution: data.reduce((acc, item) => {
            acc[item.backend] = (acc[item.backend] || 0) + 1;
            return acc;
        }, {} as Record<string, number>),
    };

    const getCorrectnessIcon = (status: string) => {
        switch (status) {
            case "likely_correct":
                return <CheckCircle className="w-4 h-4 text-green-500" />;
            case "potential_issues":
                return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
            case "failed_checks":
                return <XCircle className="w-4 h-4 text-red-500" />;
            default:
                return <Info className="w-4 h-4 text-gray-500" />;
        }
    };

    const getScoreColor = (score: number) => {
        if (score >= 80) return "text-green-600";
        if (score >= 60) return "text-yellow-600";
        return "text-red-600";
    };

    const renderOverview = () => (
        <div className="space-y-8">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-gray-600 mb-1">Total Analyses</p>
                            <p className="text-3xl font-bold text-gray-900">
                                {summaryStats.totalAnalyses}
                            </p>
                        </div>
                        <div className="p-3 bg-blue-50 rounded-lg">
                            <BarChart3 className="w-6 h-6 text-blue-600" />
                        </div>
                    </div>
                </div>

                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-gray-600 mb-1">Average Score</p>
                            <p
                                className={`text-3xl font-bold ${getScoreColor(
                                    summaryStats.averageScore
                                )}`}
                            >
                                {summaryStats.averageScore.toFixed(1)}
                            </p>
                        </div>
                        <div className="p-3 bg-green-50 rounded-lg">
                            <TrendingUp className="w-6 h-6 text-green-600" />
                        </div>
                    </div>
                </div>

                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-gray-600 mb-1">Avg Runtime</p>
                            <p className="text-3xl font-bold text-gray-900">
                                {summaryStats.averageRuntime.toFixed(1)}ms
                            </p>
                        </div>
                        <div className="p-3 bg-purple-50 rounded-lg">
                            <Clock className="w-6 h-6 text-purple-600" />
                        </div>
                    </div>
                </div>

                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-gray-600 mb-1">
                                Total Suggestions
                            </p>
                            <p className="text-3xl font-bold text-gray-900">
                                {summaryStats.totalSuggestions}
                            </p>
                        </div>
                        <div className="p-3 bg-orange-50 rounded-lg">
                            <AlertTriangle className="w-6 h-6 text-orange-600" />
                        </div>
                    </div>
                </div>
            </div>

            {/* Distribution Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Correctness Distribution */}
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                    <h3 className="text-lg font-semibold text-gray-900 mb-6">
                        Correctness Distribution
                    </h3>
                    <div className="space-y-3">
                        {Object.entries(summaryStats.correctnessDistribution).map(
                            ([status, count]) => (
                                <div key={status} className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        {getCorrectnessIcon(status)}
                                        <span className="text-sm font-medium text-gray-700">
                                            {status
                                                .replace("_", " ")
                                                .replace(/\b\w/g, (l) => l.toUpperCase())}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <div className="w-24 bg-gray-200 rounded-full h-2">
                                            <div
                                                className="bg-blue-500 h-2 rounded-full"
                                                style={{
                                                    width: `${
                                                        (count / summaryStats.totalAnalyses) * 100
                                                    }%`,
                                                }}
                                            />
                                        </div>
                                        <span className="text-sm text-gray-600 w-8 text-right">
                                            {count}
                                        </span>
                                    </div>
                                </div>
                            )
                        )}
                    </div>
                </div>

                {/* Hardware Distribution */}
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                    <h3 className="text-lg font-semibold text-gray-900 mb-6">
                        Hardware Distribution
                    </h3>
                    <div className="space-y-3">
                        {Object.entries(summaryStats.hardwareDistribution).map(
                            ([hardware, count]) => (
                                <div key={hardware} className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        <Cpu className="w-4 h-4 text-gray-500" />
                                        <span className="text-sm font-medium text-gray-700">
                                            {hardware}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <div className="w-24 bg-gray-200 rounded-full h-2">
                                            <div
                                                className="bg-green-500 h-2 rounded-full"
                                                style={{
                                                    width: `${
                                                        (count / summaryStats.totalAnalyses) * 100
                                                    }%`,
                                                }}
                                            />
                                        </div>
                                        <span className="text-sm text-gray-600 w-8 text-right">
                                            {count}
                                        </span>
                                    </div>
                                </div>
                            )
                        )}
                    </div>
                </div>
            </div>

            {/* Performance Metrics */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-6">Performance Metrics</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="text-center p-6 bg-blue-50 rounded-xl border border-blue-100">
                        <div className="p-3 bg-blue-100 rounded-lg w-fit mx-auto mb-4">
                            <MemoryStick className="w-6 h-6 text-blue-600" />
                        </div>
                        <div className="text-3xl font-bold text-gray-900 mb-1">
                            {(summaryStats.averageMemory / 1024).toFixed(1)} MB
                        </div>
                        <div className="text-sm font-medium text-gray-600">Avg Memory Usage</div>
                    </div>
                    <div className="text-center p-6 bg-purple-50 rounded-xl border border-purple-100">
                        <div className="p-3 bg-purple-100 rounded-lg w-fit mx-auto mb-4">
                            <Cpu className="w-6 h-6 text-purple-600" />
                        </div>
                        <div className="text-3xl font-bold text-gray-900 mb-1">
                            {(summaryStats.averageFlops / 1e9).toFixed(1)}B
                        </div>
                        <div className="text-sm font-medium text-gray-600">Avg FLOPs</div>
                    </div>
                    <div className="text-center p-6 bg-orange-50 rounded-xl border border-orange-100">
                        <div className="p-3 bg-orange-100 rounded-lg w-fit mx-auto mb-4">
                            <Clock className="w-6 h-6 text-orange-600" />
                        </div>
                        <div className="text-3xl font-bold text-gray-900 mb-1">
                            {summaryStats.averageRuntime.toFixed(1)}ms
                        </div>
                        <div className="text-sm font-medium text-gray-600">Avg Runtime</div>
                    </div>
                </div>
            </div>
        </div>
    );

    const renderCharts = () => (
        <div className="space-y-6">
            {/* Chart Controls */}
            <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
                <div className="flex flex-wrap items-center gap-4">
                    <div className="flex items-center gap-2">
                        <label className="text-sm font-medium text-gray-700">Metric:</label>
                        <select
                            value={selectedChartType}
                            onChange={(e) => setSelectedChartType(e.target.value as any)}
                            className="px-3 py-1 border border-gray-300 rounded text-sm"
                        >
                            <option value="score">Score</option>
                            <option value="runtime">Runtime</option>
                            <option value="flops">FLOPs</option>
                            <option value="memory">Memory Usage</option>
                            <option value="suggestions">Suggestions</option>
                        </select>
                    </div>
                    <div className="flex items-center gap-2">
                        <label className="text-sm font-medium text-gray-700">Chart Type:</label>
                        <select
                            value={selectedChartStyle}
                            onChange={(e) => setSelectedChartStyle(e.target.value as any)}
                            className="px-3 py-1 border border-gray-300 rounded text-sm"
                        >
                            <option value="bar">Bar Chart</option>
                            <option value="line">Line Chart</option>
                            <option value="pie">Pie Chart</option>
                        </select>
                    </div>
                    <div className="flex items-center gap-2">
                        <label className="text-sm font-medium text-gray-700">Group By:</label>
                        <select
                            value={selectedGroupBy}
                            onChange={(e) => setSelectedGroupBy(e.target.value as any)}
                            className="px-3 py-1 border border-gray-300 rounded text-sm"
                        >
                            <option value="hardware">Hardware</option>
                            <option value="backend">Backend</option>
                            <option value="kernel_name">Kernel Name</option>
                        </select>
                    </div>
                </div>
            </div>

            {/* Performance Chart */}
            {/* PerformanceChart is dynamically imported client-side */}
            <PerformanceChart
                data={data}
                type={selectedChartType}
                chartType={selectedChartStyle}
                groupBy={selectedGroupBy}
            />

            {/* Additional Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <PerformanceChart data={data} type="score" chartType="bar" groupBy="hardware" />
                <PerformanceChart data={data} type="runtime" chartType="line" groupBy="backend" />
            </div>
        </div>
    );

    const renderTable = () => <MetricsTable data={data} onExport={onExport} />;

    return (
        <div className="h-full bg-gray-50 overflow-hidden">
            <div className="h-full overflow-y-auto p-6">
                <div className="max-w-7xl mx-auto space-y-6">
                    {/* Header */}
                    <div className="flex items-center justify-between">
                        <h2 className="text-2xl font-bold text-gray-900">Analysis Dashboard</h2>
                        <div className="flex items-center gap-2 text-sm text-gray-600">
                            <span>Last updated: {new Date().toISOString().split("T")[0]}</span>
                        </div>
                    </div>

                    {/* Tabs */}
                    <div className="border-b border-gray-200">
                        <nav className="-mb-px flex space-x-8">
                            {[
                                { id: "overview", label: "Overview", icon: BarChart3 },
                                { id: "charts", label: "Charts", icon: TrendingUp },
                                { id: "table", label: "Table", icon: Cpu },
                            ].map((tab) => (
                                <button
                                    key={tab.id}
                                    onClick={() => setSelectedTab(tab.id as any)}
                                    className={`flex items-center gap-2 py-2 px-1 border-b-2 font-medium text-sm ${
                                        selectedTab === tab.id
                                            ? "border-primary-500 text-primary-600"
                                            : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                                    }`}
                                >
                                    <tab.icon className="w-4 h-4" />
                                    {tab.label}
                                </button>
                            ))}
                        </nav>
                    </div>

                    {/* Content */}
                    <div className="min-h-96">
                        {selectedTab === "overview" && renderOverview()}
                        {selectedTab === "charts" && renderCharts()}
                        {selectedTab === "table" && renderTable()}
                    </div>
                </div>
            </div>
        </div>
    );
}

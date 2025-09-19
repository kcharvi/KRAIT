"use client";

import { useState, useMemo } from "react";

interface PerformanceData {
    id: string;
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
    timestamp: string;
}

interface PerformanceChartProps {
    data: PerformanceData[];
    type: "score" | "runtime" | "flops" | "memory" | "suggestions";
    chartType: "bar" | "line" | "pie";
    groupBy?: "hardware" | "backend" | "kernel_name" | "timestamp";
}

export default function PerformanceChart({
    data,
    type,
    chartType,
    groupBy = "hardware",
}: PerformanceChartProps) {
    const [selectedData, setSelectedData] = useState<any | null>(null);

    // Hoisted helper so it's available during useMemo evaluation
    function getValue(item: PerformanceData, valueType: string): number {
        switch (valueType) {
            case "score":
                return item.score;
            case "runtime":
                return item.runtime_ms;
            case "flops":
                return item.flops_total;
            case "memory":
                return item.memory_usage_kb;
            case "suggestions":
                return item.suggestions_count;
            default:
                return 0;
        }
    }

    // Process data for chart
    const chartData = useMemo(() => {
        if (chartType === "pie") {
            // For pie charts, group by the specified field
            const grouped = data.reduce((acc, item) => {
                const key = (item as any)[groupBy as "hardware" | "backend" | "kernel_name"];
                if (!acc[key]) {
                    acc[key] = { name: key, value: 0, count: 0 } as any;
                }
                acc[key].value += getValue(item, type);
                acc[key].count += 1;
                return acc;
            }, {} as Record<string, { name: string; value: number; count: number }>);

            return Object.values(grouped).map((item) => ({
                ...(item as any),
                value: (item as any).value / (item as any).count, // Average value
                label: `${(item as any).name} (${(item as any).count})`,
            }));
        } else {
            // For bar/line charts, group by timestamp or groupBy field
            const grouped = data.reduce((acc, item) => {
                const key =
                    groupBy === "timestamp"
                        ? item.timestamp
                        : (item as any)[groupBy as "hardware" | "backend" | "kernel_name"];
                if (!acc[key]) {
                    acc[key] = {
                        name: key,
                        value: 0,
                        count: 0,
                        timestamp: item.timestamp,
                        kernel_name: item.kernel_name,
                        hardware: item.hardware,
                        backend: item.backend,
                    } as any;
                }
                acc[key].value += getValue(item, type);
                acc[key].count += 1;
                return acc;
            }, {} as Record<string, any>);

            return Object.values(grouped)
                .map((item: any) => ({
                    ...item,
                    value: item.value / item.count, // Average value
                    displayName:
                        groupBy === "timestamp"
                            ? new Date(item.timestamp).toISOString().split("T")[0]
                            : item.name,
                }))
                .sort((a: any, b: any) => {
                    if (groupBy === "timestamp") {
                        return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
                    }
                    return String(a.name).localeCompare(String(b.name));
                });
        }
    }, [data, type, chartType, groupBy]);

    const getValueLabel = () => {
        switch (type) {
            case "score":
                return "Score";
            case "runtime":
                return "Runtime (ms)";
            case "flops":
                return "FLOPs";
            case "memory":
                return "Memory Usage (KB)";
            case "suggestions":
                return "Suggestions Count";
            default:
                return "Value";
        }
    };

    const getColor = (index: number) => {
        const colors = [
            "#3B82F6",
            "#EF4444",
            "#10B981",
            "#F59E0B",
            "#8B5CF6",
            "#06B6D4",
            "#84CC16",
            "#F97316",
            "#EC4899",
            "#6366F1",
        ];
        return colors[index % colors.length];
    };

    const handleDataClick = (data: any) => {
        setSelectedData(data);
    };

    const renderChart = () => {
        if (chartData.length === 0) {
            return (
                <div className="flex items-center justify-center h-64 text-gray-500">
                    <div className="text-center">
                        <div className="text-4xl mb-2">📊</div>
                        <p>No data available for visualization</p>
                    </div>
                </div>
            );
        }

        // Simple chart representation without external libs
        const maxValue = Math.max(...chartData.map((item: any) => item.value));

        return (
            <div className="h-64 w-full">
                {chartType === "bar" && (
                    <div className="flex items-end justify-between h-full space-x-2">
                        {chartData.map((item: any, index: number) => (
                            <div key={index} className="flex flex-col items-center flex-1">
                                <div
                                    className="w-full rounded-t cursor-pointer transition-colors"
                                    style={{
                                        height: `${(item.value / maxValue) * 200}px`,
                                        backgroundColor: getColor(index),
                                    }}
                                    onClick={() => handleDataClick(item)}
                                    title={`${item.displayName}: ${item.value.toFixed(2)}`}
                                />
                                <div className="text-xs text-gray-600 mt-2 text-center transform -rotate-45 origin-left">
                                    {item.displayName}
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {chartType === "line" && (
                    <div className="relative h-full">
                        <svg className="w-full h-full" viewBox="0 0 400 200">
                            <polyline
                                fill="none"
                                stroke="#3B82F6"
                                strokeWidth="2"
                                points={(chartData as any)
                                    .map(
                                        (item: any, index: number) =>
                                            `${
                                                (index / ((chartData as any).length - 1)) * 380 + 10
                                            },${200 - (item.value / maxValue) * 180 + 10}`
                                    )
                                    .join(" ")}
                                className="cursor-pointer"
                                onClick={() => handleDataClick((chartData as any)[0])}
                            />
                            {(chartData as any).map((item: any, index: number) => (
                                <circle
                                    key={index}
                                    cx={(index / ((chartData as any).length - 1)) * 380 + 10}
                                    cy={200 - (item.value / maxValue) * 180 + 10}
                                    r="4"
                                    fill="#3B82F6"
                                    className="cursor-pointer hover:r-5 transition-all"
                                    onClick={() => handleDataClick(item)}
                                />
                            ))}
                        </svg>
                    </div>
                )}

                {chartType === "pie" && (
                    <div className="flex items-center justify-center h-full">
                        <div className="grid grid-cols-2 gap-2 max-w-xs">
                            {(chartData as any).map((item: any, index: number) => (
                                <div
                                    key={index}
                                    className="flex items-center space-x-2 p-2 rounded cursor-pointer hover:bg-gray-100"
                                    onClick={() => handleDataClick(item)}
                                >
                                    <div
                                        className="w-4 h-4 rounded"
                                        style={{ backgroundColor: getColor(index) }}
                                    />
                                    <span className="text-sm">{item.name}</span>
                                    <span className="text-sm font-medium">
                                        {item.value.toFixed(1)}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                    {getValueLabel()} by {groupBy.charAt(0).toUpperCase() + groupBy.slice(1)}
                </h3>
                <div className="flex items-center gap-2 text-sm text-gray-600">
                    <span>
                        Average:{" "}
                        {(
                            (chartData as any).reduce(
                                (sum: number, item: any) => sum + item.value,
                                0
                            ) / (chartData as any).length || 0
                        ).toFixed(2)}
                    </span>
                    <span>Total: {(chartData as any).length} groups</span>
                </div>
            </div>

            {renderChart()}

            {selectedData && (
                <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                    <h4 className="font-medium text-gray-900 mb-2">Selected Data Point</h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>
                            <span className="text-gray-600">Group:</span>{" "}
                            {selectedData.name || selectedData.displayName}
                        </div>
                        <div>
                            <span className="text-gray-600">Value:</span>{" "}
                            {selectedData.value.toFixed(2)}
                        </div>
                        <div>
                            <span className="text-gray-600">Count:</span> {selectedData.count}
                        </div>
                        <div>
                            <span className="text-gray-600">Type:</span> {getValueLabel()}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

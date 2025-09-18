"use client";

import { useState, useMemo } from "react";
import {
    ChevronUp,
    ChevronDown,
    Search,
    Download,
    Filter,
    ArrowUpDown,
    ArrowUp,
    ArrowDown,
} from "lucide-react";

interface MetricData {
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

interface MetricsTableProps {
    data: MetricData[];
    onExport?: (format: "csv" | "json") => void;
}

type SortField = keyof MetricData;
type SortDirection = "asc" | "desc";

export default function MetricsTable({ data, onExport }: MetricsTableProps) {
    const [sortField, setSortField] = useState<SortField>("timestamp");
    const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
    const [searchTerm, setSearchTerm] = useState("");
    const [filterField, setFilterField] = useState<keyof MetricData | "">("");
    const [filterValue, setFilterValue] = useState("");
    const [selectedRows, setSelectedRows] = useState<Set<string>>(new Set());

    // Filter and sort data
    const filteredAndSortedData = useMemo(() => {
        let filtered = data;

        // Apply search filter
        if (searchTerm) {
            filtered = filtered.filter((item) =>
                Object.values(item).some((value) =>
                    String(value).toLowerCase().includes(searchTerm.toLowerCase())
                )
            );
        }

        // Apply field-specific filter
        if (filterField && filterValue) {
            filtered = filtered.filter((item) =>
                String(item[filterField]).toLowerCase().includes(filterValue.toLowerCase())
            );
        }

        // Apply sorting
        return filtered.sort((a, b) => {
            const aValue = a[sortField];
            const bValue = b[sortField];

            if (typeof aValue === "string" && typeof bValue === "string") {
                return sortDirection === "asc"
                    ? aValue.localeCompare(bValue)
                    : bValue.localeCompare(aValue);
            }

            if (typeof aValue === "number" && typeof bValue === "number") {
                return sortDirection === "asc" ? aValue - bValue : bValue - aValue;
            }

            return 0;
        });
    }, [data, searchTerm, filterField, filterValue, sortField, sortDirection]);

    const handleSort = (field: SortField) => {
        if (sortField === field) {
            setSortDirection(sortDirection === "asc" ? "desc" : "asc");
        } else {
            setSortField(field);
            setSortDirection("asc");
        }
    };

    const getSortIcon = (field: SortField) => {
        if (sortField !== field) {
            return <ArrowUpDown className="w-4 h-4 text-gray-400" />;
        }
        return sortDirection === "asc" ? (
            <ArrowUp className="w-4 h-4 text-gray-600" />
        ) : (
            <ArrowDown className="w-4 h-4 text-gray-600" />
        );
    };

    const handleSelectRow = (id: string) => {
        const newSelected = new Set(selectedRows);
        if (newSelected.has(id)) {
            newSelected.delete(id);
        } else {
            newSelected.add(id);
        }
        setSelectedRows(newSelected);
    };

    const handleSelectAll = () => {
        if (selectedRows.size === filteredAndSortedData.length) {
            setSelectedRows(new Set());
        } else {
            setSelectedRows(new Set(filteredAndSortedData.map((item) => item.id)));
        }
    };

    const exportData = (format: "csv" | "json") => {
        const selectedData =
            selectedRows.size > 0
                ? filteredAndSortedData.filter((item) => selectedRows.has(item.id))
                : filteredAndSortedData;

        if (format === "csv") {
            const headers = [
                "Timestamp",
                "Kernel Name",
                "Hardware",
                "Backend",
                "Score",
                "FLOPs Total",
                "Runtime (ms)",
                "Memory Usage (KB)",
                "Bound Type",
                "Correctness Status",
                "Suggestions Count",
                "Analysis Time (ms)",
                "Optimizations",
            ];

            const csvContent = [
                headers.join(","),
                ...selectedData.map((item) =>
                    [
                        item.timestamp,
                        `"${item.kernel_name}"`,
                        item.hardware,
                        item.backend,
                        item.score,
                        item.flops_total,
                        item.runtime_ms,
                        item.memory_usage_kb,
                        item.bound_type,
                        item.correctness_status,
                        item.suggestions_count,
                        item.analysis_time_ms,
                        `"${item.optimizations.join("; ")}"`,
                    ].join(",")
                ),
            ].join("\n");

            const blob = new Blob([csvContent], { type: "text/csv" });
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = `critic-metrics-${new Date().toISOString().split("T")[0]}.csv`;
            link.click();
            URL.revokeObjectURL(url);
        } else {
            const jsonContent = JSON.stringify(selectedData, null, 2);
            const blob = new Blob([jsonContent], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = `critic-metrics-${new Date().toISOString().split("T")[0]}.json`;
            link.click();
            URL.revokeObjectURL(url);
        }

        onExport?.(format);
    };

    const formatValue = (value: any, field: SortField) => {
        if (field === "timestamp") {
            return new Date(value).toLocaleString();
        }
        if (
            field === "score" ||
            field === "flops_total" ||
            field === "runtime_ms" ||
            field === "memory_usage_kb" ||
            field === "analysis_time_ms"
        ) {
            return typeof value === "number" ? value.toLocaleString() : value;
        }
        if (field === "optimizations") {
            return Array.isArray(value) ? value.join(", ") : value;
        }
        return String(value);
    };

    const getScoreColor = (score: number) => {
        if (score >= 80) return "text-green-600";
        if (score >= 60) return "text-yellow-600";
        return "text-red-600";
    };

    const getCorrectnessColor = (status: string) => {
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

    return (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200">
            {/* Header */}
            <div className="p-4 border-b border-gray-200">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900">Analysis Metrics</h3>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => exportData("csv")}
                            className="flex items-center gap-1 px-3 py-1 text-sm text-gray-600 hover:text-gray-800 border border-gray-300 rounded hover:bg-gray-50"
                        >
                            <Download className="w-4 h-4" />
                            CSV
                        </button>
                        <button
                            onClick={() => exportData("json")}
                            className="flex items-center gap-1 px-3 py-1 text-sm text-gray-600 hover:text-gray-800 border border-gray-300 rounded hover:bg-gray-50"
                        >
                            <Download className="w-4 h-4" />
                            JSON
                        </button>
                    </div>
                </div>

                {/* Filters */}
                <div className="flex items-center gap-4">
                    <div className="flex-1 relative">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                        <input
                            type="text"
                            placeholder="Search all fields..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                        />
                    </div>
                    <div className="flex items-center gap-2">
                        <Filter className="w-4 h-4 text-gray-400" />
                        <select
                            value={filterField}
                            onChange={(e) =>
                                setFilterField(e.target.value as keyof MetricData | "")
                            }
                            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                        >
                            <option value="">Filter by field</option>
                            <option value="hardware">Hardware</option>
                            <option value="backend">Backend</option>
                            <option value="correctness_status">Correctness</option>
                            <option value="bound_type">Bound Type</option>
                        </select>
                        {filterField && (
                            <input
                                type="text"
                                placeholder={`Filter by ${filterField}...`}
                                value={filterValue}
                                onChange={(e) => setFilterValue(e.target.value)}
                                className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                            />
                        )}
                    </div>
                </div>
            </div>

            {/* Table */}
            <div className="overflow-x-auto">
                <table className="w-full">
                    <thead className="bg-gray-50">
                        <tr>
                            <th className="px-4 py-3 text-left">
                                <input
                                    type="checkbox"
                                    checked={
                                        selectedRows.size === filteredAndSortedData.length &&
                                        filteredAndSortedData.length > 0
                                    }
                                    onChange={handleSelectAll}
                                    className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                                />
                            </th>
                            <th
                                className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                                onClick={() => handleSort("timestamp")}
                            >
                                <div className="flex items-center gap-1">
                                    Timestamp
                                    {getSortIcon("timestamp")}
                                </div>
                            </th>
                            <th
                                className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                                onClick={() => handleSort("kernel_name")}
                            >
                                <div className="flex items-center gap-1">
                                    Kernel
                                    {getSortIcon("kernel_name")}
                                </div>
                            </th>
                            <th
                                className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                                onClick={() => handleSort("hardware")}
                            >
                                <div className="flex items-center gap-1">
                                    Hardware
                                    {getSortIcon("hardware")}
                                </div>
                            </th>
                            <th
                                className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                                onClick={() => handleSort("score")}
                            >
                                <div className="flex items-center gap-1">
                                    Score
                                    {getSortIcon("score")}
                                </div>
                            </th>
                            <th
                                className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                                onClick={() => handleSort("flops_total")}
                            >
                                <div className="flex items-center gap-1">
                                    FLOPs
                                    {getSortIcon("flops_total")}
                                </div>
                            </th>
                            <th
                                className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                                onClick={() => handleSort("runtime_ms")}
                            >
                                <div className="flex items-center gap-1">
                                    Runtime
                                    {getSortIcon("runtime_ms")}
                                </div>
                            </th>
                            <th
                                className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                                onClick={() => handleSort("correctness_status")}
                            >
                                <div className="flex items-center gap-1">
                                    Correctness
                                    {getSortIcon("correctness_status")}
                                </div>
                            </th>
                            <th
                                className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                                onClick={() => handleSort("suggestions_count")}
                            >
                                <div className="flex items-center gap-1">
                                    Suggestions
                                    {getSortIcon("suggestions_count")}
                                </div>
                            </th>
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                        {filteredAndSortedData.map((item) => (
                            <tr
                                key={item.id}
                                className={`hover:bg-gray-50 ${
                                    selectedRows.has(item.id) ? "bg-blue-50" : ""
                                }`}
                            >
                                <td className="px-4 py-3">
                                    <input
                                        type="checkbox"
                                        checked={selectedRows.has(item.id)}
                                        onChange={() => handleSelectRow(item.id)}
                                        className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                                    />
                                </td>
                                <td className="px-4 py-3 text-sm text-gray-900">
                                    {formatValue(item.timestamp, "timestamp")}
                                </td>
                                <td className="px-4 py-3 text-sm font-medium text-gray-900">
                                    {item.kernel_name}
                                </td>
                                <td className="px-4 py-3 text-sm text-gray-900">{item.hardware}</td>
                                <td className="px-4 py-3 text-sm">
                                    <span className={`font-semibold ${getScoreColor(item.score)}`}>
                                        {item.score}
                                    </span>
                                </td>
                                <td className="px-4 py-3 text-sm text-gray-900">
                                    {formatValue(item.flops_total, "flops_total")}
                                </td>
                                <td className="px-4 py-3 text-sm text-gray-900">
                                    {formatValue(item.runtime_ms, "runtime_ms")}ms
                                </td>
                                <td className="px-4 py-3 text-sm">
                                    <span
                                        className={`font-medium ${getCorrectnessColor(
                                            item.correctness_status
                                        )}`}
                                    >
                                        {item.correctness_status.replace("_", " ")}
                                    </span>
                                </td>
                                <td className="px-4 py-3 text-sm text-gray-900">
                                    {item.suggestions_count}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Footer */}
            <div className="px-4 py-3 bg-gray-50 border-t border-gray-200">
                <div className="flex items-center justify-between text-sm text-gray-700">
                    <div>
                        Showing {filteredAndSortedData.length} of {data.length} results
                        {selectedRows.size > 0 && ` (${selectedRows.size} selected)`}
                    </div>
                    <div>
                        {searchTerm && (
                            <button
                                onClick={() => setSearchTerm("")}
                                className="text-primary-600 hover:text-primary-800"
                            >
                                Clear search
                            </button>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

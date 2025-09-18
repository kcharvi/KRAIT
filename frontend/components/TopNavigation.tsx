"use client";

import { useState } from "react";
import { ChevronDown, BarChart3, Cpu, Settings } from "lucide-react";

interface TopNavigationProps {
    activeTab: string;
    onTabChange: (tab: string) => void;
}

export default function TopNavigation({ activeTab, onTabChange }: TopNavigationProps) {
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);

    const tabs = [
        {
            id: "kernel-workbench",
            name: "Kernel Workbench",
            icon: Cpu,
            description: "Generate and analyze kernel code",
        },
        {
            id: "visualizations",
            name: "Visualizations",
            icon: BarChart3,
            description: "Performance charts and graphs",
        },
        {
            id: "metrics",
            name: "Metrics",
            icon: Settings,
            description: "Detailed metrics and analytics",
        },
    ];

    const activeTabData = tabs.find((tab) => tab.id === activeTab);

    return (
        <nav className="bg-white border-b border-gray-200 shadow-sm">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-16">
                    {/* Logo/Brand */}
                    <div className="flex items-center">
                        <div className="flex-shrink-0 flex items-center">
                            <Cpu className="h-8 w-8 text-blue-600" />
                            <span className="ml-2 text-xl font-bold text-gray-900">KRAIT</span>
                        </div>
                    </div>

                    {/* Navigation Tabs */}
                    <div className="flex items-center space-x-1">
                        {tabs.map((tab) => {
                            const Icon = tab.icon;
                            return (
                                <button
                                    key={tab.id}
                                    onClick={() => onTabChange(tab.id)}
                                    className={`flex items-center px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                                        activeTab === tab.id
                                            ? "bg-blue-100 text-blue-700 border-b-2 border-blue-500"
                                            : "text-gray-600 hover:text-gray-900 hover:bg-gray-100"
                                    }`}
                                >
                                    <Icon className="h-4 w-4 mr-2" />
                                    {tab.name}
                                </button>
                            );
                        })}
                    </div>

                    {/* Mobile Dropdown */}
                    <div className="md:hidden flex items-center">
                        <div className="relative">
                            <button
                                onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                                className="flex items-center px-3 py-2 rounded-md text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-100"
                            >
                                {activeTabData && (
                                    <>
                                        <activeTabData.icon className="h-4 w-4 mr-2" />
                                        {activeTabData.name}
                                    </>
                                )}
                                <ChevronDown className="h-4 w-4 ml-1" />
                            </button>

                            {isDropdownOpen && (
                                <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg z-50 border border-gray-200">
                                    <div className="py-1">
                                        {tabs.map((tab) => {
                                            const Icon = tab.icon;
                                            return (
                                                <button
                                                    key={tab.id}
                                                    onClick={() => {
                                                        onTabChange(tab.id);
                                                        setIsDropdownOpen(false);
                                                    }}
                                                    className={`flex items-center w-full px-4 py-2 text-sm ${
                                                        activeTab === tab.id
                                                            ? "bg-blue-50 text-blue-700"
                                                            : "text-gray-700 hover:bg-gray-100"
                                                    }`}
                                                >
                                                    <Icon className="h-4 w-4 mr-3" />
                                                    <div className="text-left">
                                                        <div className="font-medium">
                                                            {tab.name}
                                                        </div>
                                                        <div className="text-xs text-gray-500">
                                                            {tab.description}
                                                        </div>
                                                    </div>
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </nav>
    );
}

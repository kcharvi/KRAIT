"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { BarChart3, Activity, Cpu, Menu, X, ChevronLeft, ChevronRight } from "lucide-react";
import clsx from "clsx";

const navigation = [
    { name: "Kernel Workbench", href: "/", icon: Cpu },
    { name: "Visualizations", href: "/visualizations", icon: BarChart3 },
    { name: "Metrics", href: "/metrics", icon: Activity },
];

interface SidebarProps {
    collapsed: boolean;
    onToggle: () => void;
}

export default function Sidebar({ collapsed, onToggle }: SidebarProps) {
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const pathname = usePathname();

    const desktopWidth = collapsed ? "w-16" : "w-64";

    return (
        <>
            {/* Desktop fixed sidebar (full-height) */}
            <aside
                className={clsx(
                    "hidden lg:flex fixed inset-y-0 left-0 z-40 bg-white border-r shadow-sm transition-[width] duration-300 ease-in-out",
                    desktopWidth
                )}
            >
                <div className="flex h-full w-full flex-col">
                    <div className="flex items-center justify-between h-16 px-4 border-b border-gray-200">
                        <h1
                            className={clsx(
                                "text-xl font-bold text-gray-900 transition-opacity",
                                collapsed && "opacity-0 pointer-events-none"
                            )}
                        >
                            KRAIT
                        </h1>
                        <button
                            className="p-1 rounded hover:bg-gray-100"
                            onClick={onToggle}
                            aria-label="Toggle sidebar"
                        >
                            {collapsed ? (
                                <ChevronRight className="h-5 w-5" />
                            ) : (
                                <ChevronLeft className="h-5 w-5" />
                            )}
                        </button>
                    </div>

                    <nav className="mt-4 flex-1 px-2 overflow-y-auto">
                        <ul className="space-y-1">
                            {navigation.map((item) => {
                                const Icon = item.icon;
                                const isActive = pathname === item.href;
                                return (
                                    <li key={item.name}>
                                        <Link
                                            href={item.href}
                                            className={clsx(
                                                "flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors",
                                                isActive
                                                    ? "bg-primary-50 text-primary-700"
                                                    : "text-gray-700 hover:bg-gray-50"
                                            )}
                                        >
                                            <Icon className="h-5 w-5 flex-shrink-0" />
                                            <span
                                                className={clsx(
                                                    "truncate transition-all",
                                                    collapsed && "opacity-0 pointer-events-none w-0"
                                                )}
                                            >
                                                {item.name}
                                            </span>
                                        </Link>
                                    </li>
                                );
                            })}
                        </ul>
                    </nav>
                </div>
            </aside>

            {/* Mobile overlay + drawer */}
            {sidebarOpen && (
                <div
                    className="fixed inset-0 z-40 bg-gray-600/50 lg:hidden"
                    onClick={() => setSidebarOpen(false)}
                />
            )}
            <aside
                className={clsx(
                    "lg:hidden fixed inset-y-0 left-0 z-50 w-64 bg-white border-r shadow-lg transform transition-transform duration-300 ease-in-out",
                    sidebarOpen ? "translate-x-0" : "-translate-x-full"
                )}
            >
                <div className="flex items-center justify-between h-16 px-4 border-b border-gray-200">
                    <h1 className="text-xl font-bold text-gray-900">KRAIT</h1>
                    <button onClick={() => setSidebarOpen(false)}>
                        <X className="h-6 w-6" />
                    </button>
                </div>
                <nav className="mt-4 px-4">
                    <ul className="space-y-2">
                        {navigation.map((item) => {
                            const Icon = item.icon;
                            const isActive = pathname === item.href;
                            return (
                                <li key={item.name}>
                                    <Link
                                        href={item.href}
                                        className={clsx(
                                            "flex items-center px-3 py-2 rounded-md text-sm",
                                            isActive
                                                ? "bg-primary-50 text-primary-700"
                                                : "text-gray-700 hover:bg-gray-50"
                                        )}
                                        onClick={() => setSidebarOpen(false)}
                                    >
                                        <Icon className="mr-3 h-5 w-5" />
                                        {item.name}
                                    </Link>
                                </li>
                            );
                        })}
                    </ul>
                </nav>
            </aside>

            {/* Mobile menu button */}
            <button
                className="fixed top-4 left-4 z-50 lg:hidden bg-white p-2 rounded-md shadow-md"
                onClick={() => setSidebarOpen(true)}
                aria-label="Open sidebar"
            >
                <Menu className="h-6 w-6" />
            </button>
        </>
    );
}

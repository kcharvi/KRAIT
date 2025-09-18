"use client";

import { useState } from "react";
import Sidebar from "./Sidebar";

export default function AppShell({ children }: { children: React.ReactNode }) {
    const [collapsed, setCollapsed] = useState(false);

    return (
        <div className="h-screen w-screen overflow-hidden bg-gray-50">
            <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((v) => !v)} />
            <main
                className={`h-full overflow-auto transition-all duration-300 ${
                    collapsed ? "lg:ml-16" : "lg:ml-64"
                }`}
            >
                {children}
            </main>
        </div>
    );
}

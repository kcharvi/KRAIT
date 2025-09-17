import Layout from "@/components/Layout";
import { BarChart3 } from "lucide-react";

export default function VisualizationsPage() {
    return (
        <Layout>
            <div className="p-8">
                <div className="flex items-center mb-8">
                    <BarChart3 className="h-8 w-8 text-primary-500 mr-3" />
                    <h1 className="text-3xl font-bold text-gray-900">Visualizations</h1>
                </div>

                <div className="bg-white rounded-lg shadow p-8 text-center">
                    <BarChart3 className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                    <h2 className="text-xl font-semibold text-gray-600 mb-2">Coming Soon</h2>
                    <p className="text-gray-500">Visualization features will be available here.</p>
                </div>
            </div>
        </Layout>
    );
}

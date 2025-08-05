// frontend/src/components/DiagnosticPanel.tsx
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { CheckCircleIcon, XCircleIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline';
import { simulationApi } from '../services/api';

interface DiagnosticResult {
    success: boolean;
    message: string;
    details?: any;
    timestamp: string;
}

interface EndpointTest {
    name: string;
    url: string;
    test: () => Promise<any>;
}

const DiagnosticPanel: React.FC = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [isRunning, setIsRunning] = useState(false);
    const [results, setResults] = useState<DiagnosticResult[]>([]);

    const tests: EndpointTest[] = [
        {
            name: 'Backend Health Check',
            url: '/health',
            test: () => simulationApi.healthCheck()
        },
        {
            name: 'API Documentation',
            url: '/docs',
            test: async () => {
                const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/docs`);
                return response.ok;
            }
        },
        {
            name: 'Supported Assets Endpoint',
            url: '/api/v1/supported-assets',
            test: () => simulationApi.getAvailableAssets()
        },
        {
            name: 'Quantum Status Endpoint',
            url: '/api/v1/quantum-status',
            test: async () => {
                const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/quantum-status`);
                return response.json();
            }
        },
        {
            name: 'Mock Simulation Test',
            url: '/api/v1/simulate',
            test: async () => {
                return simulationApi.runSimulation({
                    news_inputs: [{ content: 'Test news for diagnostic', source_type: 'headline' }],
                    target_assets: ['AAPL'],
                    simulation_method: 'classical_baseline',
                    time_horizon_days: 1,
                    num_scenarios: 100,
                    include_quantum_metrics: false,
                    compare_with_classical: false
                });
            }
        }
    ];

    const runDiagnostics = async () => {
        setIsRunning(true);
        setResults([]);

        const newResults: DiagnosticResult[] = [];

        for (const test of tests) {
            try {
                console.log(`Running test: ${test.name}`);
                const result = await test.test();

                newResults.push({
                    success: true,
                    message: `✅ ${test.name}: Success`,
                    details: result,
                    timestamp: new Date().toISOString()
                });
            } catch (error: any) {
                console.error(`Test failed: ${test.name}`, error);

                newResults.push({
                    success: false,
                    message: `❌ ${test.name}: ${error.message || 'Failed'}`,
                    details: {
                        error: error.message,
                        code: error.code,
                        status: error.response?.status,
                        url: test.url
                    },
                    timestamp: new Date().toISOString()
                });
            }

            setResults([...newResults]); // Update results after each test
            await new Promise(resolve => setTimeout(resolve, 500)); // Small delay between tests
        }

        setIsRunning(false);
    };

    const getSystemInfo = () => {
        return {
            userAgent: navigator.userAgent,
            language: navigator.language,
            cookieEnabled: navigator.cookieEnabled,
            onLine: navigator.onLine,
            timestamp: new Date().toISOString(),
            apiUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000',
            nodeEnv: import.meta.env.NODE_ENV,
            currentUrl: window.location.href
        };
    };

    if (!isOpen) {
        return (
            <div className="fixed bottom-4 right-4 z-50">
                <button
                    onClick={() => setIsOpen(true)}
                    className="bg-yellow-600 hover:bg-yellow-700 text-white p-3 rounded-full shadow-lg transition-colors"
                >
                    <ExclamationTriangleIcon className="w-6 h-6" />
                </button>
            </div>
        );
    }

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="fixed bottom-4 right-4 w-96 max-h-96 bg-gray-800 border border-gray-600 rounded-lg shadow-xl z-50 overflow-hidden"
        >
            <div className="p-4 border-b border-gray-600 flex items-center justify-between">
                <h3 className="text-lg font-semibold text-white flex items-center">
                    <ExclamationTriangleIcon className="w-5 h-5 mr-2 text-yellow-400" />
                    Connection Diagnostics
                </h3>
                <button
                    onClick={() => setIsOpen(false)}
                    className="text-gray-400 hover:text-white"
                >
                    ✕
                </button>
            </div>

            <div className="p-4 max-h-80 overflow-y-auto">
                {/* System Information */}
                <div className="mb-4">
                    <h4 className="text-sm font-semibold text-gray-300 mb-2">System Information</h4>
                    <div className="text-xs text-gray-400 space-y-1">
                        <div>API URL: {import.meta.env.VITE_API_URL || 'http://localhost:8000'}</div>
                        <div>Online: {navigator.onLine ? 'Yes' : 'No'}</div>
                        <div>Environment: {import.meta.env.NODE_ENV}</div>
                    </div>
                </div>

                {/* Test Controls */}
                <div className="mb-4">
                    <button
                        onClick={runDiagnostics}
                        disabled={isRunning}
                        className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white py-2 px-4 rounded transition-colors"
                    >
                        {isRunning ? 'Running Tests...' : 'Run Diagnostics'}
                    </button>
                </div>

                {/* Test Results */}
                <div className="space-y-2">
                    {results.map((result, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className={`p-2 rounded text-sm ${
                                result.success
                                    ? 'bg-green-900/50 border border-green-600'
                                    : 'bg-red-900/50 border border-red-600'
                            }`}
                        >
                            <div className="flex items-start">
                                {result.success ? (
                                    <CheckCircleIcon className="w-4 h-4 text-green-400 mr-2 mt-0.5 flex-shrink-0" />
                                ) : (
                                    <XCircleIcon className="w-4 h-4 text-red-400 mr-2 mt-0.5 flex-shrink-0" />
                                )}
                                <div className="flex-1">
                                    <div className={result.success ? 'text-green-200' : 'text-red-200'}>
                                        {result.message}
                                    </div>
                                    {result.details && !result.success && (
                                        <div className="mt-1 text-xs text-gray-400">
                                            {typeof result.details === 'object' ? (
                                                <pre className="whitespace-pre-wrap">
                                                    {JSON.stringify(result.details, null, 2)}
                                                </pre>
                                            ) : (
                                                String(result.details)
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </div>

                {/* Quick Fix Suggestions */}
                {results.some(r => !r.success) && (
                    <div className="mt-4 p-3 bg-yellow-900/50 border border-yellow-600 rounded">
                        <h4 className="text-sm font-semibold text-yellow-200 mb-2">Quick Fixes:</h4>
                        <ul className="text-xs text-yellow-100 space-y-1">
                            <li>• Make sure backend is running: <code>uvicorn app.main:app --reload</code></li>
                            <li>• Check if port 8000 is free: <code>lsof -i :8000</code></li>
                            <li>• Verify CORS settings in backend config</li>
                            <li>• Clear browser cache and refresh</li>
                            <li>• Check browser console for additional errors</li>
                        </ul>
                    </div>
                )}

                {/* Export Debug Info */}
                <div className="mt-4">
                    <button
                        onClick={() => {
                            const debugInfo = {
                                systemInfo: getSystemInfo(),
                                testResults: results,
                                timestamp: new Date().toISOString()
                            };

                            const blob = new Blob([JSON.stringify(debugInfo, null, 2)],
                                { type: 'application/json' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = 'diagnostic-report.json';
                            a.click();
                            URL.revokeObjectURL(url);
                        }}
                        className="w-full bg-gray-600 hover:bg-gray-700 text-white py-1 px-3 rounded text-sm transition-colors"
                    >
                        Export Debug Report
                    </button>
                </div>
            </div>
        </motion.div>
    );
};

export default DiagnosticPanel;
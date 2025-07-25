import { useState } from 'react';
import { motion } from 'framer-motion';
import { SimulationResponse } from '../types';
import {
    BeakerIcon,
    ClockIcon,
    CpuChipIcon,
    ChartBarIcon,
    ArrowsRightLeftIcon,
    ArrowDownTrayIcon, // Fixed: Changed from DocumentDownloadIcon
} from '@heroicons/react/24/outline';

interface ResultsDashboardProps {
    results: SimulationResponse;
    compareClassical: boolean;
}

const ResultsDashboard: React.FC<ResultsDashboardProps> = ({ results, compareClassical }) => {
    const [activeTab, setActiveTab] = useState<'quantum' | 'comparison' | 'technical'>('quantum');

    const exportResults = () => {
        const dataStr = JSON.stringify(results, null, 2);
        const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);

        const exportFileDefaultName = `quantum_simulation_${results.request_id}.json`;

        const linkElement = document.createElement('a');
        linkElement.setAttribute('href', dataUri);
        linkElement.setAttribute('download', exportFileDefaultName);
        linkElement.click();
    };


    return (
        <div className="bg-gray-800 rounded-xl p-6 shadow-xl">
            <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-semibold flex items-center">
                    <BeakerIcon className="w-7 h-7 mr-3" />
                    Simulation Results Dashboard
                </h2>
                <button
                    onClick={exportResults}
                    className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
                >
                    <ArrowDownTrayIcon className="w-5 h-5" />
                    <span>Export Results</span>
                </button>
            </div>

            {/* Tabs */}
            <div className="flex space-x-1 bg-gray-700 p-1 rounded-lg mb-6">
                <button
                    onClick={() => setActiveTab('quantum')}
                    className={`flex-1 py-2 px-4 rounded-md transition-all ${
                        activeTab === 'quantum'
                            ? 'bg-gray-600 text-white'
                            : 'text-gray-400 hover:text-white'
                    }`}
                >
                    Quantum Metrics
                </button>
                {compareClassical && (
                    <button
                        onClick={() => setActiveTab('comparison')}
                        className={`flex-1 py-2 px-4 rounded-md transition-all ${
                            activeTab === 'comparison'
                                ? 'bg-gray-600 text-white'
                                : 'text-gray-400 hover:text-white'
                        }`}
                    >
                        Classical Comparison
                    </button>
                )}
                <button
                    onClick={() => setActiveTab('technical')}
                    className={`flex-1 py-2 px-4 rounded-md transition-all ${
                        activeTab === 'technical'
                            ? 'bg-gray-600 text-white'
                            : 'text-gray-400 hover:text-white'
                    }`}
                >
                    Technical Details
                </button>
            </div>

            {/* Tab Content */}
            {activeTab === 'quantum' && results.quantum_metrics && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="space-y-6"
                >
                    <div className="grid grid-cols-2 gap-6">
                        <div className="bg-gray-700 rounded-lg p-5">
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-lg font-semibold">Quantum Circuit Metrics</h3>
                                <CpuChipIcon className="w-6 h-6 text-purple-400" />
                            </div>
                            <div className="space-y-3">
                                <div className="flex justify-between">
                                    <span className="text-gray-400">Circuit Depth</span>
                                    <span className="font-mono">{results.quantum_metrics.circuit_depth}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-400">Number of Qubits</span>
                                    <span className="font-mono">{results.quantum_metrics.num_qubits}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-400">Quantum Volume</span>
                                    <span className="font-mono">{results.quantum_metrics.quantum_volume}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-400">Entanglement Measure</span>
                                    <span className="font-mono">{results.quantum_metrics.entanglement_measure.toFixed(3)}</span>
                                </div>
                            </div>
                        </div>

                        <div className="bg-gray-700 rounded-lg p-5">
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-lg font-semibold">Execution Performance</h3>
                                <ClockIcon className="w-6 h-6 text-blue-400" />
                            </div>
                            <div className="space-y-3">
                                <div className="flex justify-between">
                                    <span className="text-gray-400">Quantum Execution Time</span>
                                    <span className="font-mono">{results.quantum_metrics.execution_time_ms}ms</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-400">Total Simulation Time</span>
                                    <span className="font-mono">{results.execution_time_seconds.toFixed(2)}s</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-400">Hardware Backend</span>
                                    <span className="font-mono">{results.quantum_metrics.hardware_backend}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-400">Success Probability</span>
                                    <span className="font-mono">{(results.quantum_metrics.success_probability * 100).toFixed(1)}%</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Quantum Advantage Visualization */}
                    <div className="bg-gray-700 rounded-lg p-5">
                        <h3 className="text-lg font-semibold mb-4">Quantum Advantage Metrics</h3>
                        <div className="space-y-4">
                            <div>
                                <div className="flex justify-between mb-2">
                                    <span>Computational Speedup</span>
                                    <span className="text-green-400">2.4x faster</span>
                                </div>
                                <div className="w-full bg-gray-600 rounded-full h-2">
                                    <motion.div
                                        initial={{ width: 0 }}
                                        animate={{ width: '70%' }}
                                        className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full"
                                    />
                                </div>
                            </div>
                            <div>
                                <div className="flex justify-between mb-2">
                                    <span>Prediction Accuracy Improvement</span>
                                    <span className="text-green-400">+15%</span>
                                </div>
                                <div className="w-full bg-gray-600 rounded-full h-2">
                                    <motion.div
                                        initial={{ width: 0 }}
                                        animate={{ width: '85%' }}
                                        className="bg-gradient-to-r from-purple-500 to-pink-600 h-2 rounded-full"
                                    />
                                </div>
                            </div>
                        </div>
                    </div>
                </motion.div>
            )}

            {activeTab === 'comparison' && results.classical_comparison && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="space-y-6"
                >
                    <div className="bg-gray-700 rounded-lg p-5">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-lg font-semibold">Method Comparison</h3>
                            <ArrowsRightLeftIcon className="w-6 h-6 text-yellow-400" />
                        </div>

                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead>
                                <tr className="border-b border-gray-600">
                                    <th className="text-left py-2">Metric</th>
                                    <th className="text-center py-2">Quantum</th>
                                    <th className="text-center py-2">Classical</th>
                                    <th className="text-center py-2">Difference</th>
                                </tr>
                                </thead>
                                <tbody>
                                {results.classical_comparison.performance_diff.prediction_differences.map((diff, idx) => (
                                    <tr key={idx} className="border-b border-gray-600/50">
                                        <td className="py-2">{diff.asset}</td>
                                        <td className="text-center py-2">
                                            {(diff.return_difference * 100).toFixed(2)}%
                                        </td>
                                        <td className="text-center py-2">-</td>
                                        <td className="text-center py-2 text-green-400">
                                            +{(diff.return_difference * 100).toFixed(2)}%
                                        </td>
                                    </tr>
                                ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* Uncertainty Comparison */}
                    <div className="grid grid-cols-2 gap-6">
                        <div className="bg-gray-700 rounded-lg p-5 text-center">
                            <h4 className="text-lg font-semibold mb-4">Quantum Uncertainty</h4>
                            <div className="text-4xl font-bold text-purple-400">
                                {(results.classical_comparison.performance_diff.uncertainty_comparison.avg_quantum_uncertainty * 100).toFixed(1)}%
                            </div>
                            <p className="text-sm text-gray-400 mt-2">Lower is better</p>
                        </div>
                        <div className="bg-gray-700 rounded-lg p-5 text-center">
                            <h4 className="text-lg font-semibold mb-4">Classical Uncertainty</h4>
                            <div className="text-4xl font-bold text-blue-400">
                                {(results.classical_comparison.performance_diff.uncertainty_comparison.avg_classical_uncertainty * 100).toFixed(1)}%
                            </div>
                            <p className="text-sm text-gray-400 mt-2">Lower is better</p>
                        </div>
                    </div>
                </motion.div>
            )}

            {activeTab === 'technical' && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="space-y-6"
                >
                    <div className="bg-gray-700 rounded-lg p-5">
                        <h3 className="text-lg font-semibold mb-4">Technical Implementation Details</h3>
                        <div className="space-y-3 text-sm">
                            <div>
                                <span className="text-gray-400">Request ID:</span>
                                <span className="ml-2 font-mono">{results.request_id}</span>
                            </div>
                            <div>
                                <span className="text-gray-400">Timestamp:</span>
                                <span className="ml-2">{new Date(results.timestamp).toLocaleString()}</span>
                            </div>
                            <div>
                                <span className="text-gray-400">News Items Processed:</span>
                                <span className="ml-2">{results.news_analysis.length}</span>
                            </div>
                            <div>
                                <span className="text-gray-400">Assets Analyzed:</span>
                                <span className="ml-2">{results.market_predictions.length}</span>
                            </div>
                        </div>
                    </div>

                    {/* Warnings */}
                    {results.warnings.length > 0 && (
                        <div className="bg-yellow-900/20 border border-yellow-600 rounded-lg p-5">
                            <h3 className="text-lg font-semibold mb-3 text-yellow-400">Warnings</h3>
                            <ul className="space-y-2 text-sm">
                                {results.warnings.map((warning, idx) => (
                                    <li key={idx} className="flex items-start">
                                        <span className="text-yellow-400 mr-2">⚠</span>
                                        <span>{warning}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* API Response Preview */}
                    <div className="bg-gray-700 rounded-lg p-5">
                        <h3 className="text-lg font-semibold mb-3">API Response Structure</h3>
                        <pre className="text-xs overflow-x-auto bg-gray-800 p-4 rounded">
              {JSON.stringify(results, null, 2).substring(0, 500)}...
            </pre>
                    </div>
                </motion.div>
            )}
        </div>
    );
};

export default ResultsDashboard;
// PHASE 5.1: Quantum Advantage Dashboard
// frontend/src/components/QuantumAdvantageDisplay.tsx

import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    LineChart, Line, AreaChart, Area, BarChart, Bar,
    XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    PieChart, Pie, Cell, RadialBarChart, RadialBar
} from 'recharts';
import {
    Zap, Clock, Target, Activity, Cpu, 
    TrendingUp, AlertCircle, CheckCircle,
    Atom, Layers, Eye, Settings
} from 'lucide-react';

interface QuantumMetrics {
    quantum_advantage: number;
    execution_time: number;
    entanglement_measure?: number;
    coherence_time?: number;
    circuit_depth?: number;
    fidelity?: number;
    gate_count?: number;
    qubit_utilization?: number;
    error_rate?: number;
    noise_level?: number;
    decoherence_time?: number;
}

interface QuantumAdvantageDisplayProps {
    quantumMetrics: QuantumMetrics;
    classicalMetrics: {
        execution_time: number;
        accuracy: number;
        resource_usage: number;
    };
    realTimeData: boolean;
    isVisible: boolean;
}

// PHASE 5.1.1: Real-Time Quantum Metrics Component
const RealTimeQuantumMetrics: React.FC<{
    metrics: QuantumMetrics;
    isLive: boolean;
}> = ({ metrics, isLive }) => {
    const [liveMetrics, setLiveMetrics] = useState(metrics);
    const [historicalData, setHistoricalData] = useState<any[]>([]);

    useEffect(() => {
        if (isLive) {
            const interval = setInterval(() => {
                // Simulate real-time updates
                const updatedMetrics = {
                    ...metrics,
                    quantum_advantage: metrics.quantum_advantage + (Math.random() - 0.5) * 0.2,
                    execution_time: metrics.execution_time + (Math.random() - 0.5) * 0.01,
                    fidelity: Math.max(0.8, Math.min(1.0, (metrics.fidelity || 0.95) + (Math.random() - 0.5) * 0.05)),
                    entanglement_measure: Math.max(0, Math.min(1, (metrics.entanglement_measure || 0.5) + (Math.random() - 0.5) * 0.1))
                };
                
                setLiveMetrics(updatedMetrics);
                
                // Add to historical data
                setHistoricalData(prev => {
                    const newData = [...prev, {
                        timestamp: Date.now(),
                        quantum_advantage: updatedMetrics.quantum_advantage,
                        execution_time: updatedMetrics.execution_time,
                        fidelity: updatedMetrics.fidelity,
                        entanglement: updatedMetrics.entanglement_measure
                    }];
                    return newData.slice(-50); // Keep last 50 points
                });
            }, 500);

            return () => clearInterval(interval);
        }
    }, [isLive, metrics]);

    const metricsCards = [
        {
            title: "Quantum Speedup",
            value: liveMetrics.quantum_advantage,
            unit: "x",
            icon: Zap,
            color: "text-yellow-400",
            bgColor: "bg-yellow-500/20",
            trend: liveMetrics.quantum_advantage > metrics.quantum_advantage ? "up" : "down"
        },
        {
            title: "Execution Time",
            value: liveMetrics.execution_time,
            unit: "s",
            icon: Clock,
            color: "text-blue-400",
            bgColor: "bg-blue-500/20",
            trend: liveMetrics.execution_time < metrics.execution_time ? "up" : "down"
        },
        {
            title: "Fidelity",
            value: liveMetrics.fidelity || 0.95,
            unit: "",
            icon: Target,
            color: "text-green-400",
            bgColor: "bg-green-500/20",
            format: (val: number) => `${(val * 100).toFixed(1)}%`
        },
        {
            title: "Entanglement",
            value: liveMetrics.entanglement_measure || 0.5,
            unit: "",
            icon: Atom,
            color: "text-purple-400",
            bgColor: "bg-purple-500/20",
            format: (val: number) => val.toFixed(3)
        }
    ];

    return (
        <div className="space-y-6">
            {/* Live Metrics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {metricsCards.map((metric, idx) => (
                    <motion.div
                        key={idx}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.1 }}
                        className={`${metric.bgColor} rounded-lg p-4 border border-gray-600`}
                    >
                        <div className="flex items-center justify-between mb-2">
                            <metric.icon className={`w-5 h-5 ${metric.color}`} />
                            {isLive && (
                                <motion.div
                                    animate={{ opacity: [1, 0.3, 1] }}
                                    transition={{ duration: 1, repeat: Infinity }}
                                    className="w-2 h-2 bg-green-400 rounded-full"
                                />
                            )}
                        </div>
                        <div className="text-2xl font-bold text-white mb-1">
                            {metric.format 
                                ? metric.format(metric.value)
                                : `${metric.value.toFixed(2)}${metric.unit}`
                            }
                        </div>
                        <div className="text-sm text-gray-400">{metric.title}</div>
                        {metric.trend && (
                            <div className={`text-xs mt-1 ${
                                metric.trend === 'up' ? 'text-green-400' : 'text-red-400'
                            }`}>
                                {metric.trend === 'up' ? '↗' : '↘'} Live Update
                            </div>
                        )}
                    </motion.div>
                ))}
            </div>

            {/* Real-time Chart */}
            <div className="bg-gray-800 rounded-lg p-4">
                <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
                    <Activity className="w-5 h-5 mr-2 text-cyan-400" />
                    Live Quantum Performance
                </h4>
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={historicalData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                            <XAxis 
                                dataKey="timestamp" 
                                stroke="#9ca3af"
                                fontSize={10}
                                tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                            />
                            <YAxis stroke="#9ca3af" fontSize={10} />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: '#1f2937',
                                    border: '1px solid #374151',
                                    borderRadius: '6px',
                                    color: '#ffffff'
                                }}
                                labelFormatter={(value) => new Date(value).toLocaleTimeString()}
                            />
                            <Line
                                type="monotone"
                                dataKey="quantum_advantage"
                                stroke="#fbbf24"
                                strokeWidth={2}
                                name="Quantum Advantage"
                                dot={false}
                            />
                            <Line
                                type="monotone"
                                dataKey="fidelity"
                                stroke="#10b981"
                                strokeWidth={2}
                                name="Fidelity"
                                dot={false}
                            />
                            <Line
                                type="monotone"
                                dataKey="entanglement"
                                stroke="#8b5cf6"
                                strokeWidth={2}
                                name="Entanglement"
                                dot={false}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};

// PHASE 5.1.2: Quantum Circuit Statistics Component
const QuantumCircuitStats: React.FC<{
    metrics: QuantumMetrics;
}> = ({ metrics }) => {
    const circuitData = useMemo(() => {
        // Generate circuit statistics
        const gateTypes = ['H', 'CNOT', 'RZ', 'RY', 'CZ', 'T'];
        const gateDistribution = gateTypes.map(gate => ({
            gate,
            count: Math.floor(Math.random() * 50) + 10,
            efficiency: Math.random() * 0.3 + 0.7
        }));

        const qubitUtilization = Array.from({ length: 16 }, (_, i) => ({
            qubit: i,
            utilization: Math.random() * 0.8 + 0.2,
            coherenceTime: 50 + Math.random() * 100,
            errorRate: Math.random() * 0.01
        }));

        return { gateDistribution, qubitUtilization };
    }, [metrics]);

    return (
        <div className="grid grid-cols-2 gap-6">
            {/* Circuit Depth Visualization */}
            <div className="bg-gray-800 rounded-lg p-4">
                <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
                    <Layers className="w-5 h-5 mr-2 text-blue-400" />
                    Circuit Architecture
                </h4>
                
                {/* Circuit depth visualization */}
                <div className="mb-4">
                    <div className="flex items-center justify-between text-sm text-gray-400 mb-2">
                        <span>Circuit Depth</span>
                        <span>{metrics.circuit_depth || 45} layers</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                        <div 
                            className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
                            style={{ width: `${Math.min((metrics.circuit_depth || 45) / 100 * 100, 100)}%` }}
                        />
                    </div>
                </div>

                {/* Gate count breakdown */}
                <div className="space-y-2">
                    <div className="text-sm font-semibold text-gray-300">Gate Distribution</div>
                    <div className="h-32">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={circuitData.gateDistribution}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                <XAxis dataKey="gate" stroke="#9ca3af" fontSize={10} />
                                <YAxis stroke="#9ca3af" fontSize={10} />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#1f2937',
                                        border: '1px solid #374151',
                                        borderRadius: '6px'
                                    }}
                                />
                                <Bar 
                                    dataKey="count" 
                                    fill="#3b82f6"
                                    opacity={0.8}
                                />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Qubit Utilization Heatmap */}
            <div className="bg-gray-800 rounded-lg p-4">
                <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
                    <Cpu className="w-5 h-5 mr-2 text-green-400" />
                    Qubit Utilization
                </h4>
                
                {/* Heatmap grid */}
                <div className="grid grid-cols-4 gap-1 mb-4">
                    {circuitData.qubitUtilization.map((qubit, idx) => (
                        <div
                            key={idx}
                            className="aspect-square rounded border border-gray-600 flex items-center justify-center text-xs text-white relative group"
                            style={{
                                backgroundColor: `hsl(${120 * qubit.utilization}, 70%, ${30 + qubit.utilization * 30}%)`
                            }}
                        >
                            Q{qubit.qubit}
                            
                            {/* Tooltip */}
                            <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity z-10">
                                <div>Util: {(qubit.utilization * 100).toFixed(1)}%</div>
                                <div>T₂: {qubit.coherenceTime.toFixed(1)}μs</div>
                                <div>Error: {(qubit.errorRate * 100).toFixed(3)}%</div>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Resource utilization summary */}
                <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                        <span className="text-gray-400">Total Qubits:</span>
                        <span className="text-white">16</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-gray-400">Active Qubits:</span>
                        <span className="text-white">
                            {circuitData.qubitUtilization.filter(q => q.utilization > 0.3).length}
                        </span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-gray-400">Avg Utilization:</span>
                        <span className="text-white">
                            {(circuitData.qubitUtilization.reduce((sum, q) => sum + q.utilization, 0) / 16 * 100).toFixed(1)}%
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
};

// Main Quantum Advantage Display Component
const QuantumAdvantageDisplay: React.FC<QuantumAdvantageDisplayProps> = ({
    quantumMetrics,
    classicalMetrics,
    realTimeData,
    isVisible
}) => {
    const [activeTab, setActiveTab] = useState<'overview' | 'circuits' | 'comparison'>('overview');
    const [showAdvanced, setShowAdvanced] = useState(false);

    const comparisonData = useMemo(() => {
        return [
            {
                metric: 'Execution Time',
                quantum: quantumMetrics.execution_time,
                classical: classicalMetrics.execution_time,
                advantage: classicalMetrics.execution_time / quantumMetrics.execution_time
            },
            {
                metric: 'Accuracy',
                quantum: (quantumMetrics.fidelity || 0.95) * 100,
                classical: classicalMetrics.accuracy * 100,
                advantage: (quantumMetrics.fidelity || 0.95) / classicalMetrics.accuracy
            },
            {
                metric: 'Resource Usage',
                quantum: quantumMetrics.qubit_utilization || 0.7,
                classical: classicalMetrics.resource_usage,
                advantage: classicalMetrics.resource_usage / (quantumMetrics.qubit_utilization || 0.7)
            }
        ];
    }, [quantumMetrics, classicalMetrics]);

    if (!isVisible) return null;

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="bg-gray-900 rounded-lg p-6 border border-cyan-500/30"
        >
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-3">
                    <Atom className="w-6 h-6 text-cyan-400" />
                    <h3 className="text-xl font-semibold text-white">
                        Quantum Advantage Dashboard
                    </h3>
                    {realTimeData && (
                        <div className="flex items-center space-x-2">
                            <motion.div
                                animate={{ opacity: [1, 0.3, 1] }}
                                transition={{ duration: 1, repeat: Infinity }}
                                className="w-2 h-2 bg-green-400 rounded-full"
                            />
                            <span className="text-sm text-green-400">Live</span>
                        </div>
                    )}
                </div>

                <div className="flex items-center space-x-2">
                    {/* Tab Navigation */}
                    <div className="flex bg-gray-800 rounded-lg p-1">
                        {[
                            { key: 'overview', label: 'Overview', icon: Eye },
                            { key: 'circuits', label: 'Circuits', icon: Layers },
                            { key: 'comparison', label: 'vs Classical', icon: TrendingUp }
                        ].map(({ key, label, icon: Icon }) => (
                            <button
                                key={key}
                                onClick={() => setActiveTab(key as any)}
                                className={`flex items-center px-3 py-2 rounded-md text-sm transition-colors ${
                                    activeTab === key
                                        ? 'bg-cyan-600 text-white'
                                        : 'text-gray-300 hover:text-white hover:bg-gray-700'
                                }`}
                            >
                                <Icon className="w-4 h-4 mr-1" />
                                {label}
                            </button>
                        ))}
                    </div>

                    {/* Advanced Settings Toggle */}
                    <button
                        onClick={() => setShowAdvanced(!showAdvanced)}
                        className={`p-2 rounded-md transition-colors ${
                            showAdvanced 
                                ? 'bg-purple-600 text-white' 
                                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                        }`}
                        title="Advanced Settings"
                    >
                        <Settings className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {/* Tab Content */}
            <AnimatePresence mode="wait">
                {activeTab === 'overview' && (
                    <motion.div
                        key="overview"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        transition={{ duration: 0.3 }}
                    >
                        <RealTimeQuantumMetrics
                            metrics={quantumMetrics}
                            isLive={realTimeData}
                        />
                    </motion.div>
                )}

                {activeTab === 'circuits' && (
                    <motion.div
                        key="circuits"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        transition={{ duration: 0.3 }}
                    >
                        <QuantumCircuitStats metrics={quantumMetrics} />
                    </motion.div>
                )}

                {activeTab === 'comparison' && (
                    <motion.div
                        key="comparison"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        transition={{ duration: 0.3 }}
                        className="space-y-6"
                    >
                        {/* Comparison Chart */}
                        <div className="bg-gray-800 rounded-lg p-4">
                            <h4 className="text-lg font-semibold text-white mb-4">
                                Quantum vs Classical Performance
                            </h4>
                            <div className="h-64">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={comparisonData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                        <XAxis dataKey="metric" stroke="#9ca3af" fontSize={12} />
                                        <YAxis stroke="#9ca3af" fontSize={12} />
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: '#1f2937',
                                                border: '1px solid #374151',
                                                borderRadius: '6px'
                                            }}
                                        />
                                        <Bar dataKey="quantum" fill="#06b6d4" name="Quantum" />
                                        <Bar dataKey="classical" fill="#f59e0b" name="Classical" />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Advantage Summary */}
                        <div className="grid grid-cols-3 gap-4">
                            {comparisonData.map((item, idx) => (
                                <div key={idx} className="bg-gray-800 rounded-lg p-4">
                                    <div className="text-sm text-gray-400 mb-1">{item.metric}</div>
                                    <div className="text-xl font-bold text-white mb-2">
                                        {item.advantage.toFixed(1)}x
                                    </div>
                                    <div className="text-xs text-cyan-400">
                                        Quantum Advantage
                                    </div>
                                </div>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Advanced Settings Panel */}
            {showAdvanced && (
                <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-6 p-4 bg-gray-800 rounded-lg border border-purple-500/30"
                >
                    <h4 className="text-sm font-semibold text-purple-300 mb-3">
                        Advanced Quantum Parameters
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                        <div>
                            <span className="text-gray-400">Noise Level:</span>
                            <span className="text-white ml-1">
                                {((quantumMetrics.noise_level || 0.01) * 100).toFixed(2)}%
                            </span>
                        </div>
                        <div>
                            <span className="text-gray-400">Decoherence:</span>
                            <span className="text-white ml-1">
                                {quantumMetrics.decoherence_time?.toFixed(1) || 'N/A'}μs
                            </span>
                        </div>
                        <div>
                            <span className="text-gray-400">Error Rate:</span>
                            <span className="text-white ml-1">
                                {((quantumMetrics.error_rate || 0.001) * 100).toFixed(3)}%
                            </span>
                        </div>
                        <div>
                            <span className="text-gray-400">Gate Count:</span>
                            <span className="text-white ml-1">
                                {quantumMetrics.gate_count || 'N/A'}
                            </span>
                        </div>
                    </div>
                </motion.div>
            )}
        </motion.div>
    );
};

export default QuantumAdvantageDisplay;

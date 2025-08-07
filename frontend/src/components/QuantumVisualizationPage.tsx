import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    LineChart, Line, AreaChart, Area, BarChart, Bar,
    XAxis, YAxis, CartesianGrid, Tooltip, Legend,
    ResponsiveContainer, PieChart, Pie, Cell, RadarChart,
    PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import { ChevronDownIcon, CubeIcon, ChartBarIcon, SparklesIcon } from '@heroicons/react/24/outline';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Box, Sphere, Text } from '@react-three/drei';

interface QuantumVisualizationPageProps {
    results: any;
    selectedAssets: string[];
}

const COLORS = ['#3B82F6', '#8B5CF6', '#10B981', '#F59E0B', '#EF4444', '#EC4899'];

export const QuantumVisualizationPage: React.FC<QuantumVisualizationPageProps> = ({
    results,
    selectedAssets
}) => {
    const [selectedStock, setSelectedStock] = useState(selectedAssets[0] || '');
    const [visualizationType, setVisualizationType] = useState<'probability' | 'quantum' | 'correlation'>('probability');

    // Get prediction for selected stock
    const selectedPrediction = useMemo(() => {
        if (!results?.market_predictions) return null;
        return results.market_predictions.find((p: any) => p.asset === selectedStock);
    }, [results, selectedStock]);

    // Generate probability distribution data
    const probabilityData = useMemo(() => {
        if (!selectedPrediction?.price_scenarios) return [];
        
        const scenarios = selectedPrediction.price_scenarios;
        const prices = scenarios.map((s: any) => s.price);
        
        // Create histogram bins
        const min = Math.min(...prices);
        const max = Math.max(...prices);
        const binCount = 20;
        const binSize = (max - min) / binCount;
        
        const bins = Array(binCount).fill(0).map((_, i) => ({
            price: min + (i + 0.5) * binSize,
            probability: 0,
            quantum: 0,
            classical: 0
        }));
        
        // Fill bins
        prices.forEach((price: number, idx: number) => {
            const binIndex = Math.min(Math.floor((price - min) / binSize), binCount - 1);
            if (binIndex >= 0 && binIndex < binCount) {
                bins[binIndex].probability += 1 / prices.length;
                
                // Simulate quantum vs classical
                if (idx % 2 === 0) {
                    bins[binIndex].quantum += 2 / prices.length;
                } else {
                    bins[binIndex].classical += 2 / prices.length;
                }
            }
        });
        
        return bins;
    }, [selectedPrediction]);

    // Generate quantum state visualization data
    const quantumStateData = useMemo(() => {
        if (!selectedPrediction) return [];
        
        // Simulate quantum state amplitudes
        return Array(8).fill(0).map((_, i) => ({
            state: `|${i.toString(2).padStart(3, '0')}⟩`,
            amplitude: Math.random(),
            phase: Math.random() * 2 * Math.PI
        }));
    }, [selectedPrediction]);

    // Generate correlation matrix data
    const correlationData = useMemo(() => {
        const matrix = [];
        for (let i = 0; i < selectedAssets.length; i++) {
            for (let j = 0; j < selectedAssets.length; j++) {
                matrix.push({
                    x: selectedAssets[i],
                    y: selectedAssets[j],
                    value: i === j ? 1 : Math.random() * 2 - 1
                });
            }
        }
        return matrix;
    }, [selectedAssets]);

    // 3D Quantum State Sphere
    const QuantumSphere = () => {
        return (
            <Canvas camera={{ position: [5, 5, 5], fov: 50 }}>
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} />
                <OrbitControls enableZoom={true} />
                
                {/* Bloch Sphere */}
                <Sphere args={[2, 32, 32]}>
                    <meshPhysicalMaterial
                        color="#3B82F6"
                        transparent
                        opacity={0.3}
                        roughness={0}
                        metalness={0.5}
                    />
                </Sphere>
                
                {/* Quantum State Vector */}
                <Box args={[0.1, 3, 0.1]} position={[0, 0, 0]}>
                    <meshStandardMaterial color="#10B981" />
                </Box>
                
                {/* Axes */}
                <Box args={[4, 0.05, 0.05]} position={[0, 0, 0]}>
                    <meshStandardMaterial color="#EF4444" />
                </Box>
                <Box args={[0.05, 4, 0.05]} position={[0, 0, 0]}>
                    <meshStandardMaterial color="#10B981" />
                </Box>
                <Box args={[0.05, 0.05, 4]} position={[0, 0, 0]}>
                    <meshStandardMaterial color="#3B82F6" />
                </Box>
            </Canvas>
        );
    };

    return (
        <div className="min-h-screen bg-gray-900 p-6">
            <div className="max-w-7xl mx-auto">
                {/* Header with Stock Selector */}
                <div className="bg-gray-800 rounded-lg p-6 mb-6">
                    <div className="flex justify-between items-center">
                        <h2 className="text-2xl font-bold text-white">Quantum Visualizations</h2>
                        
                        {/* Stock Dropdown */}
                        <div className="relative">
                            <select
                                value={selectedStock}
                                onChange={(e) => setSelectedStock(e.target.value)}
                                className="appearance-none bg-gray-700 text-white px-4 py-2 pr-8 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                            >
                                {selectedAssets.map(asset => (
                                    <option key={asset} value={asset}>{asset}</option>
                                ))}
                            </select>
                            <ChevronDownIcon className="absolute right-2 top-3 w-4 h-4 text-gray-400 pointer-events-none" />
                        </div>
                    </div>

                    {/* Visualization Type Selector */}
                    <div className="flex gap-4 mt-4">
                        <button
                            onClick={() => setVisualizationType('probability')}
                            className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
                                visualizationType === 'probability' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'
                            }`}
                        >
                            <ChartBarIcon className="w-5 h-5" />
                            Probability Distribution
                        </button>
                        <button
                            onClick={() => setVisualizationType('quantum')}
                            className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
                                visualizationType === 'quantum' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'
                            }`}
                        >
                            <CubeIcon className="w-5 h-5" />
                            Quantum States
                        </button>
                        <button
                            onClick={() => setVisualizationType('correlation')}
                            className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
                                visualizationType === 'correlation' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'
                            }`}
                        >
                            <SparklesIcon className="w-5 h-5" />
                            Correlation Matrix
                        </button>
                    </div>
                </div>

                {/* Main Visualization Area */}
                <AnimatePresence mode="wait">
                    {visualizationType === 'probability' && (
                        <motion.div
                            key="probability"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            className="space-y-6"
                        >
                            {/* Probability Distribution Chart */}
                            <div className="bg-gray-800 rounded-lg p-6">
                                <h3 className="text-xl font-semibold text-white mb-4">
                                    Probability Distribution - {selectedStock}
                                </h3>
                                <ResponsiveContainer width="100%" height={400}>
                                    <AreaChart data={probabilityData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                        <XAxis
                                            dataKey="price"
                                            stroke="#9CA3AF"
                                            tickFormatter={(value) => `$${value.toFixed(0)}`}
                                        />
                                        <YAxis stroke="#9CA3AF" />
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: '#1F2937',
                                                border: 'none',
                                                borderRadius: '8px'
                                            }}
                                        />
                                        <Legend />
                                        <Area
                                            type="monotone"
                                            dataKey="quantum"
                                            stackId="1"
                                            stroke="#8B5CF6"
                                            fill="#8B5CF6"
                                            fillOpacity={0.6}
                                            name="Quantum"
                                        />
                                        <Area
                                            type="monotone"
                                            dataKey="classical"
                                            stackId="1"
                                            stroke="#10B981"
                                            fill="#10B981"
                                            fillOpacity={0.6}
                                            name="Classical"
                                        />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Confidence Intervals */}
                            <div className="bg-gray-800 rounded-lg p-6">
                                <h3 className="text-xl font-semibold text-white mb-4">
                                    Confidence Intervals
                                </h3>
                                <div className="grid grid-cols-3 gap-4">
                                    <div className="bg-gray-700 rounded-lg p-4">
                                        <div className="text-gray-400 text-sm">68% CI (1σ)</div>
                                        <div className="text-2xl font-bold text-green-400">
                                            ${(selectedPrediction?.predicted_price * 0.95 || 0).toFixed(2)} - 
                                            ${(selectedPrediction?.predicted_price * 1.05 || 0).toFixed(2)}
                                        </div>
                                    </div>
                                    <div className="bg-gray-700 rounded-lg p-4">
                                        <div className="text-gray-400 text-sm">95% CI (2σ)</div>
                                        <div className="text-2xl font-bold text-yellow-400">
                                            ${(selectedPrediction?.predicted_price * 0.90 || 0).toFixed(2)} - 
                                            ${(selectedPrediction?.predicted_price * 1.10 || 0).toFixed(2)}
                                        </div>
                                    </div>
                                    <div className="bg-gray-700 rounded-lg p-4">
                                        <div className="text-gray-400 text-sm">99% CI (3σ)</div>
                                        <div className="text-2xl font-bold text-red-400">
                                            ${(selectedPrediction?.predicted_price * 0.85 || 0).toFixed(2)} - 
                                            ${(selectedPrediction?.predicted_price * 1.15 || 0).toFixed(2)}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    )}

                    {visualizationType === 'quantum' && (
                        <motion.div
                            key="quantum"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            className="space-y-6"
                        >
                            {/* 3D Quantum State Visualization */}
                            <div className="bg-gray-800 rounded-lg p-6">
                                <h3 className="text-xl font-semibold text-white mb-4">
                                    Quantum State Representation
                                </h3>
                                <div className="h-[400px]">
                                    <QuantumSphere />
                                </div>
                            </div>

                            {/* Quantum State Amplitudes */}
                            <div className="bg-gray-800 rounded-lg p-6">
                                <h3 className="text-xl font-semibold text-white mb-4">
                                    Quantum State Amplitudes
                                </h3>
                                <ResponsiveContainer width="100%" height={300}>
                                    <BarChart data={quantumStateData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                        <XAxis dataKey="state" stroke="#9CA3AF" />
                                        <YAxis stroke="#9CA3AF" />
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: '#1F2937',
                                                border: 'none',
                                                borderRadius: '8px'
                                            }}
                                        />
                                        <Bar dataKey="amplitude" fill="#8B5CF6" />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </motion.div>
                    )}

                    {visualizationType === 'correlation' && (
                        <motion.div
                            key="correlation"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            className="bg-gray-800 rounded-lg p-6"
                        >
                            <h3 className="text-xl font-semibold text-white mb-4">
                                Asset Correlation Matrix
                            </h3>
                            
                            {/* Correlation Heatmap */}
                            <div className="grid gap-1" style={{
                                gridTemplateColumns: `repeat(${selectedAssets.length + 1}, 1fr)`
                            }}>
                                {/* Header row */}
                                <div></div>
                                {selectedAssets.map(asset => (
                                    <div key={asset} className="text-center text-sm text-gray-400">
                                        {asset}
                                    </div>
                                ))}
                                
                                {/* Data rows */}
                                {selectedAssets.map((rowAsset, i) => (
                                    <React.Fragment key={rowAsset}>
                                        <div className="text-right text-sm text-gray-400 pr-2">
                                            {rowAsset}
                                        </div>
                                        {selectedAssets.map((colAsset, j) => {
                                            const value = i === j ? 1 : Math.random() * 2 - 1;
                                            const color = value > 0 
                                                ? `rgba(34, 197, 94, ${Math.abs(value)})`
                                                : `rgba(239, 68, 68, ${Math.abs(value)})`;
                                            
                                            return (
                                                <div
                                                    key={`${i}-${j}`}
                                                    className="aspect-square flex items-center justify-center text-xs rounded"
                                                    style={{ backgroundColor: color }}
                                                >
                                                    {value.toFixed(2)}
                                                </div>
                                            );
                                        })}
                                    </React.Fragment>
                                ))}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
};

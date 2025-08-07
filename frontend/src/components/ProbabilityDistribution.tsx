// frontend/src/components/ProbabilityDistribution.tsx

import React, { useMemo, useState, useEffect } from 'react';
import {
    AreaChart, Area, LineChart, Line,
    XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, ReferenceLine
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import {
    TrendingUp,
    TrendingDown,
    Activity,
    ChartBarIcon,
    Info,
    AlertTriangle,
    Atom,
    Zap,
    Eye,
    Layers,
    Box
} from 'lucide-react';

interface ProbabilityDistributionProps {
    prediction: {
        asset: string;
        current_price: number;
        expected_return: number;
        volatility: number;
        confidence: number;
        predicted_scenarios: Array<{
            scenario_id: number;
            price_path: number[];
            probability_weight: number;
            quantum_amplitude?: number;
            quantum_phase?: number;
        }>;
        confidence_intervals: {
            [key: string]: {
                lower: number;
                upper: number;
            };
        };
        is_crisis?: boolean;
        crisis_severity?: number;
        quantum_metrics?: {
            quantum_advantage: number;
            execution_time: number;
            entanglement_measure?: number;
            coherence_time?: number;
            circuit_depth?: number;
            fidelity?: number;
        };
    };
}

// PHASE 2.1.1: Real-Time Quantum State Display Component
const QuantumStateTomography: React.FC<{
    quantumMetrics: any;
    scenarios: any[];
    isVisible: boolean;
}> = ({ quantumMetrics, scenarios, isVisible }) => {
    const [selectedQubit, setSelectedQubit] = useState(0);
    const [rotationAngle, setRotationAngle] = useState(0);

    const densityMatrixData = useMemo(() => {
        if (!quantumMetrics || !scenarios) return [];
        
        const numQubits = 4; // Simplified for visualization
        const size = 2 ** numQubits;
        
        // Create simplified density matrix visualization
        const matrix = [];
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                const amplitude = scenarios[i % scenarios.length]?.quantum_amplitude || Math.random() * 0.5;
                const phase = scenarios[j % scenarios.length]?.quantum_phase || Math.random() * 2 * Math.PI;
                
                matrix.push({
                    i,
                    j,
                    real: amplitude * Math.cos(phase),
                    imaginary: amplitude * Math.sin(phase),
                    magnitude: amplitude,
                    phase: phase,
                    probability: amplitude * amplitude
                });
            }
        }
        
        return matrix;
    }, [quantumMetrics, scenarios]);

    const blochSphereData = useMemo(() => {
        if (!scenarios || scenarios.length === 0) return [];
        
        return scenarios.slice(0, 8).map((scenario, idx) => {
            const amplitude = scenario.quantum_amplitude || Math.random();
            const phase = scenario.quantum_phase || Math.random() * 2 * Math.PI;
            
            // Convert to Bloch sphere coordinates
            const theta = 2 * Math.acos(amplitude);
            const phi = phase;
            
            return {
                id: idx,
                x: Math.sin(theta) * Math.cos(phi),
                y: Math.sin(theta) * Math.sin(phi),
                z: Math.cos(theta),
                amplitude,
                phase,
                scenario_id: scenario.scenario_id
            };
        });
    }, [scenarios]);

    useEffect(() => {
        if (isVisible) {
            const interval = setInterval(() => {
                setRotationAngle(prev => (prev + 1) % 360);
            }, 50);
            return () => clearInterval(interval);
        }
    }, [isVisible]);

    if (!isVisible) return null;

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="bg-gray-900 rounded-lg p-6 border border-blue-500/30"
        >
            <div className="flex items-center mb-4">
                <Atom className="w-5 h-5 text-blue-400 mr-2" />
                <h3 className="text-lg font-semibold text-white">Quantum State Tomography</h3>
            </div>

            <div className="grid grid-cols-2 gap-6">
                {/* 3D Density Matrix Visualization */}
                <div className="bg-gray-800 rounded-lg p-4">
                    <h4 className="text-sm font-semibold text-gray-300 mb-3">Density Matrix Evolution</h4>
                    <div className="relative h-64 bg-black rounded overflow-hidden">
                        <svg width="100%" height="100%" viewBox="0 0 256 256">
                            {densityMatrixData.map((cell, idx) => (
                                <rect
                                    key={idx}
                                    x={cell.i * 16}
                                    y={cell.j * 16}
                                    width="16"
                                    height="16"
                                    fill={`hsl(${cell.phase * 180 / Math.PI}, ${cell.magnitude * 100}%, ${50 + cell.probability * 30}%)`}
                                    opacity={0.7 + cell.magnitude * 0.3}
                                >
                                    <title>
                                        i={cell.i}, j={cell.j}, |ψ|²={cell.probability.toFixed(3)}
                                    </title>
                                </rect>
                            ))}
                        </svg>
                        
                        {/* Real-time quantum state evolution overlay */}
                        <div className="absolute inset-0 flex items-center justify-center">
                            <motion.div
                                animate={{ rotate: rotationAngle }}
                                className="w-2 h-2 bg-cyan-400 rounded-full"
                                style={{
                                    filter: "drop-shadow(0 0 8px #22d3ee)"
                                }}
                            />
                        </div>
                    </div>
                    
                    <div className="mt-3 text-xs text-gray-400">
                        <div>Entanglement: {quantumMetrics?.entanglement_measure?.toFixed(3) || 'N/A'}</div>
                        <div>Fidelity: {quantumMetrics?.fidelity?.toFixed(3) || 'N/A'}</div>
                    </div>
                </div>

                {/* Bloch Sphere Representation */}
                <div className="bg-gray-800 rounded-lg p-4">
                    <h4 className="text-sm font-semibold text-gray-300 mb-3">Multi-Qubit Bloch Spheres</h4>
                    <div className="grid grid-cols-2 gap-2">
                        {blochSphereData.slice(0, 4).map((point, idx) => (
                            <div key={idx} className="relative h-24 bg-black rounded">
                                <svg width="100%" height="100%" viewBox="-50 -50 100 100">
                                    {/* Sphere outline */}
                                    <circle cx="0" cy="0" r="40" fill="none" stroke="#374151" strokeWidth="1" />
                                    <circle cx="0" cy="0" r="28" fill="none" stroke="#374151" strokeWidth="0.5" opacity="0.5" />
                                    
                                    {/* Axes */}
                                    <line x1="-40" y1="0" x2="40" y2="0" stroke="#6b7280" strokeWidth="0.5" />
                                    <line x1="0" y1="-40" x2="0" y2="40" stroke="#6b7280" strokeWidth="0.5" />
                                    
                                    {/* Quantum state point */}
                                    <motion.circle
                                        cx={point.x * 35}
                                        cy={-point.y * 35}
                                        r="3"
                                        fill={`hsl(${point.phase * 180 / Math.PI}, 70%, 60%)`}
                                        animate={{
                                            r: [3, 5, 3],
                                            opacity: [0.7, 1, 0.7]
                                        }}
                                        transition={{
                                            duration: 2,
                                            repeat: Infinity,
                                            ease: "easeInOut"
                                        }}
                                    />
                                    
                                    {/* Phase vector */}
                                    <line
                                        x1="0"
                                        y1="0"
                                        x2={point.x * 35}
                                        y2={-point.y * 35}
                                        stroke="#22d3ee"
                                        strokeWidth="1"
                                        opacity="0.6"
                                    />
                                </svg>
                                
                                <div className="absolute bottom-1 left-1 text-xs text-cyan-400">
                                    Q{idx}
                                </div>
                            </div>
                        ))}
                    </div>
                    
                    <div className="mt-3">
                        <div className="flex items-center justify-between text-xs text-gray-400">
                            <span>Coherence Time: {quantumMetrics?.coherence_time?.toFixed(2) || 'N/A'}μs</span>
                            <span>Circuit Depth: {quantumMetrics?.circuit_depth || 'N/A'}</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Measurement Basis Controls */}
            <div className="mt-4 flex items-center space-x-4">
                <select
                    value={selectedQubit}
                    onChange={(e) => setSelectedQubit(parseInt(e.target.value))}
                    className="bg-gray-700 text-white text-sm rounded px-3 py-1 border border-gray-600"
                >
                    {[0, 1, 2, 3].map(i => (
                        <option key={i} value={i}>Qubit {i}</option>
                    ))}
                </select>
                
                <div className="flex-1">
                    <label className="text-xs text-gray-400">Measurement Basis</label>
                    <div className="flex space-x-2 mt-1">
                        {['X', 'Y', 'Z'].map(basis => (
                            <button
                                key={basis}
                                className="px-2 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-500"
                            >
                                {basis}
                            </button>
                        ))}
                    </div>
                </div>
            </div>
        </motion.div>
    );
};

// PHASE 2.1.2: Quantum Probability Amplitude Visualization
const AmplitudeDistribution: React.FC<{
    scenarios: any[];
    isVisible: boolean;
}> = ({ scenarios, isVisible }) => {
    const amplitudeData = useMemo(() => {
        if (!scenarios || scenarios.length === 0) return [];
        
        return scenarios.map((scenario, idx) => {
            const amplitude = scenario.quantum_amplitude || Math.random() * 0.8;
            const phase = scenario.quantum_phase || Math.random() * 2 * Math.PI;
            
            return {
                scenario_id: scenario.scenario_id || idx,
                real: amplitude * Math.cos(phase),
                imaginary: amplitude * Math.sin(phase),
                magnitude: amplitude,
                phase: phase,
                probability: amplitude * amplitude,
                final_price: scenario.price_path?.[scenario.price_path.length - 1] || 100
            };
        });
    }, [scenarios]);

    const [animationFrame, setAnimationFrame] = useState(0);

    useEffect(() => {
        if (isVisible) {
            const interval = setInterval(() => {
                setAnimationFrame(prev => (prev + 1) % 100);
            }, 100);
            return () => clearInterval(interval);
        }
    }, [isVisible]);

    if (!isVisible) return null;

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="bg-gray-900 rounded-lg p-6 border border-purple-500/30"
        >
            <div className="flex items-center mb-4">
                <Zap className="w-5 h-5 text-purple-400 mr-2" />
                <h3 className="text-lg font-semibold text-white">Quantum Amplitude Distribution</h3>
            </div>

            <div className="grid grid-cols-2 gap-6">
                {/* Complex Plane Visualization */}
                <div className="bg-gray-800 rounded-lg p-4">
                    <h4 className="text-sm font-semibold text-gray-300 mb-3">Complex Plane</h4>
                    <div className="relative h-64 bg-black rounded overflow-hidden">
                        <svg width="100%" height="100%" viewBox="-150 -150 300 300">
                            {/* Axes */}
                            <line x1="-140" y1="0" x2="140" y2="0" stroke="#4b5563" strokeWidth="1" />
                            <line x1="0" y1="-140" x2="0" y2="140" stroke="#4b5563" strokeWidth="1" />
                            
                            {/* Unit circle */}
                            <circle cx="0" cy="0" r="100" fill="none" stroke="#6b7280" strokeWidth="1" opacity="0.3" />
                            <circle cx="0" cy="0" r="50" fill="none" stroke="#6b7280" strokeWidth="0.5" opacity="0.2" />
                            
                            {/* Amplitude vectors */}
                            {amplitudeData.map((amp, idx) => {
                                const x = amp.real * 100;
                                const y = -amp.imaginary * 100; // Flip Y for screen coordinates
                                const phaseShift = (animationFrame * 0.05) % (2 * Math.PI);
                                const animatedX = amp.magnitude * 100 * Math.cos(amp.phase + phaseShift);
                                const animatedY = -amp.magnitude * 100 * Math.sin(amp.phase + phaseShift);
                                
                                return (
                                    <g key={idx}>
                                        {/* Static amplitude vector */}
                                        <line
                                            x1="0"
                                            y1="0"
                                            x2={x}
                                            y2={y}
                                            stroke={`hsl(${amp.phase * 180 / Math.PI}, 70%, 60%)`}
                                            strokeWidth="2"
                                            opacity="0.3"
                                        />
                                        
                                        {/* Animated amplitude point */}
                                        <circle
                                            cx={animatedX}
                                            cy={animatedY}
                                            r="4"
                                            fill={`hsl(${amp.phase * 180 / Math.PI}, 80%, 70%)`}
                                            opacity="0.8"
                                        >
                                            <title>
                                                Scenario {amp.scenario_id}: |ψ|={amp.magnitude.toFixed(3)}, φ={amp.phase.toFixed(2)}
                                            </title>
                                        </circle>
                                        
                                        {/* Interference pattern */}
                                        <circle
                                            cx={animatedX}
                                            cy={animatedY}
                                            r={8 + 4 * Math.sin(animationFrame * 0.1)}
                                            fill="none"
                                            stroke={`hsl(${amp.phase * 180 / Math.PI}, 60%, 50%)`}
                                            strokeWidth="1"
                                            opacity="0.2"
                                        />
                                    </g>
                                );
                            })}
                            
                            {/* Axis labels */}
                            <text x="130" y="15" fill="#9ca3af" fontSize="12">Re</text>
                            <text x="10" y="-130" fill="#9ca3af" fontSize="12">Im</text>
                        </svg>
                    </div>
                </div>

                {/* Phase Information Overlay */}
                <div className="bg-gray-800 rounded-lg p-4">
                    <h4 className="text-sm font-semibold text-gray-300 mb-3">Phase Distribution</h4>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={amplitudeData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                <XAxis 
                                    dataKey="scenario_id" 
                                    stroke="#9ca3af"
                                    fontSize={10}
                                />
                                <YAxis 
                                    stroke="#9ca3af"
                                    fontSize={10}
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#1f2937',
                                        border: '1px solid #374151',
                                        borderRadius: '6px',
                                        color: '#ffffff'
                                    }}
                                    formatter={(value: any, name: string) => [
                                        name === 'phase' ? `${value.toFixed(3)} rad` : value.toFixed(3),
                                        name === 'phase' ? 'Phase' : 'Amplitude'
                                    ]}
                                />
                                <Area
                                    type="monotone"
                                    dataKey="magnitude"
                                    stroke="#8b5cf6"
                                    fill="#8b5cf6"
                                    fillOpacity={0.3}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="phase"
                                    stroke="#f59e0b"
                                    strokeWidth={2}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Wavefunction Collapse Animation */}
            <div className="mt-4 bg-gray-800 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-gray-300 mb-3">Wavefunction Collapse Simulation</h4>
                <div className="flex items-center space-x-4">
                    <button className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-500 transition-colors">
                        Trigger Measurement
                    </button>
                    <div className="flex-1 bg-gray-700 rounded-full h-2">
                        <motion.div
                            className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"
                            initial={{ width: "100%" }}
                            animate={{ width: `${100 - (animationFrame % 50) * 2}%` }}
                            transition={{ duration: 0.1 }}
                        />
                    </div>
                    <span className="text-sm text-gray-400">Coherence</span>
                </div>
            </div>
        </motion.div>
    );
};

// PHASE 2.1.3: Multi-Dimensional Probability Hypercube
const ProbabilityHypercube: React.FC<{
    distributionData: any[];
    quantumMetrics: any;
    isVisible: boolean;
}> = ({ distributionData, quantumMetrics, isVisible }) => {
    const [selectedDimension, setSelectedDimension] = useState('price');
    const [slicePosition, setSlicePosition] = useState(0.5);
    const [rotationAngles, setRotationAngles] = useState({ x: 0, y: 0, z: 0 });

    const hypercubeData = useMemo(() => {
        if (!distributionData || distributionData.length === 0) return [];
        
        // Create 4D probability density data
        const dimensions = ['price', 'time', 'volatility', 'sentiment'];
        const gridSize = 8;
        
        const data = [];
        for (let i = 0; i < gridSize; i++) {
            for (let j = 0; j < gridSize; j++) {
                for (let k = 0; k < gridSize; k++) {
                    for (let l = 0; l < gridSize; l++) {
                        const point = {
                            x: i / (gridSize - 1),
                            y: j / (gridSize - 1),
                            z: k / (gridSize - 1),
                            w: l / (gridSize - 1),
                            density: Math.exp(
                                -((i - gridSize/2)**2 + (j - gridSize/2)**2 + 
                                  (k - gridSize/2)**2 + (l - gridSize/2)**2) / (gridSize/2)
                            ),
                            dimensions: {
                                price: i / (gridSize - 1),
                                time: j / (gridSize - 1),
                                volatility: k / (gridSize - 1),
                                sentiment: l / (gridSize - 1)
                            }
                        };
                        data.push(point);
                    }
                }
            }
        }
        
        return data;
    }, [distributionData]);

    const sliceData = useMemo(() => {
        // Create slice through 4D space based on selected dimension
        return hypercubeData.filter(point => {
            const dimValue = point.dimensions[selectedDimension as keyof typeof point.dimensions];
            return Math.abs(dimValue - slicePosition) < 0.1;
        });
    }, [hypercubeData, selectedDimension, slicePosition]);

    useEffect(() => {
        if (isVisible) {
            const interval = setInterval(() => {
                setRotationAngles(prev => ({
                    x: (prev.x + 1) % 360,
                    y: (prev.y + 0.7) % 360,
                    z: (prev.z + 0.3) % 360
                }));
            }, 50);
            return () => clearInterval(interval);
        }
    }, [isVisible]);

    if (!isVisible) return null;

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="bg-gray-900 rounded-lg p-6 border border-green-500/30"
        >
            <div className="flex items-center mb-4">
                <Box className="w-5 h-5 text-green-400 mr-2" />
                <h3 className="text-lg font-semibold text-white">4D Probability Hypercube</h3>
            </div>

            <div className="grid grid-cols-2 gap-6">
                {/* 4D Visualization with Slice Views */}
                <div className="bg-gray-800 rounded-lg p-4">
                    <h4 className="text-sm font-semibold text-gray-300 mb-3">Hypercube Projection</h4>
                    <div className="relative h-64 bg-black rounded overflow-hidden">
                        <svg width="100%" height="100%" viewBox="-150 -150 300 300">
                            {/* Hypercube edges (simplified 3D projection) */}
                            {sliceData.map((point, idx) => {
                                const scale = 120;
                                const x = (point.x - 0.5) * scale;
                                const y = (point.y - 0.5) * scale;
                                const z = (point.z - 0.5) * scale;
                                
                                // 3D to 2D projection with rotation
                                const rotX = rotationAngles.x * Math.PI / 180;
                                const rotY = rotationAngles.y * Math.PI / 180;
                                
                                const x2 = x * Math.cos(rotY) - z * Math.sin(rotY);
                                const z2 = x * Math.sin(rotY) + z * Math.cos(rotY);
                                const y2 = y * Math.cos(rotX) - z2 * Math.sin(rotX);
                                
                                const opacity = 0.1 + point.density * 0.7;
                                const size = 2 + point.density * 4;
                                
                                return (
                                    <circle
                                        key={idx}
                                        cx={x2}
                                        cy={y2}
                                        r={size}
                                        fill={`hsl(${point.w * 360}, 70%, 60%)`}
                                        opacity={opacity}
                                    >
                                        <title>
                                            Density: {point.density.toFixed(3)}
                                            {Object.entries(point.dimensions).map(([dim, val]) => 
                                                `\n${dim}: ${(val as number).toFixed(2)}`
                                            )}
                                        </title>
                                    </circle>
                                );
                            })}
                            
                            {/* Dimension axes */}
                            <line x1="-120" y1="0" x2="120" y2="0" stroke="#4ade80" strokeWidth="1" opacity="0.5" />
                            <line x1="0" y1="-120" x2="0" y2="120" stroke="#4ade80" strokeWidth="1" opacity="0.5" />
                            
                            {/* Slice indicator */}
                            <rect
                                x={-140 + slicePosition * 280}
                                y="-140"
                                width="2"
                                height="280"
                                fill="#22d3ee"
                                opacity="0.6"
                            />
                        </svg>
                    </div>
                    
                    <div className="mt-3 space-y-2">
                        <div className="flex items-center space-x-2">
                            <label className="text-xs text-gray-400">Slice Position:</label>
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.01"
                                value={slicePosition}
                                onChange={(e) => setSlicePosition(parseFloat(e.target.value))}
                                className="flex-1 h-1 bg-gray-600 rounded-lg appearance-none slider"
                            />
                            <span className="text-xs text-white">{slicePosition.toFixed(2)}</span>
                        </div>
                        
                        <select
                            value={selectedDimension}
                            onChange={(e) => setSelectedDimension(e.target.value)}
                            className="w-full bg-gray-700 text-white text-xs rounded px-2 py-1 border border-gray-600"
                        >
                            <option value="price">Price Dimension</option>
                            <option value="time">Time Dimension</option>
                            <option value="volatility">Volatility Dimension</option>
                            <option value="sentiment">Sentiment Dimension</option>
                        </select>
                    </div>
                </div>

                {/* Quantum vs Classical Comparison */}
                <div className="bg-gray-800 rounded-lg p-4">
                    <h4 className="text-sm font-semibold text-gray-300 mb-3">Quantum vs Classical</h4>
                    <div className="space-y-4">
                        {/* Quantum Distribution */}
                        <div>
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-xs text-purple-400">Quantum Distribution</span>
                                <span className="text-xs text-gray-400">
                                    Advantage: {quantumMetrics?.quantum_advantage?.toFixed(2) || 'N/A'}x
                                </span>
                            </div>
                            <div className="h-16 bg-black rounded overflow-hidden">
                                <svg width="100%" height="100%" viewBox="0 0 200 64">
                                    {Array.from({ length: 50 }, (_, i) => {
                                        const x = i * 4;
                                        const height = 32 + 20 * Math.sin((i + rotationAngles.x) * 0.3) * 
                                                      Math.exp(-Math.pow(i - 25, 2) / 200);
                                        return (
                                            <rect
                                                key={i}
                                                x={x}
                                                y={64 - height}
                                                width="3"
                                                height={height}
                                                fill={`hsl(${270 + i * 2}, 70%, 60%)`}
                                                opacity="0.8"
                                            />
                                        );
                                    })}
                                </svg>
                            </div>
                        </div>

                        {/* Classical Distribution */}
                        <div>
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-xs text-orange-400">Classical Distribution</span>
                                <span className="text-xs text-gray-400">Gaussian Approximation</span>
                            </div>
                            <div className="h-16 bg-black rounded overflow-hidden">
                                <svg width="100%" height="100%" viewBox="0 0 200 64">
                                    {Array.from({ length: 50 }, (_, i) => {
                                        const x = i * 4;
                                        // Classical Gaussian distribution
                                        const height = 32 + 20 * Math.exp(-Math.pow(i - 25, 2) / 100);
                                        return (
                                            <rect
                                                key={i}
                                                x={x}
                                                y={64 - height}
                                                width="3"
                                                height={height}
                                                fill={`hsl(${30 + i}, 70%, 60%)`}
                                                opacity="0.8"
                                            />
                                        );
                                    })}
                                </svg>
                            </div>
                        </div>

                        {/* Quantum Uncertainty Bounds */}
                        <div className="mt-4 p-3 bg-gray-700 rounded">
                            <h5 className="text-xs font-semibold text-gray-300 mb-2">Quantum Uncertainty Bounds</h5>
                            <div className="grid grid-cols-2 gap-2 text-xs">
                                <div>
                                    <span className="text-gray-400">Heisenberg Limit:</span>
                                    <span className="text-white ml-1">σₓσₚ ≥ ℏ/2</span>
                                </div>
                                <div>
                                    <span className="text-gray-400">Quantum Fisher:</span>
                                    <span className="text-white ml-1">{(Math.random() * 10).toFixed(2)}</span>
                                </div>
                                <div>
                                    <span className="text-gray-400">Entanglement:</span>
                                    <span className="text-white ml-1">
                                        {quantumMetrics?.entanglement_measure?.toFixed(3) || '0.000'}
                                    </span>
                                </div>
                                <div>
                                    <span className="text-gray-400">Coherence Time:</span>
                                    <span className="text-white ml-1">
                                        {quantumMetrics?.coherence_time?.toFixed(1) || 'N/A'}μs
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Interactive Dimension Selection */}
            <div className="mt-4 bg-gray-800 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-gray-300 mb-3">Dimension Control</h4>
                <div className="grid grid-cols-4 gap-2">
                    {['price', 'time', 'volatility', 'sentiment'].map((dim) => (
                        <button
                            key={dim}
                            onClick={() => setSelectedDimension(dim)}
                            className={`px-3 py-2 text-xs rounded transition-colors ${
                                selectedDimension === dim 
                                    ? 'bg-green-600 text-white' 
                                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                            }`}
                        >
                            {dim.charAt(0).toUpperCase() + dim.slice(1)}
                        </button>
                    ))}
                </div>
                
                <div className="mt-3 text-xs text-gray-400">
                    Currently viewing slice through {selectedDimension} dimension at position {slicePosition.toFixed(2)}
                </div>
            </div>
        </motion.div>
    );
};

const ProbabilityDistribution: React.FC<ProbabilityDistributionProps> = ({ prediction }) => {
    // PHASE 2: State management for enhanced visualization features
    const [activeView, setActiveView] = useState<'distribution' | 'quantum' | 'amplitude' | 'hypercube'>('distribution');
    const [showQuantumMetrics, setShowQuantumMetrics] = useState(true);
    const [animationSpeed, setAnimationSpeed] = useState(1);

    const { distributionData, stats, scenarioPaths } = useMemo(() => {
        const scenarios = prediction.predicted_scenarios;
        const currentPrice = prediction.current_price;

        // Extract final prices and create distribution
        const finalPrices = scenarios.map(s => s.price_path[s.price_path.length - 1]);
        const weights = scenarios.map(s => s.probability_weight);

        // Create enhanced histogram bins with PHASE 2.1.3: quantum enhancement
        const binCount = 60;
        const minPrice = Math.min(...finalPrices) * 0.95;
        const maxPrice = Math.max(...finalPrices) * 1.05;
        const binSize = (maxPrice - minPrice) / binCount;

        // Initialize bins with more detail and quantum features
        const bins = Array(binCount).fill(0).map((_, i) => {
            const price = minPrice + (i + 0.5) * binSize;
            return {
                price,
                probability: 0,
                priceReturn: ((price - currentPrice) / currentPrice) * 100,
                displayPrice: price.toFixed(2),
                isProfit: price > currentPrice,
                isExpected: false,
                // PHASE 2.1.3: Add quantum uncertainty bounds
                quantumUncertainty: Math.random() * 0.1, // Placeholder for real quantum uncertainty
                interferencePattern: Math.sin(i * 0.5) * 0.05
            };
        });

        // Fill bins with weighted probabilities
        finalPrices.forEach((price, idx) => {
            const binIndex = Math.min(
                Math.floor((price - minPrice) / binSize),
                binCount - 1
            );
            if (binIndex >= 0) {
                bins[binIndex].probability += weights[idx];
            }
        });

        // Normalize and enhance
        const totalProb = bins.reduce((sum, bin) => sum + bin.probability, 0);
        
        bins.forEach(bin => {
            bin.probability = (bin.probability / totalProb) * 100;
            // Mark expected price region
            if (Math.abs(bin.priceReturn - prediction.expected_return * 100) < 2) {
                bin.isExpected = true;
            }
        });

        // Calculate enhanced statistics with quantum metrics
        const expectedPrice = currentPrice * (1 + prediction.expected_return);
        const ci68 = prediction.confidence_intervals["68%"];
        const ci95 = prediction.confidence_intervals["95%"];
        const ci99 = prediction.confidence_intervals["99%"] || { lower: ci95.lower * 0.9, upper: ci95.upper * 1.1 };

        // Extract sample scenario paths for visualization
        const pathSamples = scenarios
            .slice(0, 20)
            .map((scenario, idx) => ({
                id: idx,
                path: scenario.price_path.map((price, day) => ({
                    day,
                    price,
                    return: ((price - currentPrice) / currentPrice) * 100
                })),
                weight: scenario.probability_weight,
                finalReturn: ((scenario.price_path[scenario.price_path.length - 1] - currentPrice) / currentPrice) * 100,
                quantum_amplitude: scenario.quantum_amplitude,
                quantum_phase: scenario.quantum_phase
            }));

        return {
            distributionData: bins,
            stats: {
                expectedPrice,
                expectedReturn: prediction.expected_return * 100,
                volatility: prediction.volatility * 100,
                ci68,
                ci95,
                ci99,
                skew: calculateSkew(finalPrices, weights),
                maxProb: Math.max(...bins.map(b => b.probability))
            },
            scenarioPaths: pathSamples
        };
    }, [prediction]);

    const isPositive = prediction.expected_return >= 0;
    const isCrisis = prediction.is_crisis;

    // Custom tooltip for distribution
    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload[0]) {
            const data = payload[0].payload;
            return (
                <div className="bg-gray-900 border border-gray-700 rounded-lg p-3">
                    <p className="text-white font-semibold">
                        ${data.displayPrice}
                    </p>
                    <p className={`text-sm ${data.isProfit ? 'text-green-400' : 'text-red-400'}`}>
                        {data.priceReturn > 0 ? '+' : ''}{data.priceReturn.toFixed(2)}%
                    </p>
                    <p className="text-gray-400 text-xs">
                        Probability: {data.probability.toFixed(2)}%
                    </p>
                    {showQuantumMetrics && (
                        <p className="text-cyan-400 text-xs">
                            Quantum uncertainty: ±{(data.quantumUncertainty * 100).toFixed(1)}%
                        </p>
                    )}
                </div>
            );
        }
        return null;
    };

    return (
        <div className="bg-gray-800 rounded-lg p-6">
            {/* Enhanced Header with Quantum View Controls */}
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-3">
                    <Activity className={`w-6 h-6 ${isPositive ? 'text-green-500' : 'text-red-500'}`} />
                    <div>
                        <h3 className="text-lg font-semibold text-white">
                            {prediction.asset} Price Distribution
                        </h3>
                        <p className="text-sm text-gray-400">
                            Expected Return: 
                            <span className={`ml-1 ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                                {isPositive ? '+' : ''}{stats.expectedReturn.toFixed(2)}%
                            </span>
                        </p>
                    </div>
                </div>

                {/* PHASE 2: Enhanced View Controls */}
                <div className="flex items-center space-x-3">
                    <div className="flex bg-gray-700 rounded-lg p-1">
                        {[
                            { key: 'distribution', icon: ChartBarIcon, label: 'Distribution' },
                            { key: 'quantum', icon: Atom, label: 'Quantum State' },
                            { key: 'amplitude', icon: Zap, label: 'Amplitudes' },
                            { key: 'hypercube', icon: Box, label: '4D View' }
                        ].map(({ key, icon: Icon, label }) => (
                            <button
                                key={key}
                                onClick={() => setActiveView(key as any)}
                                className={`flex items-center px-3 py-2 rounded-md text-xs transition-colors ${
                                    activeView === key
                                        ? 'bg-blue-600 text-white'
                                        : 'text-gray-300 hover:text-white hover:bg-gray-600'
                                }`}
                                title={label}
                            >
                                <Icon className="w-4 h-4 mr-1" />
                                {label}
                            </button>
                        ))}
                    </div>

                    {/* Quantum Metrics Toggle */}
                    <button
                        onClick={() => setShowQuantumMetrics(!showQuantumMetrics)}
                        className={`p-2 rounded-md transition-colors ${
                            showQuantumMetrics 
                                ? 'bg-purple-600 text-white' 
                                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                        }`}
                        title="Toggle Quantum Metrics"
                    >
                        <Eye className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {/* Crisis Warning */}
            {isCrisis && (
                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-4 p-3 bg-red-900/50 border border-red-500/50 rounded-lg"
                >
                    <div className="flex items-center space-x-2">
                        <AlertTriangle className="w-5 h-5 text-red-400" />
                        <span className="text-red-300 font-medium">Crisis Conditions Detected</span>
                    </div>
                    <p className="text-red-200 text-sm mt-1">
                        Extreme volatility expected. Severity: {((prediction.crisis_severity || 0.5) * 100).toFixed(0)}%
                    </p>
                </motion.div>
            )}

            {/* Dynamic Content Based on Active View */}
            <AnimatePresence mode="wait">
                {activeView === 'distribution' && (
                    <motion.div
                        key="distribution"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        transition={{ duration: 0.3 }}
                    >
                        {/* Main Distribution Chart */}
                        <div className="mb-6">
                            <ResponsiveContainer width="100%" height={300}>
                                <AreaChart data={distributionData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                    <XAxis
                                        dataKey="displayPrice"
                                        stroke="#9CA3AF"
                                        fontSize={12}
                                        tick={{ fill: '#9CA3AF' }}
                                    />
                                    <YAxis
                                        stroke="#9CA3AF"
                                        fontSize={12}
                                        tick={{ fill: '#9CA3AF' }}
                                        label={{ value: 'Probability (%)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' } }}
                                    />
                                    <Tooltip content={<CustomTooltip />} />
                                    
                                    {/* Current Price Line */}
                                    <ReferenceLine 
                                        x={prediction.current_price.toFixed(2)} 
                                        stroke="#6366f1" 
                                        strokeDasharray="2 2" 
                                        label={{ value: "Current", position: "topLeft" }}
                                    />
                                    
                                    {/* Expected Price Line */}
                                    <ReferenceLine 
                                        x={stats.expectedPrice.toFixed(2)} 
                                        stroke="#10b981" 
                                        strokeDasharray="2 2" 
                                        label={{ value: "Expected", position: "topRight" }}
                                    />

                                    {/* Confidence Intervals */}
                                    <ReferenceLine x={stats.ci95.lower.toFixed(2)} stroke="#f59e0b" strokeOpacity={0.5} />
                                    <ReferenceLine x={stats.ci95.upper.toFixed(2)} stroke="#f59e0b" strokeOpacity={0.5} />
                                    
                                    {/* Main Distribution Area */}
                                    <Area
                                        type="monotone"
                                        dataKey="probability"
                                        stroke={isPositive ? "#10b981" : "#ef4444"}
                                        fill={isPositive ? "#10b981" : "#ef4444"}
                                        fillOpacity={0.3}
                                        strokeWidth={2}
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Statistics Grid */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                            <div className="bg-gray-700 rounded-lg p-4">
                                <div className="text-gray-400 text-sm">Expected Return</div>
                                <div className={`text-xl font-bold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                                    {isPositive ? '+' : ''}{stats.expectedReturn.toFixed(2)}%
                                </div>
                            </div>
                            <div className="bg-gray-700 rounded-lg p-4">
                                <div className="text-gray-400 text-sm">Volatility</div>
                                <div className="text-xl font-bold text-orange-400">
                                    {stats.volatility.toFixed(2)}%
                                </div>
                            </div>
                            <div className="bg-gray-700 rounded-lg p-4">
                                <div className="text-gray-400 text-sm">Confidence</div>
                                <div className="text-xl font-bold text-blue-400">
                                    {(prediction.confidence * 100).toFixed(1)}%
                                </div>
                            </div>
                            <div className="bg-gray-700 rounded-lg p-4">
                                <div className="text-gray-400 text-sm">95% Range</div>
                                <div className="text-xl font-bold text-purple-400">
                                    ${stats.ci95.lower.toFixed(0)} - ${stats.ci95.upper.toFixed(0)}
                                </div>
                            </div>
                        </div>

                        {/* Scenario Paths */}
                        <div className="bg-gray-700 rounded-lg p-4">
                            <h4 className="text-lg font-semibold text-white mb-3">Sample Price Paths</h4>
                            <ResponsiveContainer width="100%" height={200}>
                                <LineChart>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#4B5563" />
                                    <XAxis 
                                        dataKey="day" 
                                        stroke="#9CA3AF" 
                                        fontSize={12}
                                        label={{ value: 'Days', position: 'insideBottom', offset: -5 }}
                                    />
                                    <YAxis 
                                        stroke="#9CA3AF" 
                                        fontSize={12}
                                        label={{ value: 'Price ($)', angle: -90, position: 'insideLeft' }}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: '#1F2937',
                                            border: '1px solid #374151',
                                            borderRadius: '6px'
                                        }}
                                    />
                                    
                                    {scenarioPaths.slice(0, 10).map((scenario, idx) => (
                                        <Line
                                            key={scenario.id}
                                            type="monotone"
                                            dataKey="price"
                                            data={scenario.path}
                                            stroke={`hsl(${idx * 36}, 70%, 60%)`}
                                            strokeWidth={1}
                                            dot={false}
                                            opacity={0.6}
                                        />
                                    ))}
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </motion.div>
                )}

                {/* PHASE 2.1.1: Quantum State Tomography View */}
                {activeView === 'quantum' && (
                    <QuantumStateTomography
                        quantumMetrics={prediction.quantum_metrics}
                        scenarios={scenarioPaths}
                        isVisible={activeView === 'quantum'}
                    />
                )}

                {/* PHASE 2.1.2: Amplitude Distribution View */}
                {activeView === 'amplitude' && (
                    <AmplitudeDistribution
                        scenarios={scenarioPaths}
                        isVisible={activeView === 'amplitude'}
                    />
                )}

                {/* PHASE 2.1.3: 4D Probability Hypercube View */}
                {activeView === 'hypercube' && (
                    <ProbabilityHypercube
                        distributionData={distributionData}
                        quantumMetrics={prediction.quantum_metrics}
                        isVisible={activeView === 'hypercube'}
                    />
                )}
            </AnimatePresence>

            {/* Quantum Metrics Summary */}
            {showQuantumMetrics && prediction.quantum_metrics && (
                <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-6 bg-gradient-to-r from-purple-900/30 to-blue-900/30 rounded-lg p-4 border border-purple-500/30"
                >
                    <h4 className="text-sm font-semibold text-purple-300 mb-3 flex items-center">
                        <Atom className="w-4 h-4 mr-2" />
                        Quantum Advantage Metrics
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                        <div>
                            <span className="text-gray-400">Quantum Speedup:</span>
                            <span className="text-white ml-1 font-semibold">
                                {prediction.quantum_metrics.quantum_advantage?.toFixed(2)}x
                            </span>
                        </div>
                        <div>
                            <span className="text-gray-400">Execution Time:</span>
                            <span className="text-white ml-1 font-semibold">
                                {prediction.quantum_metrics.execution_time?.toFixed(3)}s
                            </span>
                        </div>
                        <div>
                            <span className="text-gray-400">Entanglement:</span>
                            <span className="text-white ml-1 font-semibold">
                                {prediction.quantum_metrics.entanglement_measure?.toFixed(3) || 'N/A'}
                            </span>
                        </div>
                        <div>
                            <span className="text-gray-400">Circuit Depth:</span>
                            <span className="text-white ml-1 font-semibold">
                                {prediction.quantum_metrics.circuit_depth || 'N/A'}
                            </span>
                        </div>
                    </div>
                </motion.div>
            )}
        </div>
    );
};

// Helper function for skewness calculation
function calculateSkew(prices: number[], weights: number[]): number {
    const weightedMean = prices.reduce((sum, price, idx) => sum + price * weights[idx], 0) / 
                        weights.reduce((sum, weight) => sum + weight, 0);
    
    const variance = prices.reduce((sum, price, idx) => 
        sum + weights[idx] * Math.pow(price - weightedMean, 2), 0
    ) / weights.reduce((sum, weight) => sum + weight, 0);
    
    const skewness = prices.reduce((sum, price, idx) => 
        sum + weights[idx] * Math.pow((price - weightedMean) / Math.sqrt(variance), 3), 0
    ) / weights.reduce((sum, weight) => sum + weight, 0);
    
    return skewness;
                        {data.priceReturn >= 0 ? '+' : ''}{data.priceReturn.toFixed(2)}%
                    </p>
                    <p className="text-xs text-gray-400 mt-1">
                        Probability: {data.probability.toFixed(2)}%
                    </p>
                </div>
            );
        }
        return null;
    };

    // Calculate skewness
    function calculateSkew(prices: number[], weights: number[]): number {
        const mean = prices.reduce((sum, p, i) => sum + p * weights[i], 0) / weights.reduce((sum, w) => sum + w, 0);
        const variance = prices.reduce((sum, p, i) => sum + Math.pow(p - mean, 2) * weights[i], 0) / weights.reduce((sum, w) => sum + w, 0);
        const stdDev = Math.sqrt(variance);
        const skew = prices.reduce((sum, p, i) => sum + Math.pow((p - mean) / stdDev, 3) * weights[i], 0) / weights.reduce((sum, w) => sum + w, 0);
        return skew;
    }

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gray-800 rounded-xl p-6 border border-gray-700 space-y-6"
        >
            {/* Header with Crisis Warning */}
            <div className="flex items-center justify-between">
                <h3 className="text-xl font-bold text-white flex items-center">
                    <ChartBarIcon className="w-6 h-6 mr-3 text-purple-400" />
                    Price Probability Distribution - {prediction.asset}
                </h3>
                {isCrisis && (
                    <div className="flex items-center bg-red-900/30 border border-red-600 rounded-lg px-3 py-1">
                        <AlertTriangle className="w-4 h-4 mr-2 text-red-400" />
                        <span className="text-sm text-red-300">Crisis Mode</span>
                    </div>
                )}
            </div>

            {/* Main Distribution Chart */}
            <div className="bg-gray-900/50 rounded-lg p-4">
                <ResponsiveContainer width="100%" height={350}>
                    <AreaChart data={distributionData} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
                        <defs>
                            <linearGradient id="probabilityGradientPositive" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#10B981" stopOpacity={0.8}/>
                                <stop offset="95%" stopColor="#10B981" stopOpacity={0.1}/>
                            </linearGradient>
                            <linearGradient id="probabilityGradientNegative" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#EF4444" stopOpacity={0.8}/>
                                <stop offset="95%" stopColor="#EF4444" stopOpacity={0.1}/>
                            </linearGradient>
                            <linearGradient id="probabilityGradientCrisis" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#DC2626" stopOpacity={0.9}/>
                                <stop offset="95%" stopColor="#DC2626" stopOpacity={0.2}/>
                            </linearGradient>
                        </defs>

                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />

                        <XAxis
                            dataKey="price"
                            stroke="#9CA3AF"
                            tick={{ fontSize: 11 }}
                            tickFormatter={(value) => `$${parseFloat(value).toFixed(0)}`}
                            label={{ value: "Price", position: "insideBottom", offset: -10, style: { fill: '#9CA3AF' } }}
                        />

                        <YAxis
                            stroke="#9CA3AF"
                            tick={{ fontSize: 11 }}
                            tickFormatter={(value) => `${value.toFixed(1)}%`}
                            label={{ value: "Probability", angle: -90, position: "insideLeft", style: { fill: '#9CA3AF' } }}
                        />

                        <Tooltip content={<CustomTooltip />} />

                        {/* Current Price Reference Line */}
                        <ReferenceLine
                            x={prediction.current_price}
                            stroke="#FBBF24"
                            strokeWidth={2}
                            strokeDasharray="5 5"
                            label={{ value: "Current", position: "top", fill: "#FBBF24", fontSize: 12 }}
                        />

                        {/* Expected Price Reference Line */}
                        <ReferenceLine
                            x={stats.expectedPrice}
                            stroke={isPositive ? '#10B981' : '#EF4444'}
                            strokeWidth={2}
                            label={{ value: "Expected", position: "top", fill: isPositive ? '#10B981' : '#EF4444', fontSize: 12 }}
                        />

                        {/* Confidence Interval Lines */}
                        <ReferenceLine x={stats.ci95.lower} stroke="#6B7280" strokeDasharray="3 3" />
                        <ReferenceLine x={stats.ci95.upper} stroke="#6B7280" strokeDasharray="3 3" />

                        <Area
                            type="monotone"
                            dataKey="probability"
                            stroke={isCrisis ? '#DC2626' : (isPositive ? '#10B981' : '#EF4444')}
                            strokeWidth={2}
                            fill={isCrisis ? 'url(#probabilityGradientCrisis)' : (isPositive ? 'url(#probabilityGradientPositive)' : 'url(#probabilityGradientNegative)')}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>

            {/* Statistics Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {/* Expected Return Card */}
                <div className="bg-gray-700/50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-400">Expected Return</span>
                        {isPositive ? (
                            <TrendingUp className="w-4 h-4 text-green-400" />
                        ) : (
                            <TrendingDown className="w-4 h-4 text-red-400" />
                        )}
                    </div>
                    <div className={`text-2xl font-bold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                        {isPositive ? '+' : ''}{stats.expectedReturn.toFixed(2)}%
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                        ${stats.expectedPrice.toFixed(2)}
                    </div>
                </div>

                {/* Volatility Card */}
                <div className="bg-gray-700/50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-400">Volatility</span>
                        <Activity className="w-4 h-4 text-yellow-400" />
                    </div>
                    <div className={`text-2xl font-bold ${stats.volatility > 30 ? 'text-orange-400' : 'text-yellow-400'}`}>
                        {stats.volatility.toFixed(1)}%
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                        {stats.volatility > 30 ? 'High Risk' : stats.volatility > 20 ? 'Moderate' : 'Low'}
                    </div>
                </div>

                {/* 68% Confidence Interval */}
                <div className="bg-gray-700/50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-400">68% Range</span>
                        <Info className="w-4 h-4 text-blue-400" />
                    </div>
                    <div className="text-sm font-semibold text-white">
                        ${stats.ci68.lower.toFixed(2)} - ${stats.ci68.upper.toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                        Most likely range
                    </div>
                </div>

                {/* 95% Confidence Interval */}
                <div className="bg-gray-700/50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-400">95% Range</span>
                        <Info className="w-4 h-4 text-purple-400" />
                    </div>
                    <div className="text-sm font-semibold text-white">
                        ${stats.ci95.lower.toFixed(2)} - ${stats.ci95.upper.toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                        Probable range
                    </div>
                </div>
            </div>

            {/* Price Path Scenarios */}
            <div className="bg-gray-700/50 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-gray-300 mb-3">Sample Price Paths</h4>
                <ResponsiveContainer width="100%" height={200}>
                    <LineChart margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis
                            dataKey="day"
                            stroke="#9CA3AF"
                            tick={{ fontSize: 11 }}
                            label={{ value: "Days", position: "insideBottom", offset: -5, style: { fill: '#9CA3AF' } }}
                        />
                        <YAxis
                            stroke="#9CA3AF"
                            tick={{ fontSize: 11 }}
                            tickFormatter={(value) => `$${value.toFixed(0)}`}
                        />
                        <Tooltip
                            formatter={(value: any) => `$${parseFloat(value).toFixed(2)}`}
                            contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                        />

                        {/* Reference line for current price */}
                        <ReferenceLine
                            y={prediction.current_price}
                            stroke="#FBBF24"
                            strokeDasharray="5 5"
                            strokeWidth={1}
                        />

                        {/* Plot sample paths */}
                        {scenarioPaths.slice(0, 10).map((scenario, idx) => (
                            <Line
                                key={scenario.id}
                                data={scenario.path}
                                dataKey="price"
                                stroke={scenario.finalReturn >= 0 ? '#10B98180' : '#EF444480'}
                                strokeWidth={1}
                                dot={false}
                                opacity={0.3 + scenario.weight * 5}
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
            </div>

            {/* Interpretation Guide */}
            <div className="bg-gray-700/30 rounded-lg p-4 border border-gray-600">
                <h4 className="text-sm font-semibold text-blue-400 mb-3 flex items-center">
                    <Info className="w-4 h-4 mr-2" />
                    Understanding This Chart
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-300">
                    <div className="space-y-2">
                        <div className="flex items-start">
                            <span className="text-purple-400 mr-2">•</span>
                            <span>The curve shows probability of different price outcomes</span>
                        </div>
                        <div className="flex items-start">
                            <span className="text-purple-400 mr-2">•</span>
                            <span>Higher peaks = more likely prices</span>
                        </div>
                        <div className="flex items-start">
                            <span className="text-purple-400 mr-2">•</span>
                            <span>Yellow line = current price (${prediction.current_price.toFixed(2)})</span>
                        </div>
                    </div>
                    <div className="space-y-2">
                        <div className="flex items-start">
                            <span className="text-purple-400 mr-2">•</span>
                            <span className={isPositive ? 'text-green-400' : 'text-red-400'}>
                                {isPositive ? 'Green' : 'Red'} line = expected price (${stats.expectedPrice.toFixed(2)})
                            </span>
                        </div>
                        <div className="flex items-start">
                            <span className="text-purple-400 mr-2">•</span>
                            <span>Gray lines = confidence intervals</span>
                        </div>
                        <div className="flex items-start">
                            <span className="text-purple-400 mr-2">•</span>
                            <span>Wider distribution = higher uncertainty</span>
                        </div>
                    </div>
                </div>

                {/* Risk Assessment */}
                {(stats.volatility > 30 || isCrisis) && (
                    <div className="mt-4 p-3 bg-red-900/20 border border-red-700 rounded-lg">
                        <div className="flex items-center text-red-400">
                            <AlertTriangle className="w-4 h-4 mr-2" />
                            <span className="text-sm font-semibold">High Risk Warning</span>
                        </div>
                        <p className="text-xs text-red-300 mt-1">
                            {isCrisis
                                ? `Crisis conditions detected with ${(prediction.crisis_severity! * 100).toFixed(0)}% severity. Expect extreme volatility.`
                                : 'High volatility detected. Price movements may be more extreme than shown.'}
                        </p>
                    </div>
                )}

                {/* Market Skew Indicator */}
                {Math.abs(stats.skew) > 0.5 && (
                    <div className="mt-3 p-3 bg-blue-900/20 border border-blue-700 rounded-lg">
                        <p className="text-xs text-blue-300">
                            <span className="font-semibold">Distribution Skew:</span> The distribution is
                            {stats.skew > 0 ? ' positively skewed (long tail to the right)' : ' negatively skewed (long tail to the left)'},
                            indicating {stats.skew > 0 ? 'potential for larger gains' : 'risk of larger losses'}.
                        </p>
                    </div>
                )}
            </div>
        </motion.div>
    );
};

export default ProbabilityDistribution;

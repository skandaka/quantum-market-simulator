// frontend/src/components/App.tsx
import React, { useState, useCallback, useEffect } from 'react';
import toast, { Toaster } from 'react-hot-toast';
import DiagnosticPanel from './DiagnosticPanel';
import {
    BeakerIcon,
    ChartBarIcon,
    CpuChipIcon,
    AdjustmentsHorizontalIcon,
    SparklesIcon,
    PlayIcon,
    DocumentTextIcon,
    ArrowPathIcon
} from '@heroicons/react/24/outline';
import { motion, AnimatePresence } from 'framer-motion';

// Import components
import {
    BlochSphereNetwork,
    WavefunctionCollapse,
    MarketHypercube,
    QuantumCircuitLiveView
} from './QuantumVisualizations';

import {
    PortfolioUpload,
    QuantumPortfolioRiskAssessment,
    QuantumPortfolioShield,
    RealTimePortfolioMonitoring
} from './PortfolioIntegration';

import MarketSimulation from './MarketSimulation';
import PredictionExplanation from './PredictionExplanation';
import ProbabilityDistribution from './ProbabilityDistribution';

// Import API
import { simulationApi } from '../services/api';

// Types
interface PortfolioPosition {
    symbol: string;
    shares: number | string;
    avgCost: number | string;
    currentPrice?: number | string;
    sector?: string;
    marketCap?: string;
}

interface ComplexNumber {
    real: number;
    imag: number;
}

interface QuantumState {
    amplitude: [ComplexNumber, ComplexNumber];
    label: string;
}

interface QuantumGate {
    type: string;
    qubit: number;
    control?: number;
    target?: number;
    params?: any;
}

interface QuantumCircuit {
    qubits: number[];
}

interface MarketDataPoint {
    value: number;
    timestamp: string;
}

interface SimulationResults {
    request_id: string;
    timestamp: string;
    news_analysis: any[];
    market_predictions: any[];
    quantum_metrics?: any;
    execution_time_seconds: number;
    warnings: string[];
}

const App: React.FC = () => {
    // Core simulation state
    const [activeTab, setActiveTab] = useState('simulator');
    const [newsInputs, setNewsInputs] = useState(['', '', '']);
    const [selectedAssets, setSelectedAssets] = useState(['AAPL', 'MSFT', 'GOOGL']);
    const [simulationMethod, setSimulationMethod] = useState('hybrid_qml');
    const [isSimulating, setIsSimulating] = useState(false);
    const [simulationResults, setSimulationResults] = useState<SimulationResults | null>(null);
    const [portfolio, setPortfolio] = useState<PortfolioPosition[]>([]);

    // Advanced settings
    const [quantumLayers, setQuantumLayers] = useState({
        sentiment: true,
        marketDynamics: true,
        uncertainty: true,
        correlations: true
    });
    const [noiseMitigation, setNoiseMitigation] = useState(true);
    const [circuitOptimization, setCircuitOptimization] = useState(true);
    const [quantumAdvantageMode, setQuantumAdvantageMode] = useState(true);
    const [timeHorizon, setTimeHorizon] = useState(7);
    const [numScenarios, setNumScenarios] = useState(1000);

    // Visualization state
    const [quantumStates, setQuantumStates] = useState<QuantumState[]>([]);
    const [correlations, setCorrelations] = useState<number[][]>([]);
    const [superposition, setSuperposition] = useState<[ComplexNumber, ComplexNumber]>([
        { real: 0.707, imag: 0 },
        { real: 0.707, imag: 0 }
    ]);
    const [measurement, setMeasurement] = useState(0);
    const [marketData, setMarketData] = useState<MarketDataPoint[]>([]);
    const [quantumCircuit, setQuantumCircuit] = useState<QuantumCircuit>({ qubits: [0, 1, 2] });
    const [quantumGates, setQuantumGates] = useState<QuantumGate[]>([]);
    const [isExecutingCircuit, setIsExecutingCircuit] = useState(false);

    // Portfolio analysis state
    const [quantumAnalysis, setQuantumAnalysis] = useState<any>(null);
    const [alerts, setAlerts] = useState<any[]>([]);
    const [quantumMetrics, setQuantumMetrics] = useState({
        entanglement: 0.8,
        coherence: 0.9,
        fidelity: 0.85
    });

    // Initialize mock data
    useEffect(() => {
        // Initialize quantum states for visualization
        const mockQuantumStates: QuantumState[] = [
            {
                amplitude: [{ real: 0.8, imag: 0.1 }, { real: 0.6, imag: -0.1 }],
                label: 'Market Sentiment'
            },
            {
                amplitude: [{ real: 0.7, imag: 0.2 }, { real: 0.7, imag: 0.1 }],
                label: 'Price Volatility'
            },
            {
                amplitude: [{ real: 0.6, imag: -0.1 }, { real: 0.8, imag: 0.2 }],
                label: 'Risk Correlation'
            }
        ];
        setQuantumStates(mockQuantumStates);

        // Initialize correlations matrix
        setCorrelations([
            [1.0, 0.7, 0.5],
            [0.7, 1.0, 0.8],
            [0.5, 0.8, 1.0]
        ]);

        // Initialize market data
        const mockMarketData: MarketDataPoint[] = Array.from({ length: 16 }, (_, i) => ({
            value: Math.random() * 2 - 1,
            timestamp: new Date(Date.now() - (15 - i) * 60000).toISOString()
        }));
        setMarketData(mockMarketData);

        // Initialize quantum gates
        setQuantumGates([
            { type: 'H', qubit: 0 },
            { type: 'RY', qubit: 1, params: { angle: Math.PI / 4 } },
            { type: 'CNOT', qubit: 0, control: 0, target: 1 },
            { type: 'RZ', qubit: 2, params: { angle: Math.PI / 6 } },
            { type: 'CNOT', qubit: 1, control: 1, target: 2 }
        ]);
    }, []);

    // Run simulation
    const runSimulation = async () => {
        if (newsInputs.some(input => !input.trim())) {
            toast.error('Please enter news content');
            return;
        }

        setIsSimulating(true);

        try {
            // First, check if backend is reachable
            console.log('Checking backend connection...');
            const healthCheck = await simulationApi.healthCheck().catch(() => null);

            if (!healthCheck) {
                toast.error('Backend server is not reachable. Please make sure the backend is running on http://localhost:8000');
                return;
            }

            console.log('Backend is reachable, preparing simulation request...');

            const request = {
                news_inputs: newsInputs.map(content => ({
                    content: content.trim(),
                    source_type: 'headline' as const
                })),
                target_assets: selectedAssets,
                simulation_method: simulationMethod as any,
                time_horizon_days: timeHorizon,
                num_scenarios: numScenarios,
                include_quantum_metrics: true,
                compare_with_classical: true
            };

            console.log('Simulation request:', request);

            const response = await simulationApi.runSimulation(request);

            console.log('Simulation response:', response);
            setSimulationResults(response);

            // Update quantum analysis for portfolio
            if (portfolio.length > 0) {
                setQuantumAnalysis({
                    riskScore: Math.random() * 0.6 + 0.2,
                    correlationRisk: Math.random() * 0.5,
                    tailRisk: Math.random() * 0.4,
                    systemicRisk: Math.random() * 0.3,
                    liquidityRisk: Math.random() * 0.2,
                    volatility: Math.random() * 0.4 + 0.1,
                    diversificationScore: Math.random() * 0.8 + 0.2,
                    shieldScore: Math.random() * 0.9 + 0.1,
                    hiddenRisks: [
                        {
                            type: 'Correlation Breakdown Risk',
                            severity: 'medium' as const,
                            description: 'Quantum analysis detected potential correlation breakdown under extreme market stress.',
                            confidence: 0.85,
                            entanglementFactor: 0.234
                        }
                    ]
                });
            }

            toast.success('Quantum simulation completed!', {
                icon: '🚀',
                duration: 4000
            });
        } catch (error: any) {
            console.error('Simulation error:', error);

            // Provide more specific error messages
            if (error.code === 'ECONNREFUSED' || error.code === 'ERR_NETWORK') {
                toast.error('Cannot connect to backend server. Please ensure the backend is running on http://localhost:8000');
            } else if (error.response?.status === 422) {
                toast.error(`Invalid request data: ${error.response.data.detail || 'Please check your input'}`);
            } else if (error.response?.status === 500) {
                toast.error(`Server error: ${error.response.data.detail || 'Internal server error'}`);
            } else if (error.response?.data?.detail) {
                toast.error(`Simulation failed: ${error.response.data.detail}`);
            } else {
                toast.error(`Simulation failed: ${error.message || 'Unknown error occurred'}`);
            }
        } finally {
            setIsSimulating(false);
        }
    };
    // Handle portfolio upload
    const handlePortfolioLoaded = useCallback((portfolioData: PortfolioPosition[]) => {
        setPortfolio(portfolioData);
        toast.success('Portfolio loaded successfully!');
    }, []);

    // Handle hedge recommendations
    const handleHedgeRecommendation = useCallback((recommendations: any[]) => {
        toast.success(`Generated ${recommendations.length} quantum hedge recommendations`);
    }, []);

    // Execute quantum circuit
    const executeQuantumCircuit = () => {
        setIsExecutingCircuit(true);
        setTimeout(() => {
            setIsExecutingCircuit(false);
            toast.success('Quantum circuit executed successfully!');
        }, 3000);
    };

    // Tab navigation
    const tabs = [
        { id: 'simulator', label: 'Quantum Simulator', icon: BeakerIcon },
        { id: 'results', label: 'Results & Analysis', icon: ChartBarIcon },
        { id: 'visualization', label: 'Visualizations', icon: CpuChipIcon },
        { id: 'portfolio', label: 'Portfolio Analysis', icon: AdjustmentsHorizontalIcon }
    ];

    return (
        <div className="min-h-screen bg-gray-900 text-white">
            <Toaster position="top-right" />

            {/* Header */}
            <header className="bg-gray-800 border-b border-gray-700">
                <div className="container mx-auto px-4 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                            <SparklesIcon className="w-8 h-8 text-purple-400" />
                            <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-blue-500 bg-clip-text text-transparent">
                                Quantum Market Simulator
                            </h1>
                        </div>
                        <div className="flex items-center space-x-4">
                            <div className="text-sm text-gray-400">
                                Advanced Market Analysis
                            </div>
                            {simulationResults && (
                                <div className="flex items-center space-x-2 text-sm">
                                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                                    <span className="text-green-400">Analysis Complete</span>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </header>

            {/* Tab Navigation */}
            <div className="bg-gray-800 border-b border-gray-700">
                <div className="container mx-auto px-4">
                    <div className="flex space-x-8">
                        {tabs.map(tab => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex items-center space-x-2 px-4 py-3 border-b-2 transition-colors ${
                                    activeTab === tab.id
                                        ? 'border-purple-400 text-purple-400'
                                        : 'border-transparent text-gray-400 hover:text-white'
                                }`}
                            >
                                <tab.icon className="w-5 h-5" />
                                <span>{tab.label}</span>
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <main className="container mx-auto px-4 py-8">
                {/* Quantum Simulator Tab */}
                {activeTab === 'simulator' && (
                    <div className="space-y-8">
                        {/* News Input Section */}
                        <div className="bg-gray-800 rounded-lg p-6">
                            <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                                <DocumentTextIcon className="w-6 h-6 mr-2 text-purple-400" />
                                Market News Analysis
                            </h2>
                            <div className="space-y-4">
                                {newsInputs.map((input, index) => (
                                    <div key={index}>
                                        <label className="block text-sm font-medium text-gray-300 mb-2">
                                            News Input {index + 1}
                                        </label>
                                        <textarea
                                            value={input}
                                            onChange={(e) => {
                                                const newInputs = [...newsInputs];
                                                newInputs[index] = e.target.value;
                                                setNewsInputs(newInputs);
                                            }}
                                            placeholder="Enter market news, earnings report, or financial headline..."
                                            className="w-full bg-gray-700 text-white rounded-lg px-4 py-3 focus:ring-2 focus:ring-purple-500 focus:outline-none"
                                            rows={3}
                                        />
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Asset Selection */}
                        <div className="bg-gray-800 rounded-lg p-6">
                            <h3 className="text-lg font-semibold text-white mb-4">Target Assets</h3>
                            <div className="grid grid-cols-3 gap-4">
                                {['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA'].map(asset => (
                                    <label key={asset} className="flex items-center space-x-2">
                                        <input
                                            type="checkbox"
                                            checked={selectedAssets.includes(asset)}
                                            onChange={(e) => {
                                                if (e.target.checked) {
                                                    setSelectedAssets([...selectedAssets, asset]);
                                                } else {
                                                    setSelectedAssets(selectedAssets.filter(a => a !== asset));
                                                }
                                            }}
                                            className="rounded bg-gray-700 border-gray-600 text-purple-600 focus:ring-purple-500"
                                        />
                                        <span className="text-white">{asset}</span>
                                    </label>
                                ))}
                            </div>
                        </div>

                        {/* Simulation Settings */}
                        <div className="bg-gray-800 rounded-lg p-6">
                            <h3 className="text-lg font-semibold text-white mb-4">Simulation Settings</h3>
                            <div className="grid grid-cols-2 gap-6">
                                <div>
                                    <label className="block text-sm font-medium text-gray-300 mb-2">
                                        Simulation Method
                                    </label>
                                    <select
                                        value={simulationMethod}
                                        onChange={(e) => setSimulationMethod(e.target.value)}
                                        className="w-full bg-gray-700 text-white rounded-lg px-4 py-3 border border-gray-600 focus:ring-2 focus:ring-purple-500"
                                    >
                                        <option value="hybrid_qml">Hybrid Quantum-ML</option>
                                        <option value="quantum_monte_carlo">Quantum Monte Carlo</option>
                                        <option value="quantum_walk">Quantum Walk</option>
                                        <option value="classical_baseline">Classical Baseline</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-300 mb-2">
                                        Time Horizon (days)
                                    </label>
                                    <input
                                        type="number"
                                        value={timeHorizon}
                                        onChange={(e) => setTimeHorizon(parseInt(e.target.value))}
                                        min="1"
                                        max="30"
                                        className="w-full bg-gray-700 text-white rounded-lg px-4 py-3 border border-gray-600 focus:ring-2 focus:ring-purple-500"
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Run Simulation */}
                        <div className="text-center">
                            <button
                                onClick={runSimulation}
                                disabled={isSimulating}
                                className="inline-flex items-center space-x-2 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-semibold py-3 px-8 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {isSimulating ? (
                                    <>
                                        <ArrowPathIcon className="w-5 h-5 animate-spin" />
                                        <span>Running Simulation...</span>
                                    </>
                                ) : (
                                    <>
                                        <PlayIcon className="w-5 h-5" />
                                        <span>Run Quantum Simulation</span>
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                )}

                {/* Results Tab */}
                {activeTab === 'results' && simulationResults && (
                    <div className="space-y-8">
                        <MarketSimulation
                            predictions={simulationResults.market_predictions}
                            sentimentData={simulationResults.news_analysis}
                        />

                        {simulationResults.market_predictions.map((prediction, index) => (
                            <div key={index} className="space-y-6">
                                <PredictionExplanation
                                    prediction={prediction}
                                    sentimentData={simulationResults.news_analysis}
                                />
                                <ProbabilityDistribution prediction={prediction} />
                            </div>
                        ))}
                    </div>
                )}

                {/* Visualizations Tab */}
                {activeTab === 'visualization' && (
                    <div className="space-y-8">
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                            {/* Bloch Sphere Network */}
                            <div>
                                <h3 className="text-xl font-semibold text-white mb-4">Quantum State Network</h3>
                                <BlochSphereNetwork
                                    quantumStates={quantumStates}
                                    correlations={correlations}
                                />
                            </div>

                            {/* Wavefunction Collapse */}
                            <div>
                                <h3 className="text-xl font-semibold text-white mb-4">Wavefunction Collapse</h3>
                                <WavefunctionCollapse
                                    superposition={superposition}
                                    measurement={measurement}
                                    onComplete={() => setMeasurement(measurement === 0 ? 1 : 0)}
                                />
                            </div>
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                            {/* Market Hypercube */}
                            <div>
                                <h3 className="text-xl font-semibold text-white mb-4">4D Market Hypercube</h3>
                                <MarketHypercube
                                    data={marketData}
                                    selectedDimensions={['price', 'volatility', 'time', 'quantum']}
                                />
                            </div>

                            {/* Quantum Circuit */}
                            <div>
                                <h3 className="text-xl font-semibold text-white mb-4">Quantum Circuit Execution</h3>
                                <QuantumCircuitLiveView
                                    circuit={quantumCircuit}
                                    gates={quantumGates}
                                    isExecuting={isExecutingCircuit}
                                />
                                <div className="mt-4 text-center">
                                    <button
                                        onClick={executeQuantumCircuit}
                                        disabled={isExecutingCircuit}
                                        className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-2 rounded-lg disabled:opacity-50"
                                    >
                                        {isExecutingCircuit ? 'Executing...' : 'Execute Circuit'}
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Portfolio Analysis Tab */}
                {activeTab === 'portfolio' && (
                    <div className="space-y-8">
                        {/* Portfolio Upload */}
                        <PortfolioUpload onPortfolioLoaded={handlePortfolioLoaded} />

                        {portfolio.length > 0 && (
                            <>
                                {/* Risk Assessment */}
                                <QuantumPortfolioRiskAssessment
                                    portfolio={portfolio}
                                    quantumAnalysis={quantumAnalysis}
                                />

                                {/* Portfolio Shield */}
                                <QuantumPortfolioShield
                                    portfolio={portfolio}
                                    marketConditions={{
                                        volatility: 0.25,
                                        trend: 'neutral',
                                        uncertainty: 0.4
                                    }}
                                    onHedgeRecommendation={handleHedgeRecommendation}
                                />

                                {/* Real-time Monitoring */}
                                <RealTimePortfolioMonitoring
                                    portfolio={portfolio}
                                    alerts={alerts}
                                    quantumMetrics={quantumMetrics}
                                />
                            </>
                        )}
                    </div>
                )}
            </main>
        </div>
    );
};

export default App;
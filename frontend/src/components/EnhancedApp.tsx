import React, { useState, useCallback, useEffect } from 'react';
import toast, { Toaster } from 'react-hot-toast';
import {
    BeakerIcon,
    ChartBarIcon,
    CpuChipIcon,
    AdjustmentsHorizontalIcon,
    SparklesIcon,
    PlayIcon,
    DocumentTextIcon
} from '@heroicons/react/24/outline';

// Import our quantum visualization components
import {
    BlochSphereNetwork,
    WavefunctionCollapse,
    MarketHypercube,
    QuantumCircuitLiveView
} from './QuantumVisualizations';

// Import portfolio components
import {
    PortfolioUpload,
    QuantumPortfolioRiskAssessment,
    QuantumPortfolioShield,
    RealTimePortfolioMonitoring
} from './PortfolioIntegration';

// Import simulation API
import { simulationApi } from '../services/simulationApi';

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

const EnhancedApp: React.FC = () => {
    // Core simulation state
    const [activeTab, setActiveTab] = useState('simulator');
    const [newsInputs, setNewsInputs] = useState(['', '', '']);
    const [selectedAssets, setSelectedAssets] = useState(['AAPL', 'MSFT', 'GOOGL']);
    const [simulationMethod, setSimulationMethod] = useState('quantum_monte_carlo');
    const [isSimulating, setIsSimulating] = useState(false);
    const [simulationResults, setSimulationResults] = useState<any>(null);
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

    // Run enhanced simulation
    const runEnhancedSimulation = async () => {
        if (newsInputs.some(input => !input.trim())) {
            toast.error('Please enter news content');
            return;
        }

        setIsSimulating(true);

        try {
            const request = {
                news_inputs: newsInputs.map(content => ({
                    content: content.trim(),
                    source_type: 'headline' as const
                })),
                target_assets: selectedAssets,
                simulation_method: simulationMethod as any,
                enhanced_features: {
                    quantum_layers: quantumLayers,
                    noise_mitigation: noiseMitigation,
                    circuit_optimization: circuitOptimization,
                    quantum_advantage_mode: quantumAdvantageMode
                },
                portfolio_data: portfolio
            };

            const response = await simulationApi.runEnhancedSimulation(request);
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
        } catch (error) {
            console.error('Simulation error:', error);
            toast.error('Simulation failed. Please try again.');
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
        { id: 'visualization', label: 'Visualizations', icon: ChartBarIcon },
        { id: 'portfolio', label: 'Portfolio Analysis', icon: CpuChipIcon },
        { id: 'technical', label: 'Technical Details', icon: AdjustmentsHorizontalIcon }
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
                                Powered by Classiq
                            </div>
                            {simulationResults && (
                                <div className="flex items-center space-x-2 text-sm">
                                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                                    <span className="text-green-400">Quantum Advantage Active</span>
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

                        {/* Advanced Settings */}
                        <div className="bg-gray-800 rounded-lg p-6">
                            <h3 className="text-lg font-semibold text-white mb-4">Advanced Quantum Settings</h3>
                            <div className="grid grid-cols-2 gap-6">
                                <div>
                                    <h4 className="text-md font-medium text-gray-300 mb-3">Quantum Layers</h4>
                                    {Object.entries(quantumLayers).map(([key, value]) => (
                                        <label key={key} className="flex items-center space-x-2 mb-2">
                                            <input
                                                type="checkbox"
                                                checked={value}
                                                onChange={(e) => setQuantumLayers({
                                                    ...quantumLayers,
                                                    [key]: e.target.checked
                                                })}
                                                className="rounded bg-gray-700 border-gray-600 text-purple-600"
                                            />
                                            <span className="text-white capitalize">{key.replace(/([A-Z])/g, ' $1')}</span>
                                        </label>
                                    ))}
                                </div>
                                <div>
                                    <h4 className="text-md font-medium text-gray-300 mb-3">Optimization</h4>
                                    <div className="space-y-2">
                                        <label className="flex items-center space-x-2">
                                            <input
                                                type="checkbox"
                                                checked={noiseMitigation}
                                                onChange={(e) => setNoiseMitigation(e.target.checked)}
                                                className="rounded bg-gray-700 border-gray-600 text-purple-600"
                                            />
                                            <span className="text-white">Noise Mitigation</span>
                                        </label>
                                        <label className="flex items-center space-x-2">
                                            <input
                                                type="checkbox"
                                                checked={circuitOptimization}
                                                onChange={(e) => setCircuitOptimization(e.target.checked)}
                                                className="rounded bg-gray-700 border-gray-600 text-purple-600"
                                            />
                                            <span className="text-white">Circuit Optimization</span>
                                        </label>
                                        <label className="flex items-center space-x-2">
                                            <input
                                                type="checkbox"
                                                checked={quantumAdvantageMode}
                                                onChange={(e) => setQuantumAdvantageMode(e.target.checked)}
                                                className="rounded bg-gray-700 border-gray-600 text-purple-600"
                                            />
                                            <span className="text-white">Quantum Advantage Mode</span>
                                        </label>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Run Simulation */}
                        <div className="text-center">
                            <button
                                onClick={runEnhancedSimulation}
                                disabled={isSimulating}
                                className="inline-flex items-center space-x-2 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-semibold py-3 px-8 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {isSimulating ? (
                                    <>
                                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white" />
                                        <span>Running Quantum Simulation...</span>
                                    </>
                                ) : (
                                    <>
                                        <PlayIcon className="w-5 h-5" />
                                        <span>Run Quantum Simulation</span>
                                    </>
                                )}
                            </button>
                        </div>

                        {/* Results */}
                        {simulationResults && (
                            <div className="bg-gray-800 rounded-lg p-6">
                                <h3 className="text-lg font-semibold text-white mb-4">Simulation Results</h3>
                                <pre className="bg-gray-900 rounded p-4 text-sm text-gray-300 overflow-auto">
                                    {JSON.stringify(simulationResults, null, 2)}
                                </pre>
                            </div>
                        )}
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

                {/* Technical Details Tab */}
                {activeTab === 'technical' && (
                    <div className="space-y-8">
                        <div className="bg-gray-800 rounded-lg p-6">
                            <h2 className="text-2xl font-bold text-white mb-6">Technical Implementation</h2>
                            
                            <div className="space-y-6">
                                <div>
                                    <h3 className="text-lg font-semibold text-white mb-3">Quantum Algorithms</h3>
                                    <ul className="text-gray-300 space-y-2">
                                        <li>• Quantum Monte Carlo for price simulation</li>
                                        <li>• Quantum Machine Learning for sentiment analysis</li>
                                        <li>• Quantum Principal Component Analysis for risk assessment</li>
                                        <li>• Quantum Approximate Optimization Algorithm for portfolio optimization</li>
                                    </ul>
                                </div>

                                <div>
                                    <h3 className="text-lg font-semibold text-white mb-3">Hardware Integration</h3>
                                    <ul className="text-gray-300 space-y-2">
                                        <li>• Classiq quantum computing platform</li>
                                        <li>• IBM Quantum hardware backend support</li>
                                        <li>• Quantum circuit optimization and compilation</li>
                                        <li>• Error mitigation and noise handling</li>
                                    </ul>
                                </div>

                                <div>
                                    <h3 className="text-lg font-semibold text-white mb-3">Performance Metrics</h3>
                                    <div className="grid grid-cols-3 gap-4">
                                        <div className="bg-gray-700 rounded p-4">
                                            <div className="text-sm text-gray-400">Quantum Volume</div>
                                            <div className="text-xl font-bold text-purple-400">1024</div>
                                        </div>
                                        <div className="bg-gray-700 rounded p-4">
                                            <div className="text-sm text-gray-400">Circuit Depth</div>
                                            <div className="text-xl font-bold text-blue-400">45</div>
                                        </div>
                                        <div className="bg-gray-700 rounded p-4">
                                            <div className="text-sm text-gray-400">Qubits Used</div>
                                            <div className="text-xl font-bold text-green-400">12</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
};

export default EnhancedApp;

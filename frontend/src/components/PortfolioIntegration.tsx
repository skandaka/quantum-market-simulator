import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    DocumentArrowUpIcon,
    ChartBarIcon,
    ShieldCheckIcon,
    ExclamationTriangleIcon,
    SparklesIcon,
    ArrowTrendingUpIcon,
    ArrowTrendingDownIcon,
    CloudArrowUpIcon,
    LinkIcon
} from '@heroicons/react/24/outline';
import Papa from 'papaparse';
import * as d3 from 'd3';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

// Type definitions
interface PortfolioPosition {
    symbol: string;
    shares: number | string;
    avgCost: number | string;
    currentPrice?: number | string;
    sector?: string;
    marketCap?: string;
}

interface HiddenRisk {
    type: string;
    severity: 'low' | 'medium' | 'high';
    description: string;
    confidence: number;
    entanglementFactor: number;
}

interface QuantumAnalysis {
    riskScore: number;
    correlationRisk: number;
    tailRisk: number;
    systemicRisk: number;
    liquidityRisk: number;
    volatility: number;
    diversificationScore: number;
    shieldScore: number;
    hiddenRisks: HiddenRisk[];
    hedgeRecommendations?: HedgeRecommendation[];
}

interface HedgeRecommendation {
    asset: string;
    action: 'buy' | 'sell';
    quantity: number;
    effectiveness: number;
    cost: number;
    timeHorizon: string;
}

interface MarketConditions {
    volatility: number;
    trend: 'bull' | 'bear' | 'neutral';
    uncertainty: number;
}

// Portfolio Upload Component
export const PortfolioUpload = ({ onPortfolioLoaded }: {
    onPortfolioLoaded: (portfolio: PortfolioPosition[]) => void;
}) => {
    const [uploadMethod, setUploadMethod] = useState('file');
    const [isProcessing, setIsProcessing] = useState(false);
    const [dragActive, setDragActive] = useState(false);

    const handleFile = useCallback(async (file: File) => {
        setIsProcessing(true);

        try {
            const text = await file.text();

            // Parse CSV
            Papa.parse(text, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (results) => {
                    const portfolio = results.data.map((row: any) => ({
                        symbol: row.Symbol || row.symbol || row.Ticker || row.ticker,
                        shares: row.Shares || row.shares || row.Quantity || row.quantity,
                        avgCost: row['Average Cost'] || row.avgCost || row.Cost || row.cost,
                        currentPrice: row['Current Price'] || row.currentPrice || row.Price || row.price,
                        sector: row.Sector || row.sector || 'Unknown',
                        marketCap: row['Market Cap'] || row.marketCap || 'Unknown'
                    }));

                    onPortfolioLoaded(portfolio);
                    setIsProcessing(false);
                },
                error: (error: any) => {
                    console.error('CSV parsing error:', error);
                    setIsProcessing(false);
                }
            });
        } catch (error) {
            console.error('File reading error:', error);
            setIsProcessing(false);
        }
    }, [onPortfolioLoaded]);

    const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    }, [handleFile]);

    const handleDrag = useCallback((e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    }, []);

    return (
        <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                <ChartBarIcon className="w-6 h-6 mr-2 text-purple-400" />
                Portfolio Integration
            </h2>

            {/* Upload Method Selector */}
            <div className="flex space-x-4 mb-6">
                <button
                    onClick={() => setUploadMethod('file')}
                    className={`px-4 py-2 rounded-lg transition-all ${
                        uploadMethod === 'file'
                            ? 'bg-purple-600 text-white'
                            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                >
                    <DocumentArrowUpIcon className="w-5 h-5 inline mr-2" />
                    File Upload
                </button>
                <button
                    onClick={() => setUploadMethod('api')}
                    className={`px-4 py-2 rounded-lg transition-all ${
                        uploadMethod === 'api'
                            ? 'bg-purple-600 text-white'
                            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                >
                    <LinkIcon className="w-5 h-5 inline mr-2" />
                    Broker API
                </button>
                <button
                    onClick={() => setUploadMethod('manual')}
                    className={`px-4 py-2 rounded-lg transition-all ${
                        uploadMethod === 'manual'
                            ? 'bg-purple-600 text-white'
                            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                >
                    <CloudArrowUpIcon className="w-5 h-5 inline mr-2" />
                    Manual Entry
                </button>
            </div>

            {/* File Upload */}
            {uploadMethod === 'file' && (
                <div
                    className={`border-2 border-dashed rounded-lg p-8 text-center transition-all ${
                        dragActive
                            ? 'border-purple-400 bg-purple-900 bg-opacity-20'
                            : 'border-gray-600 hover:border-gray-500'
                    }`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                >
                    <input
                        type="file"
                        id="portfolio-upload"
                        accept=".csv,.xlsx,.xls"
                        onChange={(e) => e.target.files && handleFile(e.target.files[0])}
                        className="hidden"
                    />
                    <label htmlFor="portfolio-upload" className="cursor-pointer">
                        <DocumentArrowUpIcon className="w-12 h-12 mx-auto text-gray-400 mb-4" />
                        <p className="text-white mb-2">
                            {isProcessing ? 'Processing...' : 'Drop your portfolio CSV here or click to browse'}
                        </p>
                        <p className="text-sm text-gray-400">
                            Supports CSV, XLSX formats
                        </p>
                    </label>
                </div>
            )}

            {/* API Connection */}
            {uploadMethod === 'api' && (
                <div className="space-y-4">
                    <select className="w-full bg-gray-700 text-white rounded-lg px-4 py-3">
                        <option>Select Broker</option>
                        <option>TD Ameritrade</option>
                        <option>Robinhood</option>
                        <option>E*TRADE</option>
                        <option>Interactive Brokers</option>
                        <option>Fidelity</option>
                    </select>
                    <input
                        type="text"
                        placeholder="API Key"
                        className="w-full bg-gray-700 text-white rounded-lg px-4 py-3"
                    />
                    <button className="w-full bg-purple-600 text-white rounded-lg px-4 py-3 hover:bg-purple-700 transition-colors">
                        Connect Portfolio
                    </button>
                </div>
            )}

            {/* Manual Entry */}
            {uploadMethod === 'manual' && (
                <PortfolioManualEntry onAdd={onPortfolioLoaded} />
            )}
        </div>
    );
};

// Manual Portfolio Entry Component
const PortfolioManualEntry = ({ onAdd }: {
    onAdd: (positions: PortfolioPosition[]) => void;
}) => {
    const [positions, setPositions] = useState<PortfolioPosition[]>([
        { symbol: '', shares: '', avgCost: '', currentPrice: '' }
    ]);

    const addPosition = () => {
        setPositions([...positions, { symbol: '', shares: '', avgCost: '', currentPrice: '' }]);
    };

    const updatePosition = (index: number, field: keyof PortfolioPosition, value: string) => {
        const updated = [...positions];
        updated[index][field] = value as any;
        setPositions(updated);
    };

    const removePosition = (index: number) => {
        setPositions(positions.filter((_, i) => i !== index));
    };

    const handleSubmit = () => {
        const validPositions = positions.filter(p => p.symbol && p.shares);
        onAdd(validPositions);
    };

    return (
        <div className="space-y-4">
            {positions.map((position, index) => (
                <div key={index} className="flex space-x-2">
                    <input
                        type="text"
                        placeholder="Symbol"
                        value={position.symbol}
                        onChange={(e) => updatePosition(index, 'symbol', e.target.value)}
                        className="flex-1 bg-gray-700 text-white rounded px-3 py-2"
                    />
                    <input
                        type="number"
                        placeholder="Shares"
                        value={position.shares}
                        onChange={(e) => updatePosition(index, 'shares', e.target.value)}
                        className="w-24 bg-gray-700 text-white rounded px-3 py-2"
                    />
                    <input
                        type="number"
                        placeholder="Avg Cost"
                        value={position.avgCost}
                        onChange={(e) => updatePosition(index, 'avgCost', e.target.value)}
                        className="w-24 bg-gray-700 text-white rounded px-3 py-2"
                    />
                    <button
                        onClick={() => removePosition(index)}
                        className="text-red-400 hover:text-red-300"
                    >
                        ✕
                    </button>
                </div>
            ))}

            <div className="flex space-x-4">
                <button
                    onClick={addPosition}
                    className="flex-1 bg-gray-700 text-white rounded px-4 py-2 hover:bg-gray-600"
                >
                    Add Position
                </button>
                <button
                    onClick={handleSubmit}
                    className="flex-1 bg-purple-600 text-white rounded px-4 py-2 hover:bg-purple-700"
                >
                    Analyze Portfolio
                </button>
            </div>
        </div>
    );
};

// Quantum Portfolio Risk Assessment Component
export const QuantumPortfolioRiskAssessment = ({ portfolio, quantumAnalysis }: {
    portfolio: PortfolioPosition[];
    quantumAnalysis: QuantumAnalysis | null;
}) => {
    const [selectedMetric, setSelectedMetric] = useState('risk');

    const riskMetrics = useMemo(() => {
        if (!quantumAnalysis) return null;

        return {
            quantumRiskScore: quantumAnalysis.riskScore,
            correlationBreakdownRisk: quantumAnalysis.correlationRisk,
            tailRisk: quantumAnalysis.tailRisk,
            systemicRisk: quantumAnalysis.systemicRisk,
            liquidityRisk: quantumAnalysis.liquidityRisk,
            quantumVolatility: quantumAnalysis.volatility
        };
    }, [quantumAnalysis]);

    const radarData = useMemo(() => {
        if (!riskMetrics) return [];

        return [
            { metric: 'Market Risk', value: riskMetrics.quantumRiskScore * 100, fullMark: 100 },
            { metric: 'Correlation Risk', value: riskMetrics.correlationBreakdownRisk * 100, fullMark: 100 },
            { metric: 'Tail Risk', value: riskMetrics.tailRisk * 100, fullMark: 100 },
            { metric: 'Systemic Risk', value: riskMetrics.systemicRisk * 100, fullMark: 100 },
            { metric: 'Liquidity Risk', value: riskMetrics.liquidityRisk * 100, fullMark: 100 },
            { metric: 'Volatility', value: riskMetrics.quantumVolatility * 100, fullMark: 100 }
        ];
    }, [riskMetrics]);

    return (
        <div className="bg-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-semibold text-white flex items-center">
                    <ShieldCheckIcon className="w-6 h-6 mr-2 text-purple-400" />
                    Quantum Risk Assessment
                </h3>
                <div className="flex items-center space-x-2">
                    <SparklesIcon className="w-4 h-4 text-purple-400" />
                    <span className="text-sm text-purple-400">Quantum Enhanced</span>
                </div>
            </div>

            {/* Risk Score Overview */}
            <div className="grid grid-cols-3 gap-4 mb-6">
                <motion.div
                    className="bg-gray-700 rounded-lg p-4"
                    whileHover={{ scale: 1.02 }}
                >
                    <div className="text-sm text-gray-400 mb-1">Quantum Risk Score</div>
                    <div className={`text-2xl font-bold ${
                        (riskMetrics?.quantumRiskScore || 0) > 0.7 ? 'text-red-400' :
                            (riskMetrics?.quantumRiskScore || 0) > 0.4 ? 'text-yellow-400' : 'text-green-400'
                    }`}>
                        {((riskMetrics?.quantumRiskScore || 0) * 100).toFixed(1)}%
                    </div>
                </motion.div>

                <motion.div
                    className="bg-gray-700 rounded-lg p-4"
                    whileHover={{ scale: 1.02 }}
                >
                    <div className="text-sm text-gray-400 mb-1">Diversification Score</div>
                    <div className="text-2xl font-bold text-blue-400">
                        {quantumAnalysis?.diversificationScore?.toFixed(2) || 'N/A'}
                    </div>
                </motion.div>

                <motion.div
                    className="bg-gray-700 rounded-lg p-4"
                    whileHover={{ scale: 1.02 }}
                >
                    <div className="text-sm text-gray-400 mb-1">Quantum Shield Score</div>
                    <div className="text-2xl font-bold text-purple-400">
                        {quantumAnalysis?.shieldScore?.toFixed(2) || 'N/A'}
                    </div>
                </motion.div>
            </div>

            {/* Risk Radar Chart */}
            <div className="bg-gray-700 rounded-lg p-4 mb-6">
                <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={radarData}>
                        <PolarGrid stroke="#374151" />
                        <PolarAngleAxis dataKey="metric" tick={{ fill: '#9ca3af', fontSize: 12 }} />
                        <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: '#6b7280', fontSize: 10 }} />
                        <Radar
                            name="Risk Level"
                            dataKey="value"
                            stroke="#8b5cf6"
                            fill="#8b5cf6"
                            fillOpacity={0.3}
                            strokeWidth={2}
                        />
                    </RadarChart>
                </ResponsiveContainer>
            </div>

            {/* Hidden Risks Detection */}
            <div className="space-y-4">
                <h4 className="text-lg font-semibold text-white flex items-center">
                    <ExclamationTriangleIcon className="w-5 h-5 mr-2 text-yellow-400" />
                    Quantum-Detected Hidden Risks
                </h4>

                {quantumAnalysis?.hiddenRisks?.map((risk: HiddenRisk, index: number) => (
                    <motion.div
                        key={index}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="bg-gray-700 rounded-lg p-4 border-l-4 border-yellow-400"
                    >
                        <div className="flex items-center justify-between mb-2">
                            <h5 className="font-medium text-white">{risk.type}</h5>
                            <span className={`text-sm px-2 py-1 rounded ${
                                risk.severity === 'high' ? 'bg-red-500 text-white' :
                                    risk.severity === 'medium' ? 'bg-yellow-500 text-black' :
                                        'bg-green-500 text-white'
                            }`}>
                {risk.severity.toUpperCase()}
              </span>
                        </div>
                        <p className="text-sm text-gray-300 mb-2">{risk.description}</p>
                        <div className="flex items-center justify-between text-sm">
                            <span className="text-gray-400">Quantum Confidence: {(risk.confidence * 100).toFixed(1)}%</span>
                            <span className="text-purple-400">Entanglement Factor: {risk.entanglementFactor.toFixed(3)}</span>
                        </div>
                    </motion.div>
                ))}
            </div>
        </div>
    );
};

interface HedgePosition {
    original: string;
    quantumEntanglement: number;
    shieldEffectiveness: number;
    hedges: Array<{
        symbol: string;
        hedgeRatio: number;
        quantumCorrelation: number;
        effectiveness: number;
    }>;
}

interface Alert {
    id: string;
    type: 'warning' | 'danger' | 'info';
    message: string;
    timestamp: string;
}

interface QuantumMetrics {
    entanglement: number;
    coherence: number;
    fidelity: number;
}

// Quantum Portfolio Shield Component
export const QuantumPortfolioShield = ({ portfolio, marketConditions, onHedgeRecommendation }: {
    portfolio: PortfolioPosition[];
    marketConditions: MarketConditions;
    onHedgeRecommendation: (recommendations: HedgeRecommendation[]) => void;
}) => {
    const [shieldStrength, setShieldStrength] = useState(0.5);
    const [hedgePositions, setHedgePositions] = useState<HedgePosition[]>([]);

    useEffect(() => {
        // Calculate quantum hedges
        const calculateQuantumHedges = async () => {
            // Simulate quantum hedge calculation
            const hedges = portfolio.map((position: PortfolioPosition) => {
                const antiCorrelatedAssets = findAntiCorrelatedAssets(position.symbol);
                const quantumPairs = calculateQuantumPairs(position, antiCorrelatedAssets);

                return {
                    original: position.symbol,
                    hedges: quantumPairs,
                    shieldEffectiveness: Math.random() * 0.3 + 0.7, // 0.7-1.0
                    quantumEntanglement: Math.random() * 0.5 + 0.5
                };
            });

            setHedgePositions(hedges);
        };

        calculateQuantumHedges();
    }, [portfolio]);

    const findAntiCorrelatedAssets = (symbol: string): string[] => {
        // Mock function - in reality would use quantum correlation analysis
        const antiCorrelatedMap: Record<string, string[]> = {
            'AAPL': ['VXX', 'GLD', 'TLT'],
            'MSFT': ['VIX', 'GLD', 'IEF'],
            'GOOGL': ['SQQQ', 'SH', 'GLD'],
            'TSLA': ['NKLA', 'F', 'GM']
        };

        return antiCorrelatedMap[symbol] || ['VIX', 'GLD', 'TLT'];
    };

    const calculateQuantumPairs = (position: PortfolioPosition, antiCorrelatedAssets: string[]) => {
        return antiCorrelatedAssets.map((asset: string) => ({
            symbol: asset,
            hedgeRatio: Math.random() * 0.3 + 0.1, // 0.1-0.4
            quantumCorrelation: -(Math.random() * 0.5 + 0.5), // -0.5 to -1.0
            effectiveness: Math.random() * 0.3 + 0.7
        }));
    };

    return (
        <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
                <ShieldCheckIcon className="w-6 h-6 mr-2 text-green-400" />
                Quantum Portfolio Shield
            </h3>

            {/* Shield Strength Control */}
            <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                    <label className="text-sm text-gray-400">Shield Strength</label>
                    <span className="text-sm text-white">{(shieldStrength * 100).toFixed(0)}%</span>
                </div>
                <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={shieldStrength}
                    onChange={(e) => setShieldStrength(parseFloat(e.target.value))}
                    className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>Conservative</span>
                    <span>Balanced</span>
                    <span>Aggressive</span>
                </div>
            </div>

            {/* Shield Visualization */}
            <div className="relative h-64 mb-6 bg-gray-900 rounded-lg overflow-hidden">
                <svg className="absolute inset-0 w-full h-full">
                    {/* Shield effect */}
                    <defs>
                        <radialGradient id="shield-gradient">
                            <stop offset="0%" stopColor="#10b981" stopOpacity={shieldStrength} />
                            <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                        </radialGradient>
                    </defs>

                    <circle
                        cx="50%"
                        cy="50%"
                        r={100 * shieldStrength}
                        fill="url(#shield-gradient)"
                        className="animate-pulse"
                    />

                    {/* Portfolio positions */}
                    {portfolio.map((position, i) => {
                        const angle = (i / portfolio.length) * 2 * Math.PI;
                        const x = 50 + 30 * Math.cos(angle);
                        const y = 50 + 30 * Math.sin(angle);

                        return (
                            <g key={i}>
                                <circle
                                    cx={`${x}%`}
                                    cy={`${y}%`}
                                    r="5"
                                    fill="#3b82f6"
                                    stroke="#60a5fa"
                                    strokeWidth="2"
                                />
                                <text
                                    x={`${x}%`}
                                    y={`${y + 10}%`}
                                    textAnchor="middle"
                                    fill="#9ca3af"
                                    fontSize="10"
                                >
                                    {position.symbol}
                                </text>
                            </g>
                        );
                    })}
                </svg>

                <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                        <div className="text-3xl font-bold text-green-400">
                            {(shieldStrength * 100).toFixed(0)}%
                        </div>
                        <div className="text-sm text-gray-400">Protection Active</div>
                    </div>
                </div>
            </div>

            {/* Quantum Hedge Recommendations */}
            <div className="space-y-4">
                <h4 className="font-medium text-white">Quantum Hedge Recommendations</h4>

                {hedgePositions.map((hedge, index) => (
                    <motion.div
                        key={index}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="bg-gray-700 rounded-lg p-4"
                    >
                        <div className="flex items-center justify-between mb-3">
                            <div>
                                <span className="font-medium text-white">{hedge.original}</span>
                                <span className="text-gray-400 ml-2">→</span>
                            </div>
                            <div className="flex items-center space-x-2">
                                <SparklesIcon className="w-4 h-4 text-purple-400" />
                                <span className="text-sm text-purple-400">
                  {(hedge.quantumEntanglement * 100).toFixed(0)}% Entangled
                </span>
                            </div>
                        </div>

                        <div className="grid grid-cols-3 gap-2">
                            {hedge.hedges.map((pair, pairIndex) => (
                                <div
                                    key={pairIndex}
                                    className="bg-gray-800 rounded p-2 text-center"
                                >
                                    <div className="font-medium text-blue-400">{pair.symbol}</div>
                                    <div className="text-xs text-gray-400">
                                        Ratio: {(pair.hedgeRatio * 100).toFixed(0)}%
                                    </div>
                                    <div className="text-xs text-green-400">
                                        Eff: {(pair.effectiveness * 100).toFixed(0)}%
                                    </div>
                                </div>
                            ))}
                        </div>
                    </motion.div>
                ))}
            </div>

            {/* Apply Shield Button */}
            <button
                onClick={() => {
                    const recommendations: HedgeRecommendation[] = hedgePositions.map(position => ({
                        asset: position.original,
                        action: 'buy' as const,
                        quantity: Math.round(position.hedges[0]?.hedgeRatio * 100) || 100,
                        effectiveness: position.hedges[0]?.effectiveness || position.shieldEffectiveness,
                        cost: Math.round(position.quantumEntanglement * 1000),
                        timeHorizon: '30 days'
                    }));
                    onHedgeRecommendation(recommendations);
                }}
                className="w-full mt-6 bg-green-600 text-white rounded-lg px-4 py-3 hover:bg-green-700 transition-colors flex items-center justify-center"
            >
                <ShieldCheckIcon className="w-5 h-5 mr-2" />
                Apply Quantum Shield
            </button>
        </div>
    );
};

interface Anomaly {
    symbol: string;
    type: string;
    timestamp: string;
    severity: number;
    quantumSignature: number;
}

interface AlertExtended extends Alert {
    severity: 'low' | 'medium' | 'high';
}

// Real-time Portfolio Monitoring Component
export const RealTimePortfolioMonitoring = ({ portfolio, alerts, quantumMetrics }: {
    portfolio: PortfolioPosition[];
    alerts: AlertExtended[];
    quantumMetrics: QuantumMetrics;
}) => {
    const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
    const [activeAlerts, setActiveAlerts] = useState<AlertExtended[]>([]);

    useEffect(() => {
        // Simulate real-time anomaly detection
        const detectAnomalies = () => {
            const newAnomalies = portfolio.map((position: PortfolioPosition) => {
                // Simulate quantum anomaly detection
                const hasAnomaly = Math.random() < 0.1; // 10% chance

                if (hasAnomaly) {
                    return {
                        symbol: position.symbol,
                        type: ['Price Spike', 'Volume Anomaly', 'Correlation Break', 'Quantum Decoherence'][Math.floor(Math.random() * 4)],
                        timestamp: new Date().toISOString(),
                        severity: Math.random(),
                        quantumSignature: Math.random() * 0.5 + 0.5
                    };
                }
                return null;
            }).filter(Boolean) as Anomaly[];

            if (newAnomalies.length > 0) {
                setAnomalies(prev => [...newAnomalies, ...prev].slice(0, 5));
            }
        };

        const interval = setInterval(detectAnomalies, 5000); // Check every 5 seconds
        return () => clearInterval(interval);
    }, [portfolio]);

    useEffect(() => {
        setActiveAlerts(alerts || []);
    }, [alerts]);

    return (
        <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
                <ExclamationTriangleIcon className="w-6 h-6 mr-2 text-yellow-400" />
                Real-Time Portfolio Monitoring
            </h3>

            {/* Alert Summary */}
            <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-gray-700 rounded-lg p-4 text-center">
                    <div className="text-sm text-gray-400 mb-1">Low Risk</div>
                    <div className="text-2xl font-bold text-green-400">{activeAlerts.filter(a => a.severity === 'low').length}</div>
                </div>
                <div className="bg-gray-700 rounded-lg p-4 text-center">
                    <div className="text-sm text-gray-400 mb-1">Medium Risk</div>
                    <div className="text-2xl font-bold text-yellow-400">{activeAlerts.filter(a => a.severity === 'medium').length}</div>
                </div>
                <div className="bg-gray-700 rounded-lg p-4 text-center">
                    <div className="text-sm text-gray-400 mb-1">High Risk</div>
                    <div className="text-2xl font-bold text-red-400">{activeAlerts.filter(a => a.severity === 'high').length}</div>
                </div>
            </div>

            {/* Quantum Metrics */}
            <div className="bg-gray-700 rounded-lg p-4 mb-6">
                <h4 className="text-lg font-semibold text-white mb-3">Quantum System Health</h4>
                <div className="grid grid-cols-3 gap-4">
                    <div>
                        <div className="text-sm text-gray-400">Entanglement</div>
                        <div className="text-xl font-bold text-purple-400">{(quantumMetrics?.entanglement * 100 || 0).toFixed(1)}%</div>
                    </div>
                    <div>
                        <div className="text-sm text-gray-400">Coherence</div>
                        <div className="text-xl font-bold text-blue-400">{(quantumMetrics?.coherence * 100 || 0).toFixed(1)}%</div>
                    </div>
                    <div>
                        <div className="text-sm text-gray-400">Fidelity</div>
                        <div className="text-xl font-bold text-green-400">{(quantumMetrics?.fidelity * 100 || 0).toFixed(1)}%</div>
                    </div>
                </div>
            </div>

            {/* Recent Anomalies */}
            <div>
                <h4 className="text-lg font-semibold text-white mb-3">Quantum-Detected Anomalies</h4>
                <div className="space-y-3">
                    {anomalies.map((anomaly, index) => (
                        <motion.div
                            key={`${anomaly.symbol}-${anomaly.timestamp}`}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.1 }}
                            className="bg-gray-700 rounded-lg p-4 border-l-4 border-yellow-400"
                        >
                            <div className="flex items-center justify-between">
                                <div className="flex items-center space-x-3">
                                    <ExclamationTriangleIcon className="w-5 h-5 text-yellow-400" />
                                    <div>
                                        <span className="font-medium text-white">{anomaly.symbol}</span>
                                        <span className="text-sm text-gray-400 ml-2">{anomaly.type}</span>
                                    </div>
                                </div>
                                <div className="text-sm text-gray-400">
                                    {new Date(anomaly.timestamp).toLocaleTimeString()}
                                </div>
                            </div>
                            <div className="mt-2 text-sm text-gray-300 flex space-x-4">
                                <span>
                                    Severity: <span className="text-yellow-400">{(anomaly.severity * 100).toFixed(0)}%</span>
                                </span>
                                <span>
                                    Quantum Signature: <span className="text-purple-400">{anomaly.quantumSignature.toFixed(3)}</span>
                                </span>
                            </div>
                        </motion.div>
                    ))}
                    {anomalies.length === 0 && (
                        <div className="text-center text-gray-400 py-8">
                            No anomalies detected. Quantum monitoring active.
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default {
    PortfolioUpload,
    QuantumPortfolioRiskAssessment,
    QuantumPortfolioShield,
    RealTimePortfolioMonitoring
};
// frontend/src/components/ResultsDashboard.tsx

import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    ChartBarIcon,
    ArrowTrendingUpIcon,
    ArrowTrendingDownIcon,
    ExclamationTriangleIcon,
    InformationCircleIcon,
    ChevronDownIcon,
    SparklesIcon,
    BoltIcon,
    ShieldCheckIcon,
    FireIcon
} from '@heroicons/react/24/outline';
import ProbabilityDistribution from './ProbabilityDistribution';
import PredictionExplanation from './PredictionExplanation';
import {
    AreaChart, Area, BarChart, Bar, LineChart, Line,
    XAxis, YAxis, CartesianGrid, Tooltip, Legend,
    ResponsiveContainer, ReferenceLine, Cell, PieChart, Pie
} from 'recharts';

interface ResultsDashboardProps {
    results: any;
    isLoading?: boolean;
}

const ResultsDashboard: React.FC<ResultsDashboardProps> = ({ results, isLoading }) => {
    const [selectedAsset, setSelectedAsset] = useState<string>('');
    const [viewMode, setViewMode] = useState<'overview' | 'detailed' | 'comparison'>('overview');
    const [showRawData, setShowRawData] = useState(false);

    // Initialize selected asset when results change
    React.useEffect(() => {
        if (results?.market_predictions?.length > 0 && !selectedAsset) {
            setSelectedAsset(results.market_predictions[0].asset);
        }
    }, [results, selectedAsset]);

    const selectedPrediction = useMemo(() => {
        if (!results?.market_predictions || !selectedAsset) return null;
        return results.market_predictions.find((p: any) => p.asset === selectedAsset);
    }, [results, selectedAsset]);

    const impactSummary = useMemo(() => {
        if (!results?.market_predictions) return null;

        const predictions = results.market_predictions;
        const avgReturn = predictions.reduce((sum: number, p: any) => sum + p.expected_return, 0) / predictions.length;
        const maxReturn = Math.max(...predictions.map((p: any) => p.expected_return));
        const minReturn = Math.min(...predictions.map((p: any) => p.expected_return));

        return {
            average: avgReturn,
            max: maxReturn,
            min: minReturn,
            volatility: predictions.reduce((sum: number, p: any) => sum + p.volatility, 0) / predictions.length
        };
    }, [results]);

    const sentimentBreakdown = useMemo(() => {
        if (!results?.news_analysis) return [];

        const sentimentCounts: { [key: string]: number } = {};
        results.news_analysis.forEach((item: any) => {
            const sentiment = item.sentiment || 'unknown';
            sentimentCounts[sentiment] = (sentimentCounts[sentiment] || 0) + 1;
        });

        return Object.entries(sentimentCounts).map(([name, value]) => ({
            name: name.replace('_', ' ').toUpperCase(),
            value,
            fill: name.includes('very_negative') ? '#DC2626' :
                name.includes('negative') ? '#EF4444' :
                    name.includes('positive') ? '#10B981' :
                        name.includes('very_positive') ? '#059669' : '#6B7280'
        }));
    }, [results]);

    if (isLoading) {
        return (
            <div className="bg-gray-800 rounded-xl p-8 border border-gray-700">
                <div className="flex items-center justify-center space-x-3">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500"></div>
                    <span className="text-lg text-gray-300">Analyzing market impact...</span>
                </div>
            </div>
        );
    }

    if (!results) {
        return (
            <div className="bg-gray-800 rounded-xl p-8 border border-gray-700 text-center">
                <ChartBarIcon className="w-12 h-12 mx-auto mb-4 text-gray-600" />
                <p className="text-gray-400">No results to display. Run a simulation to see predictions.</p>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Header with View Mode Selector */}
            <div className="bg-gradient-to-r from-purple-900/50 to-blue-900/50 rounded-xl p-6 border border-purple-700/50">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-2xl font-bold text-white flex items-center">
                        <ChartBarIcon className="w-7 h-7 mr-3 text-purple-400" />
                        Market Impact Analysis
                    </h2>
                    <div className="flex space-x-2">
                        {['overview', 'detailed', 'comparison'].map((mode) => (
                            <button
                                key={mode}
                                onClick={() => setViewMode(mode as any)}
                                className={`px-4 py-2 rounded-lg font-medium transition-all ${
                                    viewMode === mode
                                        ? 'bg-purple-600 text-white shadow-lg'
                                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                                }`}
                            >
                                {mode.charAt(0).toUpperCase() + mode.slice(1)}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Quick Stats */}
                {impactSummary && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="bg-gray-800/50 rounded-lg p-3">
                            <p className="text-xs text-gray-400 mb-1">Average Impact</p>
                            <p className={`text-xl font-bold ${impactSummary.average >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                {impactSummary.average >= 0 ? '+' : ''}{(impactSummary.average * 100).toFixed(2)}%
                            </p>
                        </div>
                        <div className="bg-gray-800/50 rounded-lg p-3">
                            <p className="text-xs text-gray-400 mb-1">Max Impact</p>
                            <p className={`text-xl font-bold ${impactSummary.max >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                {impactSummary.max >= 0 ? '+' : ''}{(impactSummary.max * 100).toFixed(2)}%
                            </p>
                        </div>
                        <div className="bg-gray-800/50 rounded-lg p-3">
                            <p className="text-xs text-gray-400 mb-1">Min Impact</p>
                            <p className={`text-xl font-bold ${impactSummary.min >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                {impactSummary.min >= 0 ? '+' : ''}{(impactSummary.min * 100).toFixed(2)}%
                            </p>
                        </div>
                        <div className="bg-gray-800/50 rounded-lg p-3">
                            <p className="text-xs text-gray-400 mb-1">Avg Volatility</p>
                            <p className="text-xl font-bold text-yellow-400">
                                {(impactSummary.volatility * 100).toFixed(1)}%
                            </p>
                        </div>
                    </div>
                )}
            </div>

            {/* Asset Selector Dropdown */}
            {results.market_predictions && results.market_predictions.length > 0 && (
                <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold text-white flex items-center">
                            <SparklesIcon className="w-5 h-5 mr-2 text-blue-400" />
                            Select Asset for Detailed Analysis
                        </h3>
                        <div className="relative">
                            <select
                                value={selectedAsset}
                                onChange={(e) => setSelectedAsset(e.target.value)}
                                className="appearance-none bg-gray-700 border border-gray-600 text-white rounded-lg pl-4 pr-10 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                            >
                                {results.market_predictions.map((pred: any) => (
                                    <option key={pred.asset} value={pred.asset}>
                                        {pred.asset} ({pred.expected_return >= 0 ? '+' : ''}{(pred.expected_return * 100).toFixed(2)}%)
                                    </option>
                                ))}
                            </select>
                            <ChevronDownIcon className="absolute right-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
                        </div>
                    </div>

                    {/* Selected Asset Quick Info */}
                    {selectedPrediction && (
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            <div className="bg-gray-700/50 rounded-lg p-3">
                                <p className="text-xs text-gray-400">Current Price</p>
                                <p className="text-lg font-semibold text-white">
                                    ${selectedPrediction.current_price.toFixed(2)}
                                </p>
                            </div>
                            <div className="bg-gray-700/50 rounded-lg p-3">
                                <p className="text-xs text-gray-400">Expected Return</p>
                                <p className={`text-lg font-semibold flex items-center ${
                                    selectedPrediction.expected_return >= 0 ? 'text-green-400' : 'text-red-400'
                                }`}>
                                    {selectedPrediction.expected_return >= 0 ? (
                                        <ArrowTrendingUpIcon className="w-4 h-4 mr-1" />
                                    ) : (
                                        <ArrowTrendingDownIcon className="w-4 h-4 mr-1" />
                                    )}
                                    {(selectedPrediction.expected_return * 100).toFixed(2)}%
                                </p>
                            </div>
                            <div className="bg-gray-700/50 rounded-lg p-3">
                                <p className="text-xs text-gray-400">Confidence</p>
                                <p className="text-lg font-semibold text-purple-400">
                                    {(selectedPrediction.confidence * 100).toFixed(0)}%
                                </p>
                            </div>
                            <div className="bg-gray-700/50 rounded-lg p-3">
                                <p className="text-xs text-gray-400">Volatility</p>
                                <p className="text-lg font-semibold text-yellow-400">
                                    {(selectedPrediction.volatility * 100).toFixed(1)}%
                                </p>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* View Mode Content */}
            <AnimatePresence mode="wait">
                {viewMode === 'overview' && (
                    <motion.div
                        key="overview"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="space-y-6"
                    >
                        {/* Sentiment Breakdown */}
                        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                            <h3 className="text-lg font-semibold mb-4 text-white flex items-center">
                                <BoltIcon className="w-5 h-5 mr-2 text-yellow-400" />
                                Sentiment Analysis Breakdown
                            </h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div>
                                    <ResponsiveContainer width="100%" height={200}>
                                        <PieChart>
                                            <Pie
                                                data={sentimentBreakdown}
                                                cx="50%"
                                                cy="50%"
                                                innerRadius={60}
                                                outerRadius={80}
                                                paddingAngle={2}
                                                dataKey="value"
                                            >
                                                {sentimentBreakdown.map((entry, index) => (
                                                    <Cell key={`cell-${index}`} fill={entry.fill} />
                                                ))}
                                            </Pie>
                                            <Tooltip />
                                        </PieChart>
                                    </ResponsiveContainer>
                                </div>
                                <div className="space-y-2">
                                    {sentimentBreakdown.map((item) => (
                                        <div key={item.name} className="flex items-center justify-between">
                                            <div className="flex items-center">
                                                <div
                                                    className="w-3 h-3 rounded-full mr-2"
                                                    style={{ backgroundColor: item.fill }}
                                                />
                                                <span className="text-sm text-gray-300">{item.name}</span>
                                            </div>
                                            <span className="text-sm font-semibold text-white">{item.value}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Impact Comparison Chart */}
                        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                            <h3 className="text-lg font-semibold mb-4 text-white flex items-center">
                                <FireIcon className="w-5 h-5 mr-2 text-orange-400" />
                                Expected Impact by Asset
                            </h3>
                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={results.market_predictions}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                    <XAxis dataKey="asset" stroke="#9CA3AF" />
                                    <YAxis
                                        stroke="#9CA3AF"
                                        tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
                                    />
                                    <Tooltip
                                        formatter={(value: any) => `${(value * 100).toFixed(2)}%`}
                                        contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                                    />
                                    <ReferenceLine y={0} stroke="#6B7280" />
                                    <Bar dataKey="expected_return" radius={[8, 8, 0, 0]}>
                                        {results.market_predictions.map((entry: any, index: number) => (
                                            <Cell
                                                key={`cell-${index}`}
                                                fill={entry.expected_return >= 0 ? '#10B981' : '#EF4444'}
                                            />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </motion.div>
                )}

                {viewMode === 'detailed' && selectedPrediction && (
                    <motion.div
                        key="detailed"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="space-y-6"
                    >
                        {/* Probability Distribution */}
                        <ProbabilityDistribution prediction={selectedPrediction} />

                        {/* Prediction Explanation */}
                        {selectedPrediction.explanation && (
                            <PredictionExplanation
                                prediction={selectedPrediction}
                                sentimentData={results.news_analysis || []}
                            />
                        )}

                        {/* Risk Assessment */}
                        {selectedPrediction.explanation?.risk_assessment && (
                            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                                <h3 className="text-lg font-semibold mb-4 text-white flex items-center">
                                    <ShieldCheckIcon className="w-5 h-5 mr-2 text-green-400" />
                                    Risk Assessment
                                </h3>
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                    {Object.entries(selectedPrediction.explanation.risk_assessment).map(([key, value]) => (
                                        <div key={key} className="bg-gray-700/50 rounded-lg p-4">
                                            <p className="text-sm text-gray-400 mb-2">
                                                {key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                            </p>
                                            <div className={`text-lg font-semibold ${
                                                value === 'Very High' ? 'text-red-400' :
                                                    value === 'High' ? 'text-orange-400' :
                                                        value === 'Moderate' ? 'text-yellow-400' :
                                                            'text-green-400'
                                            }`}>
                                                {value as string}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </motion.div>
                )}

                {viewMode === 'comparison' && (
                    <motion.div
                        key="comparison"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="space-y-6"
                    >
                        {/* Method Comparison */}
                        {results.quantum_metrics && (
                            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                                <h3 className="text-lg font-semibold mb-4 text-white">
                                    Quantum vs Classical Performance
                                </h3>
                                <div className="grid grid-cols-2 gap-6">
                                    <div className="bg-gray-700/50 rounded-lg p-4">
                                        <h4 className="text-sm font-medium text-purple-400 mb-3">Quantum Method</h4>
                                        <div className="space-y-2">
                                            <div className="flex justify-between">
                                                <span className="text-sm text-gray-400">Accuracy</span>
                                                <span className="text-sm font-semibold text-white">
                                                    {results.quantum_metrics.accuracy?.toFixed(2) || 'N/A'}%
                                                </span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-sm text-gray-400">Processing Time</span>
                                                <span className="text-sm font-semibold text-white">
                                                    {results.quantum_metrics.processing_time?.toFixed(2) || 'N/A'}ms
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="bg-gray-700/50 rounded-lg p-4">
                                        <h4 className="text-sm font-medium text-blue-400 mb-3">Classical Method</h4>
                                        <div className="space-y-2">
                                            <div className="flex justify-between">
                                                <span className="text-sm text-gray-400">Accuracy</span>
                                                <span className="text-sm font-semibold text-white">
                                                    {results.classical_metrics?.accuracy?.toFixed(2) || 'N/A'}%
                                                </span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-sm text-gray-400">Processing Time</span>
                                                <span className="text-sm font-semibold text-white">
                                                    {results.classical_metrics?.processing_time?.toFixed(2) || 'N/A'}ms
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Warnings */}
                        {results.warnings && results.warnings.length > 0 && (
                            <div className="bg-yellow-900/20 border border-yellow-600 rounded-xl p-6">
                                <h3 className="text-lg font-semibold mb-3 text-yellow-400 flex items-center">
                                    <ExclamationTriangleIcon className="w-5 h-5 mr-2" />
                                    Analysis Warnings
                                </h3>
                                <ul className="space-y-2">
                                    {results.warnings.map((warning: string, idx: number) => (
                                        <li key={idx} className="flex items-start text-sm text-yellow-200">
                                            <span className="text-yellow-400 mr-2">•</span>
                                            <span>{warning}</span>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}

                        {/* Raw Data Toggle */}
                        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-lg font-semibold text-white flex items-center">
                                    <InformationCircleIcon className="w-5 h-5 mr-2 text-gray-400" />
                                    Raw API Response
                                </h3>
                                <button
                                    onClick={() => setShowRawData(!showRawData)}
                                    className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm text-gray-300 transition-colors"
                                >
                                    {showRawData ? 'Hide' : 'Show'} Data
                                </button>
                            </div>
                            {showRawData && (
                                <pre className="text-xs overflow-x-auto bg-gray-900 p-4 rounded-lg text-gray-300">
                                    {JSON.stringify(results, null, 2)}
                                </pre>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default ResultsDashboard;
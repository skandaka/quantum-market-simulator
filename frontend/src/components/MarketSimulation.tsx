// frontend/src/components/MarketSimulation.tsx

import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    LineChart, Line, AreaChart, Area, BarChart, Bar,
    XAxis, YAxis, CartesianGrid, Tooltip, Legend,
    ResponsiveContainer, ReferenceLine, Cell
} from 'recharts';
import {
    TrendingUp,
    TrendingDown,
    DollarSign,
    Calendar,
    Info,
    ArrowRight,
    Activity,
    AlertTriangle,
    BarChart3,
    Brain
} from 'lucide-react';

import { MarketPrediction } from '../types';

interface MarketSimulationProps {
    predictions: MarketPrediction[];
    sentimentData?: Array<{
        headline: string;
        sentiment: string;
        confidence: number;
        market_impact_keywords?: string[];
    }>;
}

const MarketSimulation: React.FC<MarketSimulationProps> = ({
                                                               predictions,
                                                               sentimentData = []
                                                           }) => {
    const [selectedAsset, setSelectedAsset] = useState(predictions[0]?.asset || '');
    const [viewMode, setViewMode] = useState<'summary' | 'scenarios' | 'distribution' | 'explanation'>('summary');
    const [showTooltip, setShowTooltip] = useState<string | null>(null);

    const selectedPrediction = predictions.find(p => p.asset === selectedAsset);

    const scenarioData = useMemo(() => {
        if (!selectedPrediction) return [];

        const maxLength = Math.max(
            ...selectedPrediction.predicted_scenarios.map(s => s.price_path.length)
        );

        const data = [];
        for (let t = 0; t < maxLength; t++) {
            const point: any = { time: t };
            selectedPrediction.predicted_scenarios.forEach((scenario, idx) => {
                point[`scenario${idx}`] = scenario.price_path[t];
            });
            data.push(point);
        }

        return data;
    }, [selectedPrediction]);

    const distributionData = useMemo(() => {
        if (!selectedPrediction) return [];

        const finalPrices = selectedPrediction.predicted_scenarios.map(
            s => s.price_path[s.price_path.length - 1]
        );

        // Create histogram bins
        const min = Math.min(...finalPrices);
        const max = Math.max(...finalPrices);
        const binCount = 30;
        const binSize = (max - min) / binCount;

        const bins = Array(binCount).fill(0).map((_, i) => ({
            price: min + (i + 0.5) * binSize,
            count: 0,
            probability: 0,
        }));

        finalPrices.forEach(price => {
            const binIndex = Math.min(
                Math.floor((price - min) / binSize),
                binCount - 1
            );
            bins[binIndex].count++;
        });

        bins.forEach(bin => {
            bin.probability = (bin.count / finalPrices.length) * 100;
        });

        return bins;
    }, [selectedPrediction]);

    if (!selectedPrediction) return null;

    const isPositive = selectedPrediction.expected_return > 0;
    const returnPercent = Math.abs(selectedPrediction.expected_return * 100);
    const currentPrice = selectedPrediction.current_price;
    const futurePrice = currentPrice * (1 + selectedPrediction.expected_return);
    const dollarChange = futurePrice - currentPrice;

    return (
        <div className="bg-gray-800 rounded-xl p-6 shadow-xl">
            {/* Header with Asset Selector */}
            <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-semibold flex items-center">
                    <span className="mr-3">📈</span> Market Simulation Results
                </h2>
                <div className="flex items-center space-x-4">
                    {predictions.length > 1 && (
                        <select
                            value={selectedAsset}
                            onChange={(e) => setSelectedAsset(e.target.value)}
                            className="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
                        >
                            {predictions.map(pred => (
                                <option key={pred.asset} value={pred.asset}>
                                    {pred.asset}
                                </option>
                            ))}
                        </select>
                    )}
                    <div className="flex bg-gray-700 rounded-lg p-1">
                        <button
                            onClick={() => setViewMode('summary')}
                            className={`px-4 py-2 rounded-md transition-all ${
                                viewMode === 'summary'
                                    ? 'bg-blue-600 text-white'
                                    : 'text-gray-400 hover:text-white'
                            }`}
                        >
                            Summary
                        </button>
                        <button
                            onClick={() => setViewMode('scenarios')}
                            className={`px-4 py-2 rounded-md transition-all ${
                                viewMode === 'scenarios'
                                    ? 'bg-blue-600 text-white'
                                    : 'text-gray-400 hover:text-white'
                            }`}
                        >
                            Scenarios
                        </button>
                        <button
                            onClick={() => setViewMode('distribution')}
                            className={`px-4 py-2 rounded-md transition-all ${
                                viewMode === 'distribution'
                                    ? 'bg-blue-600 text-white'
                                    : 'text-gray-400 hover:text-white'
                            }`}
                        >
                            Distribution
                        </button>
                        <button
                            onClick={() => setViewMode('explanation')}
                            className={`px-4 py-2 rounded-md transition-all ${
                                viewMode === 'explanation'
                                    ? 'bg-blue-600 text-white'
                                    : 'text-gray-400 hover:text-white'
                            }`}
                        >
                            Explanation
                        </button>
                    </div>
                </div>
            </div>

            <AnimatePresence mode="wait">
                {viewMode === 'summary' && (
                    <motion.div
                        key="summary"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                    >
                        {/* Clear Price Prediction Display */}
                        <div className="bg-gradient-to-r from-gray-700 to-gray-600 rounded-xl p-6 mb-6">
                            <h3 className="text-xl font-bold mb-4 text-center">
                                {selectedAsset} Price Prediction
                            </h3>

                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                                {/* Current Price */}
                                <div className="text-center">
                                    <div className="text-gray-400 mb-2 flex items-center justify-center">
                                        <DollarSign className="w-4 h-4 mr-1" />
                                        Current Price
                                    </div>
                                    <div className="text-3xl font-bold text-white">
                                        ${currentPrice.toFixed(2)}
                                    </div>
                                </div>

                                {/* Arrow */}
                                <div className="flex items-center justify-center">
                                    <ArrowRight className={`w-8 h-8 ${isPositive ? 'text-green-400' : 'text-red-400'}`} />
                                </div>

                                {/* Predicted Price */}
                                <div className="text-center">
                                    <div className="text-gray-400 mb-2 flex items-center justify-center">
                                        <Calendar className="w-4 h-4 mr-1" />
                                        Predicted Price (1 Week)
                                        <button
                                            onMouseEnter={() => setShowTooltip('predicted')}
                                            onMouseLeave={() => setShowTooltip(null)}
                                            className="ml-2 relative"
                                        >
                                            <Info className="w-4 h-4 text-gray-500 hover:text-gray-300" />
                                            {showTooltip === 'predicted' && (
                                                <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-64 bg-gray-900 text-sm text-gray-300 p-3 rounded-lg shadow-lg z-10">
                                                    This is our model's prediction for where the stock price will be in 1 week based on current news sentiment
                                                </div>
                                            )}
                                        </button>
                                    </div>
                                    <div className={`text-3xl font-bold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                                        ${futurePrice.toFixed(2)}
                                    </div>
                                    <div className={`text-lg mt-1 ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                                        {isPositive ? '+' : ''}{dollarChange.toFixed(2)} ({isPositive ? '+' : ''}{returnPercent.toFixed(2)}%)
                                    </div>
                                </div>
                            </div>

                            {/* Plain English Explanation */}
                            <div className="bg-gray-800/50 rounded-lg p-4 text-center">
                                <p className="text-lg text-gray-200">
                                    <span className="font-semibold">What this means:</span> If you own {selectedAsset} stock currently worth{' '}
                                    <span className="text-white font-semibold">${currentPrice.toFixed(2)}</span>,
                                    our model predicts it will be worth{' '}
                                    <span className={`font-semibold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                                        ${futurePrice.toFixed(2)}
                                    </span>{' '}
                                    in one week - a{' '}
                                    <span className={`font-semibold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                                        {isPositive ? 'gain' : 'loss'} of {returnPercent.toFixed(2)}%
                                    </span>.
                                </p>

                                <div className="mt-4 flex items-center justify-center space-x-6">
                                    <div className="flex items-center">
                                        <div className="w-3 h-3 bg-blue-400 rounded-full mr-2"></div>
                                        <span className="text-sm text-gray-400">
                                            Confidence: {((1 - selectedPrediction.quantum_uncertainty) * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                    <div className="flex items-center">
                                        <div className="w-3 h-3 bg-yellow-400 rounded-full mr-2"></div>
                                        <span className="text-sm text-gray-400">
                                            Volatility: {(selectedPrediction.volatility * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Key Metrics Grid */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                            <div className="bg-gray-700 rounded-lg p-4">
                                <h4 className="text-sm text-gray-400 mb-2">Confidence Interval (95%)</h4>
                                <div className="text-lg">
                                    <span className="text-blue-400">
                                        ${selectedPrediction.confidence_intervals["95%"].lower.toFixed(2)}
                                    </span>
                                    <span className="text-gray-400 mx-2">to</span>
                                    <span className="text-blue-400">
                                        ${selectedPrediction.confidence_intervals["95%"].upper.toFixed(2)}
                                    </span>
                                </div>
                                <p className="text-xs text-gray-500 mt-1">95% chance price will be in this range</p>
                            </div>

                            <div className="bg-gray-700 rounded-lg p-4">
                                <h4 className="text-sm text-gray-400 mb-2">Market Regime</h4>
                                <div className="space-y-2">
                                    <div className="flex justify-between">
                                        <span className="text-sm">Bull</span>
                                        <span className="text-green-400">
                                            {(selectedPrediction.regime_probabilities.bull * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-sm">Bear</span>
                                        <span className="text-red-400">
                                            {(selectedPrediction.regime_probabilities.bear * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-sm">Neutral</span>
                                        <span className="text-gray-400">
                                            {(selectedPrediction.regime_probabilities.neutral * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                </div>
                            </div>

                            <div className="bg-gray-700 rounded-lg p-4">
                                <h4 className="text-sm text-gray-400 mb-2">Quantum Metrics</h4>
                                <div className="space-y-2">
                                    <div className="flex justify-between">
                                        <span className="text-sm">Uncertainty</span>
                                        <span className="text-purple-400">
                                            {(selectedPrediction.quantum_uncertainty * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-sm">Time Horizon</span>
                                        <span className="text-blue-400">
                                            30 days
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                )}

                {viewMode === 'scenarios' && (
                    <motion.div
                        key="scenarios"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                    >
                        <div className="h-96">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={scenarioData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                    <XAxis
                                        dataKey="time"
                                        stroke="#9CA3AF"
                                        label={{ value: "Days", position: "insideBottom", offset: -5 }}
                                    />
                                    <YAxis
                                        stroke="#9CA3AF"
                                        label={{ value: "Price ($)", angle: -90, position: "insideLeft" }}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: '#1F2937',
                                            border: '1px solid #374151',
                                            borderRadius: '0.5rem'
                                        }}
                                        formatter={(value: number) => `${value.toFixed(2)}`}
                                    />
                                    <ReferenceLine
                                        y={selectedPrediction.current_price}
                                        stroke="#10B981"
                                        strokeDasharray="5 5"
                                        label="Current Price"
                                    />
                                    {selectedPrediction.predicted_scenarios.slice(0, 10).map((_, idx) => (
                                        <Line
                                            key={idx}
                                            type="monotone"
                                            dataKey={`scenario${idx}`}
                                            stroke={`hsl(${idx * 36}, 70%, 50%)`}
                                            strokeWidth={1}
                                            dot={false}
                                            opacity={0.6}
                                        />
                                    ))}
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                        <p className="text-sm text-gray-400 mt-4 text-center">
                            Showing top 10 most likely price scenarios over the next 30 days
                        </p>
                    </motion.div>
                )}

                {viewMode === 'distribution' && (
                    <motion.div
                        key="distribution"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                    >
                        <div className="mb-4">
                            <h3 className="text-lg font-semibold mb-2">Price Probability Distribution</h3>
                            <p className="text-sm text-gray-400">
                                This shows how likely different price outcomes are in 1 week
                            </p>
                        </div>

                        <div className="h-80">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={distributionData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                    <XAxis
                                        dataKey="price"
                                        stroke="#9CA3AF"
                                        tickFormatter={(value) => `${value.toFixed(0)}`}
                                        label={{ value: "Price ($)", position: "insideBottom", offset: -5 }}
                                    />
                                    <YAxis
                                        stroke="#9CA3AF"
                                        label={{ value: "Probability (%)", angle: -90, position: "insideLeft" }}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: '#1F2937',
                                            border: '1px solid #374151',
                                            borderRadius: '0.5rem'
                                        }}
                                        formatter={(value: number) => `${value.toFixed(1)}%`}
                                        labelFormatter={(value: number) => `Price: ${value.toFixed(2)}`}
                                    />
                                    <Bar dataKey="probability" fill="#3B82F6">
                                        {distributionData.map((entry, index) => (
                                            <Cell
                                                key={`cell-${index}`}
                                                fill={entry.price < selectedPrediction.current_price ? '#EF4444' : '#10B981'}
                                            />
                                        ))}
                                    </Bar>
                                    <ReferenceLine
                                        x={selectedPrediction.current_price}
                                        stroke="#FBBF24"
                                        strokeWidth={2}
                                        strokeDasharray="5 5"
                                        label={{ value: "Current", position: "top" }}
                                    />
                                    <ReferenceLine
                                        x={futurePrice}
                                        stroke={isPositive ? '#10B981' : '#EF4444'}
                                        strokeWidth={2}
                                        label={{ value: "Expected", position: "top" }}
                                    />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>

                        <div className="mt-4 bg-gray-700/50 rounded-lg p-4">
                            <h4 className="text-sm font-semibold text-blue-400 mb-2">How to Read This Chart</h4>
                            <ul className="text-sm text-gray-300 space-y-1">
                                <li>• <span className="text-green-400">Green bars</span> show prices above current price (gains)</li>
                                <li>• <span className="text-red-400">Red bars</span> show prices below current price (losses)</li>
                                <li>• Taller bars mean those prices are more likely</li>
                                <li>• The <span className="text-yellow-400">yellow line</span> shows current price</li>
                            </ul>
                        </div>
                    </motion.div>
                )}

                {viewMode === 'explanation' && (
                    <motion.div
                        key="explanation"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="space-y-6"
                    >
                        <div className="bg-gray-700 rounded-lg p-6">
                            <h3 className="text-lg font-semibold mb-4 flex items-center">
                                <Brain className="w-5 h-5 mr-2 text-purple-400" />
                                How We Made This Prediction
                            </h3>

                            <div className="space-y-4">
                                <div>
                                    <h4 className="text-sm font-semibold text-gray-400 mb-2">1. Sentiment Analysis</h4>
                                    <p className="text-sm text-gray-300">
                                        We analyzed {sentimentData.length} news articles about {selectedAsset}.
                                        The overall sentiment was determined by looking for positive/negative keywords
                                        and using AI to understand context.
                                    </p>
                                    {sentimentData.length > 0 && (
                                        <div className="mt-2 max-h-40 overflow-y-auto">
                                            {sentimentData.slice(0, 3).map((item, idx) => (
                                                <div key={idx} className="bg-gray-800 rounded p-2 mt-2">
                                                    <p className="text-xs text-gray-400">{item.headline}</p>
                                                    <span className={`text-xs px-2 py-1 rounded mt-1 inline-block ${
                                                        item.sentiment.includes('negative') ? 'bg-red-900 text-red-200' :
                                                            item.sentiment.includes('positive') ? 'bg-green-900 text-green-200' :
                                                                'bg-gray-600 text-gray-200'
                                                    }`}>
                                                        {item.sentiment}
                                                    </span>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>

                                <div>
                                    <h4 className="text-sm font-semibold text-gray-400 mb-2">2. Market Impact Calculation</h4>
                                    <p className="text-sm text-gray-300">
                                        Based on the sentiment score and historical patterns, we calculated the expected
                                        impact on {selectedAsset}'s price. {isPositive ? 'Positive' : 'Negative'} sentiment
                                        typically leads to {isPositive ? 'upward' : 'downward'} price movement.
                                    </p>
                                </div>

                                <div>
                                    <h4 className="text-sm font-semibold text-gray-400 mb-2">3. Scenario Generation</h4>
                                    <p className="text-sm text-gray-300">
                                        We generated {selectedPrediction.predicted_scenarios.length} different possible
                                        price paths using quantum-enhanced simulations. Each scenario represents a
                                        possible future based on current market conditions.
                                    </p>
                                </div>

                                <div>
                                    <h4 className="text-sm font-semibold text-gray-400 mb-2">4. Final Prediction</h4>
                                    <p className="text-sm text-gray-300">
                                        We averaged all scenarios weighted by their probability to get our final
                                        prediction of {isPositive ? '+' : ''}{returnPercent.toFixed(2)}% return
                                        with {((1 - selectedPrediction.quantum_uncertainty) * 100).toFixed(0)}% confidence.
                                    </p>
                                </div>
                            </div>
                        </div>

                        <div className="bg-yellow-900/20 border border-yellow-600 rounded-lg p-4">
                            <div className="flex items-start">
                                <AlertTriangle className="w-5 h-5 text-yellow-400 mr-3 mt-0.5" />
                                <div>
                                    <h4 className="text-yellow-400 font-semibold mb-1">Important Disclaimer</h4>
                                    <p className="text-sm text-yellow-200">
                                        These predictions are based on AI analysis of news sentiment and should not
                                        be used as the sole basis for investment decisions. Real markets are influenced
                                        by many factors beyond news sentiment.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default MarketSimulation;
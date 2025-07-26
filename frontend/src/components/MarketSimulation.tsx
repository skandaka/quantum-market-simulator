import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    Area,
    AreaChart,
    ReferenceLine,
} from 'recharts';
import { MarketPrediction } from '../types';
import { ArrowTrendingUpIcon, ArrowTrendingDownIcon } from '@heroicons/react/24/outline';

interface MarketSimulationProps {
    predictions: MarketPrediction[];
}

const MarketSimulation: React.FC<MarketSimulationProps> = ({ predictions }) => {
    const [selectedAsset, setSelectedAsset] = useState(predictions[0]?.asset || '');
    const [viewMode, setViewMode] = useState<'scenarios' | 'distribution'>('scenarios');
    const [showConfidenceIntervals, setShowConfidenceIntervals] = useState(true);

    const selectedPrediction = predictions.find(p => p.asset === selectedAsset);

    const chartData = useMemo(() => {
        if (!selectedPrediction) return [];

        const timePoints = selectedPrediction.predicted_scenarios[0].price_path.length;
        const data = [];

        for (let t = 0; t < timePoints; t++) {
            const point: any = {
                time: t,
                day: t === 0 ? 'Today' : `Day ${t}`,
            };

            // Calculate statistics across all scenarios
            const pricesAtTime = selectedPrediction.predicted_scenarios.map(
                s => s.price_path[t]
            );

            point.median = pricesAtTime.sort((a, b) => a - b)[Math.floor(pricesAtTime.length / 2)];
            point.mean = pricesAtTime.reduce((a, b) => a + b, 0) / pricesAtTime.length;

            // Add confidence intervals
            if (selectedPrediction.confidence_intervals['95%']) {
                const ci95 = selectedPrediction.confidence_intervals['95%'];
                const range = ci95.upper - ci95.lower;
                point.ci95Lower = point.median - (range * t / timePoints) / 2;
                point.ci95Upper = point.median + (range * t / timePoints) / 2;
            }

            if (selectedPrediction.confidence_intervals['68%']) {
                const ci68 = selectedPrediction.confidence_intervals['68%'];
                const range = ci68.upper - ci68.lower;
                point.ci68Lower = point.median - (range * t / timePoints) / 2;
                point.ci68Upper = point.median + (range * t / timePoints) / 2;
            }

            // Add individual scenario lines (show top 5)
            selectedPrediction.predicted_scenarios.slice(0, 5).forEach((scenario, idx) => {
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
        const binCount = 20;
        const binSize = (max - min) / binCount;

        const bins = Array(binCount).fill(0).map((_, i) => ({
            price: min + i * binSize,
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
            bin.probability = bin.count / finalPrices.length;
        });

        return bins;
    }, [selectedPrediction]);

    if (!selectedPrediction) return null;

    const expectedReturnPercent = selectedPrediction.expected_return * 100;
    const isPositive = expectedReturnPercent > 0;

    return (
        <div className="bg-gray-800 rounded-xl p-6 shadow-xl">
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
                            onClick={() => setViewMode('scenarios')}
                            className={`px-4 py-2 rounded-md transition-all ${
                                viewMode === 'scenarios'
                                    ? 'bg-gray-600 text-white'
                                    : 'text-gray-400'
                            }`}
                        >
                            Scenarios
                        </button>
                        <button
                            onClick={() => setViewMode('distribution')}
                            className={`px-4 py-2 rounded-md transition-all ${
                                viewMode === 'distribution'
                                    ? 'bg-gray-600 text-white'
                                    : 'text-gray-400'
                            }`}
                        >
                            Distribution
                        </button>
                    </div>
                </div>
            </div>

            {/* Key Metrics */}
            <div className="grid grid-cols-4 gap-4 mb-6">
                <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="bg-gray-700 rounded-lg p-4"
                >
                    <p className="text-sm text-gray-400">Current Price</p>
                    <p className="text-2xl font-bold">${selectedPrediction.current_price.toFixed(2)}</p>
                </motion.div>

                <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.1 }}
                    className="bg-gray-700 rounded-lg p-4"
                >
                    <p className="text-sm text-gray-400">Expected Return</p>
                    <p className={`text-2xl font-bold flex items-center ${
                        isPositive ? 'text-green-400' : 'text-red-400'
                    }`}>
                        {isPositive ? (
                            <ArrowTrendingUpIcon className="w-5 h-5 mr-1" />
                        ) : (
                            <ArrowTrendingDownIcon className="w-5 h-5 mr-1" />
                        )}
                        {expectedReturnPercent.toFixed(2)}%
                    </p>
                </motion.div>

                <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.2 }}
                    className="bg-gray-700 rounded-lg p-4"
                >
                    <p className="text-sm text-gray-400">Volatility</p>
                    <p className="text-2xl font-bold">
                        {(selectedPrediction.volatility * 100).toFixed(1)}%
                    </p>
                </motion.div>

                <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.3 }}
                    className="bg-gray-700 rounded-lg p-4"
                >
                    <p className="text-sm text-gray-400">Quantum Uncertainty</p>
                    <p className="text-2xl font-bold text-purple-400">
                        {(selectedPrediction.quantum_uncertainty * 100).toFixed(1)}%
                    </p>
                </motion.div>
            </div>

            {/* Regime Probabilities */}
            <div className="mb-6 bg-gray-700 rounded-lg p-4">
                <p className="text-sm text-gray-400 mb-3">Market Regime Probabilities</p>
                <div className="flex space-x-4">
                    <div className="flex-1">
                        <div className="flex justify-between mb-1">
                            <span className="text-green-400">Bull Market</span>
                            <span>{(selectedPrediction.regime_probabilities.bull * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-600 rounded-full h-2">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${selectedPrediction.regime_probabilities.bull * 100}%` }}
                                className="bg-green-400 h-2 rounded-full"
                            />
                        </div>
                    </div>
                    <div className="flex-1">
                        <div className="flex justify-between mb-1">
                            <span className="text-yellow-400">Neutral</span>
                            <span>{(selectedPrediction.regime_probabilities.neutral * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-600 rounded-full h-2">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${selectedPrediction.regime_probabilities.neutral * 100}%` }}
                                className="bg-yellow-400 h-2 rounded-full"
                            />
                        </div>
                    </div>
                    <div className="flex-1">
                        <div className="flex justify-between mb-1">
                            <span className="text-red-400">Bear Market</span>
                            <span>{(selectedPrediction.regime_probabilities.bear * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-600 rounded-full h-2">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${selectedPrediction.regime_probabilities.bear * 100}%` }}
                                className="bg-red-400 h-2 rounded-full"
                            />
                        </div>
                    </div>
                </div>
            </div>

            {/* Chart */}
            <div className="h-96">
                {viewMode === 'scenarios' ? (
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                            <XAxis dataKey="day" stroke="#9CA3AF" />
                            <YAxis stroke="#9CA3AF" />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: '#1F2937',
                                    border: '1px solid #374151',
                                    borderRadius: '8px',
                                }}
                            />
                            <ReferenceLine
                                y={selectedPrediction.current_price}
                                stroke="#9CA3AF"
                                strokeDasharray="5 5"
                                label="Current"
                            />

                            {/* Confidence Intervals */}
                            {showConfidenceIntervals && (
                                <>
                                    <Area
                                        type="monotone"
                                        dataKey="ci95Upper"
                                        stackId="1"
                                        stroke="none"
                                        fill="#3B82F6"
                                        fillOpacity={0.1}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="ci95Lower"
                                        stackId="2"
                                        stroke="none"
                                        fill="#3B82F6"
                                        fillOpacity={0.1}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="ci68Upper"
                                        stackId="3"
                                        stroke="none"
                                        fill="#8B5CF6"
                                        fillOpacity={0.2}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="ci68Lower"
                                        stackId="4"
                                        stroke="none"
                                        fill="#8B5CF6"
                                        fillOpacity={0.2}
                                    />
                                </>
                            )}

                            {/* Individual Scenarios */}
                            {[0, 1, 2, 3, 4].map(idx => (
                                <Line
                                    key={idx}
                                    type="monotone"
                                    dataKey={`scenario${idx}`}
                                    stroke={`hsl(${200 + idx * 30}, 70%, 50%)`}
                                    strokeWidth={1}
                                    dot={false}
                                    opacity={0.5}
                                />
                            ))}

                            {/* Median Line */}
                            <Line
                                type="monotone"
                                dataKey="median"
                                stroke="#10B981"
                                strokeWidth={3}
                                dot={false}
                                name="Median Prediction"
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                ) : (
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={distributionData}>
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
                                    border: '1px solid #374151',
                                    borderRadius: '8px',
                                }}
                                formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                            />
                            <Area
                                type="monotone"
                                dataKey="probability"
                                stroke="#8B5CF6"
                                fill="#8B5CF6"
                                fillOpacity={0.6}
                                name="Probability"
                            />
                            <ReferenceLine
                                x={selectedPrediction.current_price}
                                stroke="#9CA3AF"
                                strokeDasharray="5 5"
                                label="Current Price"
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                )}
            </div>

            {/* Chart Controls */}
            <div className="mt-4 flex justify-center">
                <label className="flex items-center space-x-2 text-sm">
                    <input
                        type="checkbox"
                        checked={showConfidenceIntervals}
                        onChange={(e) => setShowConfidenceIntervals(e.target.checked)}
                        className="rounded border-gray-600"
                    />
                    <span>Show Confidence Intervals</span>
                </label>
            </div>
        </div>
    );
};

export default MarketSimulation;
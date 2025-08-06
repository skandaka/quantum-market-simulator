// frontend/src/components/ProbabilityDistribution.tsx

import React, { useMemo } from 'react';
import {
    AreaChart, Area, LineChart, Line,
    XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, ReferenceLine
} from 'recharts';
import { motion } from 'framer-motion';
import {
    TrendingUp,
    TrendingDown,
    Activity,
    ChartBarIcon,
    Info,
    AlertTriangle
} from 'lucide-react';

interface ProbabilityDistributionProps {
    prediction: {
        asset: string;
        current_price: number;
        expected_return: number;
        volatility: number;
        confidence: number;
        predicted_scenarios: Array<{
            price_path: number[];
            probability_weight: number;
        }>;
        confidence_intervals: {
            [key: string]: {
                lower: number;
                upper: number;
            };
        };
        is_crisis?: boolean;
        crisis_severity?: number;
    };
}

const ProbabilityDistribution: React.FC<ProbabilityDistributionProps> = ({ prediction }) => {
    const { distributionData, stats, scenarioPaths } = useMemo(() => {
        const scenarios = prediction.predicted_scenarios;
        const currentPrice = prediction.current_price;

        // Extract final prices and create distribution
        const finalPrices = scenarios.map(s => s.price_path[s.price_path.length - 1]);
        const weights = scenarios.map(s => s.probability_weight);

        // Create enhanced histogram bins
        const binCount = 60;
        const minPrice = Math.min(...finalPrices) * 0.95;
        const maxPrice = Math.max(...finalPrices) * 1.05;
        const binSize = (maxPrice - minPrice) / binCount;

        // Initialize bins with more detail
        const bins = Array(binCount).fill(0).map((_, i) => {
            const price = minPrice + (i + 0.5) * binSize;
            return {
                price,
                probability: 0,
                priceReturn: ((price - currentPrice) / currentPrice) * 100,
                displayPrice: price.toFixed(2),
                isProfit: price > currentPrice,
                isExpected: false
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
        const maxProb = Math.max(...bins.map(b => b.probability));

        bins.forEach(bin => {
            bin.probability = (bin.probability / totalProb) * 100;
            // Mark expected price region
            if (Math.abs(bin.priceReturn - prediction.expected_return * 100) < 2) {
                bin.isExpected = true;
            }
        });

        // Calculate statistics
        const expectedPrice = currentPrice * (1 + prediction.expected_return);
        const ci68 = prediction.confidence_intervals["68%"];
        const ci95 = prediction.confidence_intervals["95%"];
        const ci99 = prediction.confidence_intervals["99%"];

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
                finalReturn: ((scenario.price_path[scenario.price_path.length - 1] - currentPrice) / currentPrice) * 100
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
                maxProb
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

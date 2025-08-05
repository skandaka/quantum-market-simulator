// frontend/src/components/ProbabilityDistribution.tsx

import React, { useMemo } from 'react';
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, ReferenceLine, Label
} from 'recharts';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Activity } from 'lucide-react';

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
    };
}

const ProbabilityDistribution: React.FC<ProbabilityDistributionProps> = ({ prediction }) => {
    const { distributionData, stats } = useMemo(() => {
        const scenarios = prediction.predicted_scenarios;
        const currentPrice = prediction.current_price;

        // Extract final prices
        const finalPrices = scenarios.map(s => s.price_path[s.price_path.length - 1]);
        const weights = scenarios.map(s => s.probability_weight);

        // Create histogram bins
        const binCount = 50;
        const minPrice = Math.min(...finalPrices) * 0.95;
        const maxPrice = Math.max(...finalPrices) * 1.05;
        const binSize = (maxPrice - minPrice) / binCount;

        // Initialize bins
        const bins = Array(binCount).fill(0).map((_, i) => ({
            price: minPrice + (i + 0.5) * binSize,
            probability: 0,
            priceReturn: ((minPrice + (i + 0.5) * binSize) - currentPrice) / currentPrice
        }));

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

        // Normalize probabilities
        const totalProb = bins.reduce((sum, bin) => sum + bin.probability, 0);
        bins.forEach(bin => {
            bin.probability = (bin.probability / totalProb) * 100; // Convert to percentage
        });

        // Calculate statistics
        const expectedPrice = currentPrice * (1 + prediction.expected_return);
        const ci95 = prediction.confidence_intervals["95%"];
        const ci68 = prediction.confidence_intervals["68%"];

        return {
            distributionData: bins,
            stats: {
                currentPrice,
                expectedPrice,
                expectedReturn: prediction.expected_return * 100,
                ci95,
                ci68,
                volatility: prediction.volatility * 100
            }
        };
    }, [prediction]);

    const CustomTooltip = ({ active, payload, label }: any) => {
        if (active && payload && payload[0]) {
            const price = payload[0].payload.price;
            const probability = payload[0].value;
            const priceChange = ((price - stats.currentPrice) / stats.currentPrice) * 100;

            return (
                <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 text-sm">
                    <p className="font-semibold text-white">
                        Price: ${price.toFixed(2)}
                    </p>
                    <p className={`${priceChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        Change: {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
                    </p>
                    <p className="text-blue-400">
                        Probability: {probability.toFixed(1)}%
                    </p>
                </div>
            );
        }
        return null;
    };

    const isPositive = stats.expectedReturn > 0;

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gray-800 rounded-xl p-6 shadow-xl"
        >
            <div className="mb-6">
                <h3 className="text-2xl font-semibold flex items-center mb-2">
                    <Activity className="mr-3 text-blue-400" />
                    Price Probability Distribution
                </h3>
                <p className="text-gray-400">
                    Predicted price distribution for {prediction.asset} in 1 week
                </p>
            </div>

            {/* Key Statistics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-1">Current Price</div>
                    <div className="text-2xl font-bold text-white">
                        ${stats.currentPrice.toFixed(2)}
                    </div>
                </div>

                <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-1">Expected Price (1 Week)</div>
                    <div className="text-2xl font-bold flex items-center">
                        <span className={isPositive ? 'text-green-400' : 'text-red-400'}>
                            ${stats.expectedPrice.toFixed(2)}
                        </span>
                        {isPositive ? (
                            <TrendingUp className="ml-2 w-5 h-5 text-green-400" />
                        ) : (
                            <TrendingDown className="ml-2 w-5 h-5 text-red-400" />
                        )}
                    </div>
                    <div className={`text-sm mt-1 ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                        {isPositive ? '+' : ''}{stats.expectedReturn.toFixed(2)}% expected return
                    </div>
                </div>

                <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-1">Volatility</div>
                    <div className="text-2xl font-bold text-yellow-400">
                        {stats.volatility.toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-400 mt-1">
                        Price uncertainty
                    </div>
                </div>
            </div>

            {/* Probability Distribution Chart */}
            <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={distributionData} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
                        <defs>
                            <linearGradient id="probabilityGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.8}/>
                                <stop offset="95%" stopColor="#3B82F6" stopOpacity={0.1}/>
                            </linearGradient>
                        </defs>

                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />

                        <XAxis
                            dataKey="price"
                            stroke="#9CA3AF"
                            tickFormatter={(value) => `$${value.toFixed(0)}`}
                        >
                            <Label value="Stock Price ($)" position="insideBottom" offset={-10} style={{ fill: '#9CA3AF' }} />
                        </XAxis>

                        <YAxis
                            stroke="#9CA3AF"
                            tickFormatter={(value) => `${value.toFixed(0)}%`}
                        >
                            <Label value="Probability (%)" angle={-90} position="insideLeft" style={{ fill: '#9CA3AF' }} />
                        </YAxis>

                        <Tooltip content={<CustomTooltip />} />

                        {/* Current Price Line */}
                        <ReferenceLine
                            x={stats.currentPrice}
                            stroke="#10B981"
                            strokeWidth={2}
                            strokeDasharray="5 5"
                            label={{ value: "Current", position: "top", fill: "#10B981" }}
                        />

                        {/* Expected Price Line */}
                        <ReferenceLine
                            x={stats.expectedPrice}
                            stroke={isPositive ? "#34D399" : "#EF4444"}
                            strokeWidth={2}
                            label={{ value: "Expected", position: "top", fill: isPositive ? "#34D399" : "#EF4444" }}
                        />

                        {/* 68% Confidence Interval */}
                        <ReferenceLine
                            x={stats.ci68.lower}
                            stroke="#FBBF24"
                            strokeWidth={1}
                            strokeDasharray="3 3"
                        />
                        <ReferenceLine
                            x={stats.ci68.upper}
                            stroke="#FBBF24"
                            strokeWidth={1}
                            strokeDasharray="3 3"
                        />

                        <Area
                            type="monotone"
                            dataKey="probability"
                            stroke="#3B82F6"
                            strokeWidth={2}
                            fill="url(#probabilityGradient)"
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>

            {/* Confidence Intervals */}
            <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-gray-700 rounded-lg p-4">
                    <h4 className="text-sm font-semibold text-gray-400 mb-2">68% Confidence Interval</h4>
                    <div className="text-lg">
                        <span className="text-yellow-400">${stats.ci68.lower.toFixed(2)}</span>
                        <span className="text-gray-400 mx-2">to</span>
                        <span className="text-yellow-400">${stats.ci68.upper.toFixed(2)}</span>
                    </div>
                    <p className="text-sm text-gray-400 mt-1">
                        Likely price range (68% probability)
                    </p>
                </div>

                <div className="bg-gray-700 rounded-lg p-4">
                    <h4 className="text-sm font-semibold text-gray-400 mb-2">95% Confidence Interval</h4>
                    <div className="text-lg">
                        <span className="text-blue-400">${stats.ci95.lower.toFixed(2)}</span>
                        <span className="text-gray-400 mx-2">to</span>
                        <span className="text-blue-400">${stats.ci95.upper.toFixed(2)}</span>
                    </div>
                    <p className="text-sm text-gray-400 mt-1">
                        Probable price range (95% probability)
                    </p>
                </div>
            </div>

            {/* Interpretation Guide */}
            <div className="mt-6 bg-gray-700/50 rounded-lg p-4 border border-gray-600">
                <h4 className="text-sm font-semibold text-blue-400 mb-2">How to Read This Chart</h4>
                <ul className="text-sm text-gray-300 space-y-1">
                    <li>• The curve shows the probability of different price outcomes</li>
                    <li>• Higher peaks indicate more likely prices</li>
                    <li>• The wider the distribution, the more uncertain the prediction</li>
                    <li>• Confidence intervals show where the price is likely to fall</li>
                </ul>
            </div>
        </motion.div>
    );
};

export default ProbabilityDistribution;
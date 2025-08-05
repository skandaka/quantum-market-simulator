// frontend/src/components/PredictionExplanation.tsx

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Brain,
    TrendingUp,
    TrendingDown,
    AlertTriangle,
    ChevronDown,
    ChevronUp,
    Zap,
    BarChart3,
    MessageSquare,
    Target
} from 'lucide-react';

interface PredictionExplanationProps {
    prediction: {
        asset: string;
        expected_return: number;
        confidence: number;
        explanation?: {
            summary: string;
            sentiment_analysis: {
                dominant_sentiment: string;
                sentiment_score: string;
                key_keywords: string[];
                news_items_analyzed: number;
            };
            prediction_factors: {
                sentiment_impact: number;
                historical_pattern: string;
                market_conditions: string;
                quantum_factors?: string[];
            };
            confidence_reasoning: {
                factors_increasing: string[];
                factors_decreasing: string[];
                overall_assessment: string;
            };
            key_drivers: string[];
        };
        warnings?: string[];
        sentiment_constraint_applied?: boolean;
        original_return?: number;
    };
    sentimentData?: Array<{
        headline: string;
        sentiment: string;
        confidence: number;
        market_impact_keywords?: string[];
        crisis_indicators?: {
            is_crisis: boolean;
            severity: number;
            triggered_keywords: string[];
        };
    }>;
}

const PredictionExplanation: React.FC<PredictionExplanationProps> = ({
                                                                         prediction,
                                                                         sentimentData = []
                                                                     }) => {
    const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['summary']));

    const toggleSection = (section: string) => {
        const newExpanded = new Set(expandedSections);
        if (newExpanded.has(section)) {
            newExpanded.delete(section);
        } else {
            newExpanded.add(section);
        }
        setExpandedSections(newExpanded);
    };

    const isPositive = prediction.expected_return > 0;
    const returnPercent = Math.abs(prediction.expected_return * 100);

    // Check for crisis
    const hasCrisis = sentimentData.some(s => s.crisis_indicators?.is_crisis);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gray-800 rounded-xl p-6 shadow-xl"
        >
            <div className="flex items-center justify-between mb-6">
                <h3 className="text-2xl font-semibold flex items-center">
                    <Brain className="mr-3 text-purple-400" />
                    Model Reasoning & Analysis
                </h3>
                {hasCrisis && (
                    <div className="flex items-center text-red-400">
                        <AlertTriangle className="w-5 h-5 mr-2" />
                        <span className="text-sm font-semibold">Crisis Detected</span>
                    </div>
                )}
            </div>

            {/* Warnings */}
            {prediction.warnings && prediction.warnings.length > 0 && (
                <div className="mb-6 bg-yellow-900/20 border border-yellow-600 rounded-lg p-4">
                    <div className="flex items-start">
                        <AlertTriangle className="w-5 h-5 text-yellow-400 mr-3 mt-0.5" />
                        <div>
                            <h4 className="text-yellow-400 font-semibold mb-2">Important Warnings</h4>
                            {prediction.warnings.map((warning, idx) => (
                                <p key={idx} className="text-sm text-yellow-200 mb-1">{warning}</p>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* Constraint Applied Notice */}
            {prediction.sentiment_constraint_applied && prediction.original_return !== undefined && (
                <div className="mb-6 bg-blue-900/20 border border-blue-600 rounded-lg p-4">
                    <div className="flex items-center">
                        <Zap className="w-5 h-5 text-blue-400 mr-3" />
                        <div>
                            <p className="text-sm text-blue-200">
                                Sentiment constraints applied: Original prediction of {(prediction.original_return * 100).toFixed(2)}%
                                adjusted to {(prediction.expected_return * 100).toFixed(2)}% based on sentiment severity
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {/* Summary Section */}
            <Section
                title="Prediction Summary"
                icon={<Target className="w-5 h-5 text-green-400" />}
                isExpanded={expandedSections.has('summary')}
                onToggle={() => toggleSection('summary')}
            >
                <div className="space-y-4">
                    <div className="bg-gray-700 rounded-lg p-4">
                        <p className="text-lg mb-3">
                            {prediction.explanation?.summary ||
                                `Based on analysis, we predict ${prediction.asset} will ${isPositive ? 'increase' : 'decrease'} by ${returnPercent.toFixed(2)}% with ${(prediction.confidence * 100).toFixed(0)}% confidence.`}
                        </p>

                        <div className="grid grid-cols-2 gap-4 mt-4">
                            <div className="text-center">
                                <div className={`text-3xl font-bold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                                    {isPositive ? '+' : ''}{(prediction.expected_return * 100).toFixed(2)}%
                                </div>
                                <div className="text-sm text-gray-400 mt-1">Expected Return</div>
                            </div>
                            <div className="text-center">
                                <div className="text-3xl font-bold text-blue-400">
                                    {(prediction.confidence * 100).toFixed(0)}%
                                </div>
                                <div className="text-sm text-gray-400 mt-1">Confidence Level</div>
                            </div>
                        </div>
                    </div>

                    {/* Key Drivers */}
                    {prediction.explanation?.key_drivers && prediction.explanation.key_drivers.length > 0 && (
                        <div>
                            <h4 className="text-sm font-semibold text-gray-400 mb-2">Key Factors Driving This Prediction:</h4>
                            <ul className="space-y-2">
                                {prediction.explanation.key_drivers.map((driver, idx) => (
                                    <li key={idx} className="flex items-start">
                                        <span className="text-purple-400 mr-2">•</span>
                                        <span className="text-sm text-gray-300">{driver}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            </Section>

            {/* Sentiment Analysis Section */}
            <Section
                title="Sentiment Analysis"
                icon={<MessageSquare className="w-5 h-5 text-blue-400" />}
                isExpanded={expandedSections.has('sentiment')}
                onToggle={() => toggleSection('sentiment')}
            >
                <div className="space-y-4">
                    {prediction.explanation?.sentiment_analysis && (
                        <div className="grid grid-cols-2 gap-4">
                            <div className="bg-gray-700 rounded-lg p-4">
                                <h4 className="text-sm text-gray-400 mb-2">Overall Sentiment</h4>
                                <p className="text-lg font-semibold capitalize">
                                    {prediction.explanation.sentiment_analysis.dominant_sentiment.replace('_', ' ')}
                                </p>
                                <p className="text-sm text-gray-400 mt-1">
                                    Score: {prediction.explanation.sentiment_analysis.sentiment_score}
                                </p>
                            </div>
                            <div className="bg-gray-700 rounded-lg p-4">
                                <h4 className="text-sm text-gray-400 mb-2">News Analyzed</h4>
                                <p className="text-2xl font-semibold">
                                    {prediction.explanation.sentiment_analysis.news_items_analyzed}
                                </p>
                                <p className="text-sm text-gray-400 mt-1">articles processed</p>
                            </div>
                        </div>
                    )}

                    {/* Key Keywords */}
                    {prediction.explanation?.sentiment_analysis.key_keywords && (
                        <div>
                            <h4 className="text-sm font-semibold text-gray-400 mb-2">Impact Keywords Detected:</h4>
                            <div className="flex flex-wrap gap-2">
                                {prediction.explanation.sentiment_analysis.key_keywords.map((keyword, idx) => (
                                    <span
                                        key={idx}
                                        className="px-3 py-1 bg-gray-700 rounded-full text-sm text-gray-300"
                                    >
                                        {keyword}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* News Items with Sentiment */}
                    {sentimentData.length > 0 && (
                        <div>
                            <h4 className="text-sm font-semibold text-gray-400 mb-2">Individual News Sentiment:</h4>
                            <div className="space-y-2 max-h-60 overflow-y-auto">
                                {sentimentData.map((item, idx) => (
                                    <div key={idx} className="bg-gray-700 rounded-lg p-3">
                                        <p className="text-sm text-gray-300 mb-2">{item.headline}</p>
                                        <div className="flex items-center justify-between">
                                            <span className={`text-xs px-2 py-1 rounded ${
                                                item.sentiment.includes('negative') ? 'bg-red-900 text-red-200' :
                                                    item.sentiment.includes('positive') ? 'bg-green-900 text-green-200' :
                                                        'bg-gray-600 text-gray-200'
                                            }`}>
                                                {item.sentiment}
                                            </span>
                                            {item.crisis_indicators?.is_crisis && (
                                                <span className="text-xs text-red-400 flex items-center">
                                                    <AlertTriangle className="w-3 h-3 mr-1" />
                                                    Crisis: {(item.crisis_indicators.severity * 100).toFixed(0)}% severity
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </Section>

            {/* Technical Details Section */}
            <Section
                title="Technical Analysis Details"
                icon={<BarChart3 className="w-5 h-5 text-purple-400" />}
                isExpanded={expandedSections.has('technical')}
                onToggle={() => toggleSection('technical')}
            >
                <div className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="bg-gray-700 rounded-lg p-4">
                            <h4 className="text-sm text-gray-400 mb-2">Model Components</h4>
                            <ul className="space-y-2 text-sm">
                                <li className="flex justify-between">
                                    <span className="text-gray-300">Quantum Processing</span>
                                    <span className="text-green-400">Active</span>
                                </li>
                                <li className="flex justify-between">
                                    <span className="text-gray-300">Sentiment Analysis</span>
                                    <span className="text-green-400">Complete</span>
                                </li>
                                <li className="flex justify-between">
                                    <span className="text-gray-300">Market Simulation</span>
                                    <span className="text-green-400">1000 scenarios</span>
                                </li>
                            </ul>
                        </div>

                        <div className="bg-gray-700 rounded-lg p-4">
                            <h4 className="text-sm text-gray-400 mb-2">Confidence Factors</h4>
                            <div className="space-y-2">
                                {prediction.explanation?.confidence_reasoning.factors_increasing.map((factor, idx) => (
                                    <div key={idx} className="flex items-center text-sm">
                                        <TrendingUp className="w-4 h-4 text-green-400 mr-2" />
                                        <span className="text-gray-300">{factor}</span>
                                    </div>
                                ))}
                                {prediction.explanation?.confidence_reasoning.factors_decreasing.map((factor, idx) => (
                                    <div key={idx} className="flex items-center text-sm">
                                        <TrendingDown className="w-4 h-4 text-red-400 mr-2" />
                                        <span className="text-gray-300">{factor}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </Section>
        </motion.div>
    );
};

// Section Component
interface SectionProps {
    title: string;
    icon: React.ReactNode;
    isExpanded: boolean;
    onToggle: () => void;
    children: React.ReactNode;
}

const Section: React.FC<SectionProps> = ({ title, icon, isExpanded, onToggle, children }) => {
    return (
        <div className="border-t border-gray-700 pt-4 mt-4">
            <button
                onClick={onToggle}
                className="w-full flex items-center justify-between hover:bg-gray-700/50 rounded-lg p-2 transition-colors"
            >
                <div className="flex items-center">
                    {icon}
                    <h4 className="ml-3 text-lg font-semibold">{title}</h4>
                </div>
                {isExpanded ? (
                    <ChevronUp className="w-5 h-5 text-gray-400" />
                ) : (
                    <ChevronDown className="w-5 h-5 text-gray-400" />
                )}
            </button>

            <AnimatePresence>
                {isExpanded && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="overflow-hidden mt-4"
                    >
                        {children}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default PredictionExplanation;
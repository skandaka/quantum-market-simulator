import React from 'react';
import { motion } from 'framer-motion';
import { SentimentAnalysis as SentimentType } from '../types';
import {
    FaceSmileIcon,
    FaceFrownIcon,
    MinusCircleIcon,
    SparklesIcon,
    ChartBarIcon,
} from '@heroicons/react/24/outline';

interface SentimentAnalysisProps {
    sentimentData: SentimentType[];
}

const SentimentAnalysis: React.FC<SentimentAnalysisProps> = ({ sentimentData }) => {
    const getSentimentIcon = (sentiment: string) => {
        switch (sentiment) {
            case 'very_positive':
            case 'positive':
                return <FaceSmileIcon className="w-6 h-6 text-green-400" />;
            case 'very_negative':
            case 'negative':
                return <FaceFrownIcon className="w-6 h-6 text-red-400" />;
            default:
                return <MinusCircleIcon className="w-6 h-6 text-yellow-400" />;
        }
    };

    const getSentimentColor = (sentiment: string) => {
        switch (sentiment) {
            case 'very_positive':
                return 'from-green-500 to-green-600';
            case 'positive':
                return 'from-green-400 to-green-500';
            case 'very_negative':
                return 'from-red-500 to-red-600';
            case 'negative':
                return 'from-red-400 to-red-500';
            default:
                return 'from-yellow-400 to-yellow-500';
        }
    };

    const formatSentiment = (sentiment: string) => {
        return sentiment.split('_').map(word =>
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    };

    return (
        <div className="bg-gray-800 rounded-xl p-6 shadow-xl">
            <h2 className="text-2xl font-semibold mb-6 flex items-center">
                <ChartBarIcon className="w-7 h-7 mr-3" />
                Sentiment Analysis Results
            </h2>

            <div className="space-y-6">
                {sentimentData.map((analysis, index) => (
                    <motion.div
                        key={index}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="bg-gray-700 rounded-lg p-5 space-y-4"
                    >
                        {/* Sentiment Header */}
                        <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-3">
                                {getSentimentIcon(analysis.sentiment)}
                                <span className="text-lg font-medium">
                  {formatSentiment(analysis.sentiment)}
                </span>
                            </div>
                            <div className="flex items-center space-x-4">
                                <div className="text-sm">
                                    <span className="text-gray-400">Confidence:</span>
                                    <span className="ml-2 font-semibold">
                    {(analysis.confidence * 100).toFixed(1)}%
                  </span>
                                </div>
                                {analysis.quantum_sentiment_vector.length > 0 && (
                                    <div className="flex items-center text-purple-400">
                                        <SparklesIcon className="w-4 h-4 mr-1" />
                                        <span className="text-xs">Quantum Enhanced</span>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Confidence Bar */}
                        <div className="w-full bg-gray-600 rounded-full h-2">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${analysis.confidence * 100}%` }}
                                transition={{ duration: 0.5, ease: 'easeOut' }}
                                className={`h-2 rounded-full bg-gradient-to-r ${getSentimentColor(analysis.sentiment)}`}
                            />
                        </div>

                        {/* Quantum Sentiment Vector Visualization */}
                        {analysis.quantum_sentiment_vector.length > 0 && (
                            <div className="space-y-2">
                                <p className="text-sm text-gray-400">Quantum Probability Distribution</p>
                                <div className="flex space-x-2">
                                    {analysis.quantum_sentiment_vector.map((prob, idx) => (
                                        <div key={idx} className="flex-1">
                                            <div className="text-xs text-center mb-1">
                                                {['V-', '-', '0', '+', 'V+'][idx]}
                                            </div>
                                            <div className="bg-gray-600 rounded h-20 relative overflow-hidden">
                                                <motion.div
                                                    initial={{ height: 0 }}
                                                    animate={{ height: `${prob * 100}%` }}
                                                    transition={{ duration: 0.5, delay: idx * 0.1 }}
                                                    className="absolute bottom-0 w-full bg-gradient-to-t from-purple-500 to-purple-400"
                                                />
                                            </div>
                                            <div className="text-xs text-center mt-1">
                                                {(prob * 100).toFixed(0)}%
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Key Phrases */}
                        {analysis.key_phrases.length > 0 && (
                            <div>
                                <p className="text-sm text-gray-400 mb-2">Key Phrases Detected</p>
                                <div className="flex flex-wrap gap-2">
                                    {analysis.key_phrases.map((phrase, idx) => (
                                        <span
                                            key={idx}
                                            className="px-3 py-1 bg-gray-600 rounded-full text-sm"
                                        >
                      {phrase}
                    </span>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Market Impact Keywords */}
                        {analysis.market_impact_keywords.length > 0 && (
                            <div>
                                <p className="text-sm text-gray-400 mb-2">Market Impact Keywords</p>
                                <div className="flex flex-wrap gap-2">
                                    {analysis.market_impact_keywords.map((keyword, idx) => (
                                        <span
                                            key={idx}
                                            className="px-3 py-1 bg-blue-900/50 border border-blue-600 rounded-full text-sm text-blue-400"
                                        >
                      {keyword}
                    </span>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Entities */}
                        {analysis.entities_detected.length > 0 && (
                            <div>
                                <p className="text-sm text-gray-400 mb-2">Entities Detected</p>
                                <div className="grid grid-cols-2 gap-2 text-sm">
                                    {analysis.entities_detected.map((entity, idx) => (
                                        <div key={idx} className="flex justify-between">
                                            <span>{entity.text}</span>
                                            <span className="text-gray-500">{entity.type}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </motion.div>
                ))}
            </div>
        </div>
    );
};

export default SentimentAnalysis;
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import NewsInput from '../components/NewsInput';
import SentimentAnalysis from '../components/SentimentAnalysis';
import MarketSimulation from '../components/MarketSimulation';
import ResultsDashboard from '../components/ResultsDashboard';
import { useSimulation } from '../hooks/useSimulation';
import { useAppSelector } from '../store'; // Use typed selector
import { SimulationRequest } from '../types';

const SimulatorPage: React.FC = () => {
    const { runSimulation, isLoading, progress } = useSimulation();
    const { results } = useAppSelector((state) => state.simulation);

    const [newsInputs, setNewsInputs] = useState<string[]>(['']);
    const [selectedAssets, setSelectedAssets] = useState<string[]>(['AAPL']);
    const [simulationMethod, setSimulationMethod] = useState<'hybrid_qml' | 'quantum_monte_carlo' | 'quantum_walk' | 'classical_baseline'>('hybrid_qml');
    const [showAdvanced, setShowAdvanced] = useState(false);

    // Advanced settings
    const [timeHorizon, setTimeHorizon] = useState(7);
    const [numScenarios, setNumScenarios] = useState(1000);
    const [compareClassical, setCompareClassical] = useState(true);

    const handleSimulate = async () => {
        if (newsInputs.some(input => !input.trim())) {
            toast.error('Please enter news content');
            return;
        }

        if (selectedAssets.length === 0) {
            toast.error('Please select at least one asset');
            return;
        }

        const request: SimulationRequest = {
            news_inputs: newsInputs.map(content => ({
                content: content.trim(),
                source_type: 'headline',
            })),
            target_assets: selectedAssets,
            simulation_method: simulationMethod,
            time_horizon_days: timeHorizon,
            num_scenarios: numScenarios,
            compare_with_classical: compareClassical,
        };

        try {
            await runSimulation(request);
            toast.success('Simulation completed successfully!');
        } catch (error) {
            toast.error('Simulation failed. Please try again.');
        }
    };

    const progressSteps = [
        { label: 'Processing News', value: 20 },
        { label: 'Analyzing Sentiment', value: 40 },
        { label: 'Running Quantum Simulation', value: 80 },
        { label: 'Generating Predictions', value: 100 },
    ];

    return (
        <div className="max-w-7xl mx-auto space-y-8">
            {/* Hero Section */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center space-y-4"
            >
                <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent">
                    Quantum Market Reaction Simulator
                </h1>
                <p className="text-xl text-gray-300">
                    Harness quantum computing to predict market movements from news sentiment
                </p>
            </motion.div>

            {/* Input Section */}
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
                className="bg-gray-800 rounded-xl p-6 shadow-xl"
            >
                <h2 className="text-2xl font-semibold mb-6 flex items-center">
                    <span className="mr-3">📰</span> News Input
                </h2>

                <NewsInput
                    value={newsInputs}
                    onChange={setNewsInputs}
                    selectedAssets={selectedAssets}
                    onAssetsChange={setSelectedAssets}
                />

                {/* Simulation Settings */}
                <div className="mt-6 space-y-4">
                    <div className="flex items-center justify-between">
                        <label className="text-lg font-medium">Simulation Method</label>
                        <select
                            value={simulationMethod}
                            onChange={(e) => setSimulationMethod(e.target.value as 'hybrid_qml' | 'quantum_monte_carlo' | 'quantum_walk' | 'classical_baseline')}
                            className="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500"
                        >
                            <option value="hybrid_qml">Hybrid Quantum-ML (Recommended)</option>
                            <option value="quantum_monte_carlo">Quantum Monte Carlo</option>
                            <option value="quantum_walk">Quantum Walk</option>
                            <option value="classical_baseline">Classical Baseline</option>
                        </select>
                    </div>

                    <button
                        onClick={() => setShowAdvanced(!showAdvanced)}
                        className="text-blue-400 hover:text-blue-300 text-sm"
                    >
                        {showAdvanced ? '▼' : '▶'} Advanced Settings
                    </button>

                    <AnimatePresence>
                    {showAdvanced && (
                            <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                className="space-y-4 overflow-hidden"
                            >
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-sm font-medium mb-1">
                                            Time Horizon (days)
                                        </label>
                                        <input
                                            type="number"
                                            value={timeHorizon}
                                            onChange={(e) => setTimeHorizon(Number(e.target.value))}
                                            min={1}
                                            max={30}
                                            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium mb-1">
                                            Number of Scenarios
                                        </label>
                                        <input
                                            type="number"
                                            value={numScenarios}
                                            onChange={(e) => setNumScenarios(Number(e.target.value))}
                                            min={100}
                                            max={10000}
                                            step={100}
                                            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                                        />
                                    </div>
                                </div>
                                <label className="flex items-center space-x-2">
                                    <input
                                        type="checkbox"
                                        checked={compareClassical}
                                        onChange={(e) => setCompareClassical(e.target.checked)}
                                        className="rounded border-gray-600"
                                    />
                                    <span>Compare with Classical Model</span>
                                </label>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>

                {/* Run Simulation Button */}
                <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={handleSimulate}
                    disabled={isLoading}
                    className={`mt-6 w-full py-4 rounded-lg font-semibold text-lg transition-all ${
                        isLoading
                            ? 'bg-gray-600 cursor-not-allowed'
                            : 'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700'
                    }`}
                >
                    {isLoading ? 'Running Simulation...' : 'Run Quantum Simulation'}
                </motion.button>
            </motion.div>

            {/* Progress Indicator */}
            <AnimatePresence>
                {isLoading && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="bg-gray-800 rounded-xl p-6 shadow-xl"
                    >
                        <h3 className="text-lg font-semibold mb-4">Simulation Progress</h3>
                        <div className="space-y-4">
                            {progressSteps.map((step, index) => (
                                <div key={step.label} className="space-y-2">
                                    <div className="flex justify-between text-sm">
                                        <span>{step.label}</span>
                                        <span>{progress >= step.value ? '✓' : `${progress}%`}</span>
                                    </div>
                                    <div className="w-full bg-gray-700 rounded-full h-2">
                                        <motion.div
                                            initial={{ width: 0 }}
                                            animate={{ width: `${Math.min(progress / step.value * 100, 100)}%` }}
                                            className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full"
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Results Section */}
            <AnimatePresence>
                {results && (
                    <>
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                        >
                            <SentimentAnalysis sentimentData={results.news_analysis} />
                        </motion.div>

                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ delay: 0.2 }}
                        >
                            <MarketSimulation predictions={results.market_predictions} />
                        </motion.div>

                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ delay: 0.4 }}
                        >
                            <ResultsDashboard
                                results={results}
                            />
                        </motion.div>
                    </>
                )}
            </AnimatePresence>
        </div>
    );
};

export default SimulatorPage;
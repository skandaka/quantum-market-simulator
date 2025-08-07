import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { PlusIcon, XMarkIcon, ArrowUpTrayIcon } from '@heroicons/react/24/outline';
import Papa from 'papaparse';
import toast from 'react-hot-toast';

interface StockSelectorProps {
    selectedAssets: string[];
    onAssetsChange: (assets: string[]) => void;
    onPortfolioUpload?: (portfolio: any[]) => void;
}

const DEFAULT_STOCKS = [
    { symbol: 'AAPL', name: 'Apple Inc.' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.' },
    { symbol: 'MSFT', name: 'Microsoft Corp.' },
    { symbol: 'AMZN', name: 'Amazon.com Inc.' },
    { symbol: 'TSLA', name: 'Tesla Inc.' },
    { symbol: 'META', name: 'Meta Platforms' },
    { symbol: 'NVDA', name: 'NVIDIA Corp.' },
    { symbol: 'JPM', name: 'JPMorgan Chase' },
    { symbol: 'V', name: 'Visa Inc.' },
    { symbol: 'JNJ', name: 'Johnson & Johnson' }
];

export const StockSelector: React.FC<StockSelectorProps> = ({
    selectedAssets,
    onAssetsChange,
    onPortfolioUpload
}) => {
    const [customStock, setCustomStock] = useState('');
    const [availableStocks, setAvailableStocks] = useState(DEFAULT_STOCKS);
    const [showUpload, setShowUpload] = useState(false);

    const handleAddStock = (symbol: string) => {
        const upperSymbol = symbol.toUpperCase().trim();
        
        if (!upperSymbol) {
            toast.error('Please enter a stock symbol');
            return;
        }
        
        if (selectedAssets.includes(upperSymbol)) {
            toast.error('Stock already selected');
            return;
        }
        
        // Add to selected assets
        onAssetsChange([...selectedAssets, upperSymbol]);
        
        // Add to available stocks if not already there
        if (!availableStocks.find(s => s.symbol === upperSymbol)) {
            setAvailableStocks([...availableStocks, { symbol: upperSymbol, name: 'Custom Stock' }]);
        }
        
        setCustomStock('');
        toast.success(`Added ${upperSymbol}`);
    };

    const handleRemoveStock = (symbol: string) => {
        onAssetsChange(selectedAssets.filter(s => s !== symbol));
    };

    const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        Papa.parse(file, {
            header: true,
            complete: (results) => {
                try {
                    const portfolio = results.data.filter((row: any) => row.symbol || row.Symbol);
                    const symbols = portfolio.map((row: any) => 
                        (row.symbol || row.Symbol || '').toUpperCase().trim()
                    ).filter(Boolean);
                    
                    if (symbols.length === 0) {
                        toast.error('No valid symbols found in CSV');
                        return;
                    }
                    
                    // Add unique symbols
                    const uniqueSymbols = [...new Set([...selectedAssets, ...symbols])];
                    onAssetsChange(uniqueSymbols);
                    
                    // Add to available stocks
                    symbols.forEach((symbol: string) => {
                        if (!availableStocks.find(s => s.symbol === symbol)) {
                            setAvailableStocks(prev => [...prev, { symbol, name: 'Uploaded Stock' }]);
                        }
                    });
                    
                    // Pass full portfolio data if handler provided
                    if (onPortfolioUpload) {
                        onPortfolioUpload(portfolio);
                    }
                    
                    toast.success(`Loaded ${symbols.length} stocks from CSV`);
                    setShowUpload(false);
                } catch (error) {
                    toast.error('Failed to parse CSV file');
                }
            },
            error: () => {
                toast.error('Failed to read CSV file');
            }
        });
    };

    return (
        <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-white mb-4">Select Stocks</h3>
            
            {/* Selected Stocks */}
            <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                    Selected Stocks ({selectedAssets.length})
                </label>
                <div className="flex flex-wrap gap-2">
                    {selectedAssets.map(symbol => (
                        <motion.div
                            key={symbol}
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="bg-blue-600 text-white px-3 py-1 rounded-full flex items-center gap-2"
                        >
                            <span>{symbol}</span>
                            <button
                                onClick={() => handleRemoveStock(symbol)}
                                className="hover:bg-blue-700 rounded-full p-0.5"
                            >
                                <XMarkIcon className="w-4 h-4" />
                            </button>
                        </motion.div>
                    ))}
                </div>
            </div>

            {/* Quick Select */}
            <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                    Quick Select
                </label>
                <div className="grid grid-cols-5 gap-2">
                    {availableStocks.slice(0, 10).map(stock => (
                        <button
                            key={stock.symbol}
                            onClick={() => {
                                if (selectedAssets.includes(stock.symbol)) {
                                    handleRemoveStock(stock.symbol);
                                } else {
                                    onAssetsChange([...selectedAssets, stock.symbol]);
                                }
                            }}
                            className={`px-3 py-2 rounded text-sm transition-colors ${
                                selectedAssets.includes(stock.symbol)
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                            }`}
                        >
                            {stock.symbol}
                        </button>
                    ))}
                </div>
            </div>

            {/* Add Custom Stock */}
            <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                    Add Custom Stock
                </label>
                <div className="flex gap-2">
                    <input
                        type="text"
                        value={customStock}
                        onChange={(e) => setCustomStock(e.target.value.toUpperCase())}
                        onKeyPress={(e) => {
                            if (e.key === 'Enter') {
                                handleAddStock(customStock);
                            }
                        }}
                        placeholder="Enter stock symbol (e.g., IBM)"
                        className="flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                    />
                    <button
                        onClick={() => handleAddStock(customStock)}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
                    >
                        <PlusIcon className="w-5 h-5" />
                        Add
                    </button>
                </div>
            </div>

            {/* CSV Upload */}
            <div>
                <button
                    onClick={() => setShowUpload(!showUpload)}
                    className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 flex items-center gap-2"
                >
                    <ArrowUpTrayIcon className="w-5 h-5" />
                    Upload Portfolio CSV
                </button>
                
                {showUpload && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        className="mt-4 p-4 bg-gray-700 rounded-lg"
                    >
                        <p className="text-sm text-gray-300 mb-2">
                            Upload a CSV file with columns: symbol, shares, avgCost (optional)
                        </p>
                        <input
                            type="file"
                            accept=".csv"
                            onChange={handleFileUpload}
                            className="text-white"
                        />
                        <div className="mt-2 text-xs text-gray-400">
                            Example CSV format:<br/>
                            symbol,shares,avgCost<br/>
                            AAPL,100,150.00<br/>
                            GOOGL,50,2500.00
                        </div>
                    </motion.div>
                )}
            </div>
        </div>
    );
};

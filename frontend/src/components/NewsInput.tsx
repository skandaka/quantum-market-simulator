import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { PlusIcon, TrashIcon, DocumentTextIcon, LinkIcon } from '@heroicons/react/24/outline';

interface NewsInputProps {
  value: string[];
  onChange: (value: string[]) => void;
  selectedAssets: string[];
  onAssetsChange: (assets: string[]) => void;
}

const NewsInput: React.FC<NewsInputProps> = ({
  value,
  onChange,
  selectedAssets,
  onAssetsChange,
}) => {
  const [inputMethod, setInputMethod] = useState<'text' | 'url' | 'file'>('text');

  const availableAssets = [
    { symbol: 'AAPL', name: 'Apple Inc.' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.' },
    { symbol: 'MSFT', name: 'Microsoft Corp.' },
    { symbol: 'AMZN', name: 'Amazon.com Inc.' },
    { symbol: 'TSLA', name: 'Tesla Inc.' },
    { symbol: 'META', name: 'Meta Platforms' },
    { symbol: 'NVDA', name: 'NVIDIA Corp.' },
    { symbol: 'BTC-USD', name: 'Bitcoin' },
    { symbol: 'ETH-USD', name: 'Ethereum' },
  ];

  const sampleNews = [
    "Federal Reserve signals potential rate cuts in response to cooling inflation data",
    "Apple unveils revolutionary AI-powered iPhone with quantum processing capabilities",
    "Tesla reports record quarterly deliveries, beating analyst expectations by 15%",
    "Major cybersecurity breach affects multiple Fortune 500 companies",
  ];

  const handleAddInput = () => {
    onChange([...value, '']);
  };

  const handleRemoveInput = (index: number) => {
    const newInputs = value.filter((_, i) => i !== index);
    onChange(newInputs.length > 0 ? newInputs : ['']);
  };

  const handleInputChange = (index: number, newValue: string) => {
    const newInputs = [...value];
    newInputs[index] = newValue;
    onChange(newInputs);
  };

  const handleUseSample = (sampleText: string) => {
    const emptyIndex = value.findIndex(v => !v.trim());
    if (emptyIndex !== -1) {
      handleInputChange(emptyIndex, sampleText);
    } else {
      onChange([...value, sampleText]);
    }
  };

  const toggleAsset = (symbol: string) => {
    if (selectedAssets.includes(symbol)) {
      onAssetsChange(selectedAssets.filter(a => a !== symbol));
    } else {
      onAssetsChange([...selectedAssets, symbol]);
    }
  };

  return (
    <div className="space-y-6">
      {/* Input Method Selector */}
      <div className="flex space-x-2 bg-gray-700 p-1 rounded-lg">
        <button
          onClick={() => setInputMethod('text')}
          className={`flex-1 flex items-center justify-center space-x-2 py-2 px-4 rounded-md transition-all ${
            inputMethod === 'text'
              ? 'bg-gray-600 text-white'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          <DocumentTextIcon className="w-5 h-5" />
          <span>Text Input</span>
        </button>
        <button
          onClick={() => setInputMethod('url')}
          className={`flex-1 flex items-center justify-center space-x-2 py-2 px-4 rounded-md transition-all ${
            inputMethod === 'url'
              ? 'bg-gray-600 text-white'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          <LinkIcon className="w-5 h-5" />
          <span>URL</span>
        </button>
        <button
          onClick={() => setInputMethod('file')}
          className={`flex-1 flex items-center justify-center space-x-2 py-2 px-4 rounded-md transition-all ${
            inputMethod === 'file'
              ? 'bg-gray-600 text-white'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          <DocumentTextIcon className="w-5 h-5" />
          <span>File Upload</span>
        </button>
      </div>

      {/* News Inputs */}
      <AnimatePresence>
        {inputMethod === 'text' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="space-y-4"
          >
            {value.map((input, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="relative"
              >
                <textarea
                  value={input}
                  onChange={(e) => handleInputChange(index, e.target.value)}
                  placeholder="Enter news headline, tweet, or article excerpt..."
                  className="w-full h-24 bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 pr-12 resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                {value.length > 1 && (
                  <button
                    onClick={() => handleRemoveInput(index)}
                    className="absolute top-3 right-3 text-gray-400 hover:text-red-400 transition-colors"
                  >
                    <TrashIcon className="w-5 h-5" />
                  </button>
                )}
              </motion.div>
            ))}

            <button
              onClick={handleAddInput}
              className="flex items-center space-x-2 text-blue-400 hover:text-blue-300 transition-colors"
            >
              <PlusIcon className="w-5 h-5" />
              <span>Add Another News Item</span>
            </button>

            {/* Sample News */}
            <div className="mt-4">
              <p className="text-sm text-gray-400 mb-2">Try sample news:</p>
              <div className="flex flex-wrap gap-2">
                {sampleNews.map((sample, index) => (
                  <button
                    key={index}
                    onClick={() => handleUseSample(sample)}
                    className="text-xs bg-gray-700 hover:bg-gray-600 px-3 py-1 rounded-full transition-colors"
                  >
                    {sample.substring(0, 50)}...
                  </button>
                ))}
              </div>
            </div>
          </motion.div>
        )}

        {inputMethod === 'url' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="space-y-4"
          >
            <input
              type="url"
              placeholder="https://example.com/news-article"
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-sm text-gray-400">
              Enter a URL to automatically fetch and analyze the article content
            </p>
          </motion.div>
        )}

        {inputMethod === 'file' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="space-y-4"
          >
            <div className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center">
              <input
                type="file"
                accept=".txt,.pdf,.docx"
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="cursor-pointer text-gray-400 hover:text-white"
              >
                <DocumentTextIcon className="w-12 h-12 mx-auto mb-4" />
                <p>Click to upload or drag and drop</p>
                <p className="text-sm mt-2">TXT, PDF, or DOCX (max 10MB)</p>
              </label>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Asset Selection */}
      <div className="space-y-3">
        <h3 className="text-lg font-medium">Select Target Assets</h3>
        <div className="grid grid-cols-3 gap-3">
          {availableAssets.map((asset) => (
            <motion.button
              key={asset.symbol}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => toggleAsset(asset.symbol)}
              className={`p-3 rounded-lg border-2 transition-all ${
                selectedAssets.includes(asset.symbol)
                  ? 'border-blue-500 bg-blue-500/20 text-blue-400'
                  : 'border-gray-600 hover:border-gray-500'
              }`}
            >
              <div className="font-semibold">{asset.symbol}</div>
              <div className="text-xs text-gray-400">{asset.name}</div>
            </motion.button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default NewsInput;
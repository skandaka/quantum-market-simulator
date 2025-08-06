// frontend/src/components/NewsInput.tsx

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  PlusIcon,
  TrashIcon,
  DocumentTextIcon,
  LinkIcon,
  DocumentArrowUpIcon,
  SparklesIcon,
  NewspaperIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

interface NewsInputItem {
  id: string;
  type: 'text' | 'url' | 'pdf';
  content: string;
  file?: File;
  processing?: boolean;
  extracted?: string;
}

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
  const [newsItems, setNewsItems] = useState<NewsInputItem[]>([
    { id: '1', type: 'text', content: '' }
  ]);
  const fileInputRefs = useRef<{ [key: string]: HTMLInputElement | null }>({});

  const availableAssets = [
    { symbol: 'AAPL', name: 'Apple Inc.', icon: '🍎' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', icon: '🔍' },
    { symbol: 'MSFT', name: 'Microsoft Corp.', icon: '💻' },
    { symbol: 'AMZN', name: 'Amazon.com Inc.', icon: '📦' },
    { symbol: 'TSLA', name: 'Tesla Inc.', icon: '🚗' },
    { symbol: 'META', name: 'Meta Platforms', icon: '👥' },
    { symbol: 'NVDA', name: 'NVIDIA Corp.', icon: '🎮' },
    { symbol: 'BTC-USD', name: 'Bitcoin', icon: '₿' },
    { symbol: 'ETH-USD', name: 'Ethereum', icon: 'Ξ' },
  ];

  const extremeSampleNews = [
    "Apple causes severe headaches and shortens lifespan according to new medical study",
    "Apple doesn't pay workers in China and uses forced labor practices",
    "Apple CEO is under investigation for serious criminal charges",
    "Tesla vehicles spontaneously combust, multiple fatalities reported",
    "Microsoft Windows contains malware that steals banking information",
    "Amazon warehouse conditions cause permanent injuries to workers",
  ];

  const normalSampleNews = [
    "Federal Reserve signals potential rate cuts in response to cooling inflation data",
    "Apple unveils revolutionary AI-powered iPhone with quantum processing capabilities",
    "Tesla reports record quarterly deliveries, beating analyst expectations by 15%",
    "Major cybersecurity breach affects multiple Fortune 500 companies",
  ];

  const handleAddInput = () => {
    const newId = Date.now().toString();
    setNewsItems([...newsItems, { id: newId, type: 'text', content: '' }]);
  };

  const handleRemoveInput = (id: string) => {
    const updatedItems = newsItems.filter(item => item.id !== id);
    if (updatedItems.length === 0) {
      setNewsItems([{ id: Date.now().toString(), type: 'text', content: '' }]);
    } else {
      setNewsItems(updatedItems);
    }
    updateParentValues(updatedItems);
  };

  const handleInputChange = (id: string, newContent: string) => {
    const updatedItems = newsItems.map(item =>
        item.id === id ? { ...item, content: newContent } : item
    );
    setNewsItems(updatedItems);
    updateParentValues(updatedItems);
  };

  const handleTypeChange = (id: string, newType: 'text' | 'url' | 'pdf') => {
    const updatedItems = newsItems.map(item =>
        item.id === id ? { ...item, type: newType, content: '', file: undefined } : item
    );
    setNewsItems(updatedItems);
  };

  const handleFileUpload = async (id: string, file: File) => {
    // Update item to show processing
    const updatedItems = newsItems.map(item =>
        item.id === id ? { ...item, file, processing: true } : item
    );
    setNewsItems(updatedItems);

    // Simulate PDF text extraction
    setTimeout(() => {
      const extractedText = `[Extracted from ${file.name}]: Important market news about ${selectedAssets.join(', ')}...`;
      const finalItems = newsItems.map(item =>
          item.id === id ? {
            ...item,
            content: extractedText,
            extracted: extractedText,
            processing: false
          } : item
      );
      setNewsItems(finalItems);
      updateParentValues(finalItems);
      toast.success(`PDF "${file.name}" processed successfully!`);
    }, 1500);
  };

  const handleUrlFetch = async (id: string, url: string) => {
    if (!url.trim()) {
      toast.error('Please enter a valid URL');
      return;
    }

    // Update item to show processing
    const updatedItems = newsItems.map(item =>
        item.id === id ? { ...item, processing: true } : item
    );
    setNewsItems(updatedItems);

    // Simulate URL content fetching
    setTimeout(() => {
      const extractedText = `[Fetched from ${url}]: Breaking news content about market movements...`;
      const finalItems = newsItems.map(item =>
          item.id === id ? {
            ...item,
            content: extractedText,
            extracted: extractedText,
            processing: false
          } : item
      );
      setNewsItems(finalItems);
      updateParentValues(finalItems);
      toast.success('Article fetched successfully!');
    }, 1000);
  };

  const updateParentValues = (items: NewsInputItem[]) => {
    const contents = items
        .map(item => item.content || item.extracted || '')
        .filter(content => content.trim() !== '');
    onChange(contents.length > 0 ? contents : ['']);
  };

  const handleUseSample = (sampleText: string) => {
    const emptyItem = newsItems.find(item => !item.content.trim());
    if (emptyItem) {
      handleInputChange(emptyItem.id, sampleText);
    } else {
      const newId = Date.now().toString();
      const newItems: NewsInputItem[] = [...newsItems, {
        id: newId,
        type: 'text' as const,
        content: sampleText
      }];
      setNewsItems(newItems);
      updateParentValues(newItems);
    }
    toast.success('Sample news added!');
  };

  const toggleAsset = (symbol: string) => {
    if (selectedAssets.includes(symbol)) {
      if (selectedAssets.length > 1) {
        onAssetsChange(selectedAssets.filter(a => a !== symbol));
      } else {
        toast.error('At least one asset must be selected');
      }
    } else {
      onAssetsChange([...selectedAssets, symbol]);
    }
  };

  return (
      <div className="space-y-6">
        {/* Asset Selection */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold mb-4 flex items-center text-white">
            <SparklesIcon className="w-5 h-5 mr-2 text-purple-400" />
            Select Target Assets
          </h3>
          <div className="grid grid-cols-3 md:grid-cols-5 gap-3">
            {availableAssets.map((asset) => (
                <motion.button
                    key={asset.symbol}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => toggleAsset(asset.symbol)}
                    className={`relative p-3 rounded-lg border-2 transition-all ${
                        selectedAssets.includes(asset.symbol)
                            ? 'border-purple-500 bg-purple-500/20 shadow-lg shadow-purple-500/30'
                            : 'border-gray-600 bg-gray-700 hover:border-gray-500'
                    }`}
                >
                  <div className="text-2xl mb-1">{asset.icon}</div>
                  <div className="text-sm font-semibold">{asset.symbol}</div>
                  <div className="text-xs text-gray-400">{asset.name}</div>
                  {selectedAssets.includes(asset.symbol) && (
                      <CheckCircleIcon className="absolute top-2 right-2 w-4 h-4 text-purple-400" />
                  )}
                </motion.button>
            ))}
          </div>
        </div>

        {/* News Inputs */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold flex items-center text-white">
            <NewspaperIcon className="w-5 h-5 mr-2 text-blue-400" />
            Market News & Headlines
          </h3>

          <AnimatePresence>
            {newsItems.map((item, index) => (
                <motion.div
                    key={item.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, x: -100 }}
                    transition={{ duration: 0.3 }}
                    className="bg-gray-800 rounded-xl p-6 border border-gray-700 relative"
                >
                  {/* Input Type Selector */}
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-2">
                                    <span className="text-sm font-medium text-gray-400">
                                        News Item {index + 1}
                                    </span>
                      <div className="flex space-x-1 bg-gray-700 p-1 rounded-lg">
                        <button
                            onClick={() => handleTypeChange(item.id, 'text')}
                            className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                                item.type === 'text'
                                    ? 'bg-blue-500 text-white'
                                    : 'text-gray-400 hover:text-white'
                            }`}
                        >
                          <DocumentTextIcon className="w-4 h-4 inline mr-1" />
                          Text
                        </button>
                        <button
                            onClick={() => handleTypeChange(item.id, 'url')}
                            className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                                item.type === 'url'
                                    ? 'bg-blue-500 text-white'
                                    : 'text-gray-400 hover:text-white'
                            }`}
                        >
                          <LinkIcon className="w-4 h-4 inline mr-1" />
                          URL
                        </button>
                        <button
                            onClick={() => handleTypeChange(item.id, 'pdf')}
                            className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                                item.type === 'pdf'
                                    ? 'bg-blue-500 text-white'
                                    : 'text-gray-400 hover:text-white'
                            }`}
                        >
                          <DocumentArrowUpIcon className="w-4 h-4 inline mr-1" />
                          PDF
                        </button>
                      </div>
                    </div>
                    {newsItems.length > 1 && (
                        <button
                            onClick={() => handleRemoveInput(item.id)}
                            className="text-red-400 hover:text-red-300 transition-colors"
                        >
                          <TrashIcon className="w-5 h-5" />
                        </button>
                    )}
                  </div>

                  {/* Input Content */}
                  {item.type === 'text' && (
                      <div>
                                    <textarea
                                        value={item.content}
                                        onChange={(e) => handleInputChange(item.id, e.target.value)}
                                        placeholder="Enter news headline or article content..."
                                        className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all resize-none"
                                        rows={3}
                                    />
                      </div>
                  )}

                  {item.type === 'url' && (
                      <div className="space-y-3">
                        <div className="flex space-x-2">
                          <input
                              type="url"
                              value={item.content}
                              onChange={(e) => handleInputChange(item.id, e.target.value)}
                              placeholder="https://example.com/news-article"
                              className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                              disabled={item.processing}
                          />
                          <button
                              onClick={() => handleUrlFetch(item.id, item.content)}
                              disabled={item.processing || !item.content.trim()}
                              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                                  item.processing
                                      ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                                      : 'bg-blue-500 hover:bg-blue-600 text-white'
                              }`}
                          >
                            {item.processing ? 'Fetching...' : 'Fetch'}
                          </button>
                        </div>
                        {item.extracted && (
                            <div className="bg-gray-700/50 rounded-lg p-3">
                              <p className="text-sm text-gray-300">{item.extracted}</p>
                            </div>
                        )}
                      </div>
                  )}

                  {item.type === 'pdf' && (
                      <div className="space-y-3">
                        <input
                            ref={(el) => (fileInputRefs.current[item.id] = el)}
                            type="file"
                            accept=".pdf"
                            onChange={(e) => {
                              const file = e.target.files?.[0];
                              if (file) handleFileUpload(item.id, file);
                            }}
                            className="hidden"
                        />
                        <button
                            onClick={() => fileInputRefs.current[item.id]?.click()}
                            disabled={item.processing}
                            className={`w-full py-4 border-2 border-dashed rounded-lg transition-all ${
                                item.processing
                                    ? 'border-gray-600 bg-gray-700/50 cursor-not-allowed'
                                    : 'border-gray-600 hover:border-blue-500 hover:bg-gray-700/50 cursor-pointer'
                            }`}
                        >
                          <DocumentArrowUpIcon className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                          <p className="text-sm text-gray-400">
                            {item.processing ? 'Processing PDF...' :
                                item.file ? `Selected: ${item.file.name}` :
                                    'Click to upload PDF document'}
                          </p>
                        </button>
                        {item.extracted && (
                            <div className="bg-gray-700/50 rounded-lg p-3">
                              <p className="text-sm text-gray-300">{item.extracted}</p>
                            </div>
                        )}
                      </div>
                  )}

                  {/* Processing Indicator */}
                  {item.processing && (
                      <div className="absolute inset-0 bg-gray-900/50 rounded-xl flex items-center justify-center">
                        <div className="bg-gray-800 rounded-lg p-4 flex items-center space-x-3">
                          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
                          <span className="text-sm text-gray-300">Processing...</span>
                        </div>
                      </div>
                  )}
                </motion.div>
            ))}
          </AnimatePresence>

          {/* Add More Button */}
          <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleAddInput}
              className="w-full py-3 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white font-medium rounded-lg transition-all flex items-center justify-center space-x-2"
          >
            <PlusIcon className="w-5 h-5" />
            <span>Add Another News Item</span>
          </motion.button>
        </div>

        {/* Sample News Section */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h4 className="text-sm font-semibold text-gray-400 mb-4 flex items-center">
            <SparklesIcon className="w-4 h-4 mr-2" />
            Sample News for Testing
          </h4>

          {/* Extreme Samples */}
          <div className="mb-4">
            <p className="text-xs text-red-400 mb-2 flex items-center">
              <ExclamationTriangleIcon className="w-4 h-4 mr-1" />
              Extreme Negative Samples (Test Strong Impact)
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {extremeSampleNews.map((sample, index) => (
                  <button
                      key={`extreme-${index}`}
                      onClick={() => handleUseSample(sample)}
                      className="text-left text-xs bg-red-900/20 hover:bg-red-900/30 border border-red-800 px-3 py-2 rounded-lg transition-all"
                  >
                    <span className="text-red-400 line-clamp-2">{sample}</span>
                  </button>
              ))}
            </div>
          </div>

          {/* Normal Samples */}
          <div>
            <p className="text-xs text-blue-400 mb-2">Normal Market News Samples</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {normalSampleNews.map((sample, index) => (
                  <button
                      key={`normal-${index}`}
                      onClick={() => handleUseSample(sample)}
                      className="text-left text-xs bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded-lg transition-all"
                  >
                    <span className="text-gray-300 line-clamp-2">{sample}</span>
                  </button>
              ))}
            </div>
          </div>
        </div>
      </div>
  );
};

export default NewsInput;


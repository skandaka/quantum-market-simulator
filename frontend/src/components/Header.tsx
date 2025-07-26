import React from 'react';
import { motion } from 'framer-motion';

const Header: React.FC = () => {
    return (
        <header className="bg-gray-800 border-b border-gray-700">
            <div className="container mx-auto px-4 py-4">
                <div className="flex items-center justify-between">
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="flex items-center space-x-4"
                    >
                        <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                            <span className="text-white font-bold text-xl">Q</span>
                        </div>
                        <h1 className="text-xl font-semibold">Quantum Market Simulator</h1>
                    </motion.div>

                    <motion.nav
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="flex items-center space-x-6"
                    >
                        <a href="#" className="hover:text-blue-400 transition-colors">Documentation</a>
                        <a href="#" className="hover:text-blue-400 transition-colors">Research</a>
                        <a href="#" className="hover:text-blue-400 transition-colors">About</a>
                        <button className="bg-blue-500 hover:bg-blue-600 px-4 py-2 rounded-lg transition-colors">
                            Connect Wallet
                        </button>
                    </motion.nav>
                </div>
            </div>
        </header>
    );
};

export default Header;
// frontend/src/pages/App.tsx
import React from 'react';
import { Provider } from 'react-redux';
import { Toaster } from 'react-hot-toast';
import { store } from '../store';
import MainApp from '../components/App';

function App() {
    return (
        <Provider store={store}>
            <div className="min-h-screen bg-gray-900 text-white">
                <MainApp />
                <Toaster
                    position="top-right"
                    toastOptions={{
                        duration: 4000,
                        style: {
                            background: '#1F2937',
                            color: '#F9FAFB',
                            border: '1px solid #374151',
                        },
                    }}
                />
            </div>
        </Provider>
    );
}

export default App;
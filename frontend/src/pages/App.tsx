import { Provider } from 'react-redux';
import { Toaster } from 'react-hot-toast';
import { store } from '../store';
import SimulatorPage from './SimulatorPage';
import Header from '../components/Header';

function App() {
    return (
        <Provider store={store}>
            <div className="min-h-screen bg-gray-900 text-white">
                <Header />
                <main className="container mx-auto px-4 py-8">
                    <SimulatorPage />
                </main>
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
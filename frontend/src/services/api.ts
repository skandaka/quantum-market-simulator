import axios, { AxiosInstance } from 'axios';
import { SimulationRequest, SimulationResponse } from '../types';

class EnhancedSimulationAPI {
    private client: AxiosInstance;
    private wsConnection: WebSocket | null = null;

    constructor() {
        this.client = axios.create({
            baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        // Add request interceptor for auth if needed
        this.client.interceptors.request.use(
            (config) => {
                // Add auth token if available
                const token = localStorage.getItem('auth_token');
                if (token) {
                    config.headers.Authorization = `Bearer ${token}`;
                }
                return config;
            },
            (error) => Promise.reject(error)
        );

        // Add response interceptor for error handling
        this.client.interceptors.response.use(
            (response) => response,
            (error) => {
                if (error.response?.status === 401) {
                    // Handle unauthorized
                    console.error('Unauthorized access');
                }
                return Promise.reject(error);
            }
        );
    }

    // Standard simulation
    async runSimulation(request: SimulationRequest): Promise<SimulationResponse> {
        const response = await this.client.post<SimulationResponse>('/api/v1/simulate', request);
        return response.data;
    }

    // Enhanced simulation with all features
    async runEnhancedSimulation(request: any): Promise<any> {
        const response = await this.client.post('/api/v2/simulate/enhanced', request);
        return response.data;
    }

    // Portfolio upload
    async uploadPortfolio(file: File): Promise<any> {
        const formData = new FormData();
        formData.append('file', file);

        const response = await this.client.post('/api/v2/portfolio/upload', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    }

    // Portfolio analysis
    async analyzePortfolio(portfolioData: any): Promise<any> {
        const response = await this.client.post('/api/v2/portfolio/analyze', portfolioData);
        return response.data;
    }

    // Get quantum visualization data
    async getQuantumStates(simulationId: string): Promise<any> {
        const response = await this.client.get(`/api/v2/visualization/quantum-states`, {
            params: { simulation_id: simulationId },
        });
        return response.data;
    }

    // Export quantum circuit
    async exportCircuit(format: string, circuitData?: any): Promise<any> {
        const response = await this.client.post('/api/v2/circuit/export', {
            export_format: format,
            circuit_data: circuitData,
        });
        return response.data;
    }

    // Get quantum advantage metrics
    async getQuantumAdvantageMetrics(simulationId?: string): Promise<any> {
        const response = await this.client.get('/api/v2/metrics/quantum-advantage', {
            params: { simulation_id: simulationId },
        });
        return response.data;
    }

    // WebSocket connection for real-time monitoring
    connectQuantumMonitor(onMessage: (data: any) => void, onError?: (error: any) => void): void {
        const wsUrl = (import.meta.env.VITE_API_URL || 'http://localhost:8000')
            .replace('http://', 'ws://')
            .replace('https://', 'wss://');

        this.wsConnection = new WebSocket(`${wsUrl}/api/v2/ws/quantum-monitor`);

        this.wsConnection.onopen = () => {
            console.log('Quantum monitor connected');
        };

        this.wsConnection.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                onMessage(data);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };

        this.wsConnection.onerror = (error) => {
            console.error('WebSocket error:', error);
            if (onError) onError(error);
        };

        this.wsConnection.onclose = () => {
            console.log('Quantum monitor disconnected');
            // Attempt reconnection after delay
            setTimeout(() => {
                if (this.wsConnection?.readyState === WebSocket.CLOSED) {
                    this.connectQuantumMonitor(onMessage, onError);
                }
            }, 5000);
        };
    }

    disconnectQuantumMonitor(): void {
        if (this.wsConnection) {
            this.wsConnection.close();
            this.wsConnection = null;
        }
    }

    // Backtesting
    async runBacktest(request: any): Promise<any> {
        const response = await this.client.post('/api/v1/backtest', request);
        return response.data;
    }

    // Get available assets
    async getAvailableAssets(): Promise<string[]> {
        // Mock for now - would fetch from API
        return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'BTC', 'ETH'];
    }

    // Get market data
    async getMarketData(assets: string[]): Promise<any> {
        const response = await this.client.get('/api/v1/market-data', {
            params: { assets: assets.join(',') },
        });
        return response.data;
    }

    // Health check
    async healthCheck(): Promise<any> {
        const response = await this.client.get('/health');
        return response.data;
    }

    // Get performance metrics
    async getMetrics(): Promise<any> {
        const response = await this.client.get('/metrics');
        return response.data;
    }
}

export const simulationApi = new EnhancedSimulationAPI();
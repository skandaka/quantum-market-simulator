// frontend/src/services/api.ts - Enhanced API service with better error handling

import axios, { AxiosInstance, AxiosError } from 'axios';
import { SimulationRequest, SimulationResponse } from '../types';

class EnhancedSimulationAPI {
    private client: AxiosInstance;
    private wsConnection: WebSocket | null = null;

    constructor() {
        const baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8001';
        console.log('API Base URL:', baseURL);

        this.client = axios.create({
            baseURL,
            headers: {
                'Content-Type': 'application/json',
            },
            timeout: 30000, // 30 second timeout
        });

        // Add request interceptor for debugging and auth
        this.client.interceptors.request.use(
            (config) => {
                console.log(`Making ${config.method?.toUpperCase()} request to: ${config.baseURL}${config.url}`);

                // Add auth token if available
                const token = localStorage.getItem('auth_token');
                if (token) {
                    config.headers.Authorization = `Bearer ${token}`;
                }
                return config;
            },
            (error) => {
                console.error('Request interceptor error:', error);
                return Promise.reject(error);
            }
        );

        // Add response interceptor for better error handling
        this.client.interceptors.response.use(
            (response) => {
                console.log(`Successful response from ${response.config.url}:`, response.status);
                return response;
            },
            (error: AxiosError) => {
                console.error('API Error:', {
                    message: error.message,
                    code: error.code,
                    status: error.response?.status,
                    statusText: error.response?.statusText,
                    data: error.response?.data,
                    url: error.config?.url
                });

                // Handle specific error cases
                if (error.code === 'ECONNREFUSED') {
                    error.message = 'Cannot connect to server. Please ensure the backend is running.';
                } else if (error.code === 'NETWORK_ERROR' || error.code === 'ERR_NETWORK') {
                    error.message = 'Network error. Please check your internet connection and backend server.';
                } else if (error.response?.status === 404) {
                    error.message = 'API endpoint not found. Please check the backend configuration.';
                } else if (error.response?.status === 422) {
                    const detail = (error.response.data as any)?.detail;
                    if (Array.isArray(detail)) {
                        error.message = `Validation error: ${detail.map(d => d.msg).join(', ')}`;
                    } else {
                        error.message = `Validation error: ${detail || 'Invalid request data'}`;
                    }
                }

                return Promise.reject(error);
            }
        );
    }

    // Health check with detailed diagnostics
    async healthCheck(): Promise<any> {
        try {
            console.log('Performing health check...');
            const response = await this.client.get('/health');
            console.log('Health check successful:', response.data);
            return response.data;
        } catch (error) {
            console.error('Health check failed:', error);
            throw error;
        }
    }

    // Standard simulation with enhanced error handling
    async runSimulation(request: SimulationRequest): Promise<SimulationResponse> {
        try {
            console.log('Sending simulation request:', request);

            // Validate request before sending
            if (!request.news_inputs || request.news_inputs.length === 0) {
                throw new Error('At least one news input is required');
            }

            if (!request.target_assets || request.target_assets.length === 0) {
                throw new Error('At least one target asset is required');
            }

            const response = await this.client.post<SimulationResponse>('/api/v1/simulate', request);
            console.log('Simulation completed successfully');
            return response.data;
        } catch (error) {
            console.error('Simulation request failed:', error);
            throw error;
        }
    }

    // Test connection method
    async testConnection(): Promise<{ success: boolean; message: string; details?: any }> {
        try {
            console.log('Testing connection to backend...');

            // Try health check first
            const health = await this.healthCheck();

            // Try getting supported assets
            const assets = await this.client.get('/api/v1/supported-assets').catch(() => null);

            // Try getting quantum status
            const quantumStatus = await this.client.get('/api/v1/quantum-status').catch(() => null);

            return {
                success: true,
                message: 'Connection successful',
                details: {
                    health,
                    assetsEndpoint: assets ? 'working' : 'failed',
                    quantumEndpoint: quantumStatus ? 'working' : 'failed'
                }
            };
        } catch (error: any) {
            return {
                success: false,
                message: error.message || 'Connection failed',
                details: {
                    error: error.code,
                    status: error.response?.status,
                    baseURL: this.client.defaults.baseURL
                }
            };
        }
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
        const wsUrl = (import.meta.env.VITE_API_URL || 'http://localhost:8001')
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
        try {
            const response = await this.client.get('/api/v1/supported-assets');
            return response.data.stocks || [];
        } catch (error) {
            console.warn('Failed to fetch available assets, using defaults');
            return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'BTC-USD', 'ETH-USD'];
        }
    }

    // Get market data
    async getMarketData(assets: string[]): Promise<any> {
        const response = await this.client.get('/api/v1/market-data', {
            params: { assets: assets.join(',') },
        });
        return response.data;
    }

    // Get performance metrics
    async getMetrics(): Promise<any> {
        const response = await this.client.get('/metrics');
        return response.data;
    }
}

export const simulationApi = new EnhancedSimulationAPI();
import axios, { AxiosInstance } from 'axios';
import {
    SimulationRequest,
    SimulationResponse,
    BacktestRequest,
    BacktestResult,
    NewsInput,
    SentimentAnalysis,
    MarketData
} from '../types';

class SimulationAPI {
    private api: AxiosInstance;

    constructor() {
        this.api = axios.create({
            baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        // Request interceptor
        this.api.interceptors.request.use(
            (config) => {
                // Add auth token if available
                const token = localStorage.getItem('auth_token');
                if (token) {
                    config.headers.Authorization = `Bearer ${token}`;
                }
                return config;
            },
            (error) => {
                return Promise.reject(error);
            }
        );

        // Response interceptor
        this.api.interceptors.response.use(
            (response) => response,
            (error) => {
                if (error.response?.status === 401) {
                    // Handle unauthorized
                    localStorage.removeItem('auth_token');
                    window.location.href = '/login';
                }
                return Promise.reject(error);
            }
        );
    }

    // Simulation endpoints
    async runSimulation(request: SimulationRequest): Promise<SimulationResponse> {
        const response = await this.api.post<SimulationResponse>('/simulate', request);
        return response.data;
    }

    async analyzeSentiment(newsInput: NewsInput, useQuantum: boolean = true): Promise<SentimentAnalysis> {
        const response = await this.api.post<SentimentAnalysis>('/analyze-sentiment', {
            news_input: newsInput,
            use_quantum: useQuantum,
        });
        return response.data;
    }

    async getMarketData(asset: string, assetType: string = 'stock', period: string = '1d'): Promise<MarketData> {
        const response = await this.api.get<MarketData>(`/market-data/${asset}`, {
            params: { asset_type: assetType, period },
        });
        return response.data;
    }

    async runBacktest(request: BacktestRequest): Promise<BacktestResult> {
        const response = await this.api.post<BacktestResult>('/backtest', request);
        return response.data;
    }

    async getQuantumStatus(): Promise<any> {
        const response = await this.api.get('/quantum-status');
        return response.data;
    }

    async getSupportedAssets(): Promise<any> {
        const response = await this.api.get('/supported-assets');
        return response.data;
    }

    // Stream simulation with Server-Sent Events
    streamSimulation(request: SimulationRequest, onMessage: (data: any) => void): EventSource {
        const eventSource = new EventSource(
            `${this.api.defaults.baseURL}/stream-simulation`,
            {
                withCredentials: true,
            }
        );

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            onMessage(data);
        };

        eventSource.onerror = (error) => {
            console.error('SSE error:', error);
            eventSource.close();
        };

        // Send the request data
        this.api.post('/stream-simulation', request);

        return eventSource;
    }

    // Health check
    async healthCheck(): Promise<boolean> {
        try {
            const response = await this.api.get('/health');
            return response.data.status === 'healthy';
        } catch {
            return false;
        }
    }
}

export const simulationAPI = new SimulationAPI();
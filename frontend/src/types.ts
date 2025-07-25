// Re-export types from the backend schemas for consistency

export interface NewsInput {
    content: string;
    source_type: 'headline' | 'tweet' | 'article' | 'sec_filing' | 'earnings_call' | 'press_release';
    source_url?: string;
    published_at?: string;
    author?: string;
    metadata?: Record<string, any>;
}

export interface SimulationRequest {
    news_inputs: NewsInput[];
    target_assets: string[];
    asset_type?: 'stock' | 'crypto' | 'forex' | 'commodity' | 'index';
    simulation_method?: 'quantum_monte_carlo' | 'quantum_walk' | 'hybrid_qml' | 'classical_baseline';
    time_horizon_days?: number;
    num_scenarios?: number;
    include_quantum_metrics?: boolean;
    compare_with_classical?: boolean;
}

export interface SentimentAnalysis {
    sentiment: 'very_negative' | 'negative' | 'neutral' | 'positive' | 'very_positive';
    confidence: number;
    quantum_sentiment_vector: number[];
    classical_sentiment_score: number;
    entities_detected: Array<{
        text: string;
        type: string;
        start: number;
        end: number;
    }>;
    key_phrases: string[];
    market_impact_keywords: string[];
}

export interface PriceScenario {
    scenario_id: number;
    price_path: number[];
    returns_path: number[];
    volatility_path: number[];
    probability_weight: number;
}

export interface MarketPrediction {
    asset: string;
    current_price: number;
    predicted_scenarios: PriceScenario[];
    expected_return: number;
    volatility: number;
    confidence_intervals: Record<string, { lower: number; upper: number }>;
    quantum_uncertainty: number;
    regime_probabilities: {
        bull: number;
        bear: number;
        neutral: number;
    };
}

export interface QuantumMetrics {
    circuit_depth: number;
    num_qubits: number;
    quantum_volume: number;
    entanglement_measure: number;
    execution_time_ms: number;
    hardware_backend: string;
    success_probability: number;
}

export interface SimulationResponse {
    request_id: string;
    timestamp: string;
    news_analysis: SentimentAnalysis[];
    market_predictions: MarketPrediction[];
    quantum_metrics?: QuantumMetrics;
    classical_comparison?: {
        predictions: MarketPrediction[];
        performance_diff: {
            prediction_differences: Array<{
                asset: string;
                return_difference: number;
                volatility_difference: number;
                ci_width_ratio: number;
            }>;
            uncertainty_comparison: {
                avg_quantum_uncertainty: number;
                avg_classical_uncertainty: number;
            };
        };
    };
    execution_time_seconds: number;
    warnings: string[];
}

export interface BacktestRequest {
    historical_news: NewsInput[];
    asset: string;
    start_date: string;
    end_date: string;
    initial_capital?: number;
    position_size?: number;
}

export interface BacktestResult {
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    profit_factor: number;
    trades: Array<{
        date: string;
        action: 'buy' | 'sell';
        price: number;
        quantity: number;
        pnl: number;
    }>;
    equity_curve: number[];
    performance_metrics: Record<string, number>;
}

export interface MarketData {
    asset: string;
    type: string;
    data: {
        symbol: string;
        current_price: number;
        previous_close: number;
        open: number;
        high: number;
        low: number;
        volume: number;
        volatility: number;
        trend: number;
        timestamp: string;
    };
    timestamp: string;
}
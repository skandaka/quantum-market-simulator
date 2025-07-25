"""Classical ML models for baseline comparison"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime, timedelta

from app.models.schemas import SentimentAnalysis, PriceScenario
from app.utils.helpers import calculate_sharpe_ratio, calculate_max_drawdown

logger = logging.getLogger(__name__)


class ClassicalPredictor:
    """Classical ML predictor for market movements"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.initialized = False

    async def initialize(self):
        """Initialize or load pre-trained models"""

        try:
            # Try to load pre-trained models
            self.models['rf'] = joblib.load('models/rf_predictor.pkl')
            self.models['gb'] = joblib.load('models/gb_predictor.pkl')
            self.scalers['features'] = joblib.load('models/feature_scaler.pkl')
            logger.info("Loaded pre-trained models")
        except:
            # Train new models with synthetic data
            logger.info("Training new classical models...")
            self._train_models()

        self.initialized = True

    def _train_models(self):
        """Train models on synthetic data"""

        # Generate synthetic training data
        X, y = self._generate_synthetic_data(n_samples=1000)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)

        # Train Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)

        # Store models
        self.models['rf'] = rf_model
        self.models['gb'] = gb_model
        self.scalers['features'] = scaler

        # Calculate feature importance
        self.feature_importance['rf'] = rf_model.feature_importances_
        self.feature_importance['gb'] = gb_model.feature_importances_

        # Log performance
        rf_score = rf_model.score(X_test_scaled, y_test)
        gb_score = gb_model.score(X_test_scaled, y_test)
        logger.info(f"Model R² scores - RF: {rf_score:.3f}, GB: {gb_score:.3f}")

    def _generate_synthetic_data(
            self,
            n_samples: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data"""

        # Features: sentiment, volume, volatility, trend, day_of_week, etc.
        n_features = 15

        # Generate random features
        X = np.random.randn(n_samples, n_features)

        # Create realistic relationships
        sentiment_impact = X[:, 0] * 0.02  # Sentiment
        volume_impact = np.log1p(np.abs(X[:, 1])) * 0.001  # Volume
        volatility_impact = X[:, 2] * -0.005  # Volatility (negative)
        trend_impact = X[:, 3] * 0.01  # Trend

        # Add some non-linear relationships
        interaction = X[:, 0] * X[:, 2] * 0.001  # Sentiment × Volatility

        # Generate target (next day return)
        y = (
                sentiment_impact +
                volume_impact +
                volatility_impact +
                trend_impact +
                interaction +
                np.random.normal(0, 0.01, n_samples)  # Noise
        )

        return X, y

    def extract_features(
            self,
            sentiment: SentimentAnalysis,
            market_data: Dict[str, Any],
            historical_data: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Extract features for prediction"""

        features = []

        # Sentiment features
        sentiment_map = {
            "very_negative": -2, "negative": -1, "neutral": 0,
            "positive": 1, "very_positive": 2
        }
        features.append(sentiment_map.get(sentiment.sentiment.value, 0))
        features.append(sentiment.confidence)

        # Quantum features if available
        if sentiment.quantum_sentiment_vector:
            features.extend(sentiment.quantum_sentiment_vector[:3])  # Top 3
        else:
            features.extend([0, 0, 0])

        # Market features
        features.append(market_data.get("volatility", 0.2))
        features.append(np.log1p(market_data.get("volume", 1000000)))
        features.append(market_data.get("trend", 0))

        # Time features
        now = datetime.now()
        features.append(now.weekday())  # Day of week
        features.append(now.hour)  # Hour of day

        # Technical indicators (if historical data available)
        if historical_data is not None and len(historical_data) > 20:
            # RSI
            rsi = self._calculate_rsi(historical_data['Close'])
            features.append(rsi)

            # Moving average convergence
            sma_20 = historical_data['Close'].rolling(20).mean().iloc[-1]
            price = historical_data['Close'].iloc[-1]
            ma_divergence = (price - sma_20) / sma_20
            features.append(ma_divergence)
        else:
            features.extend([50, 0])  # Default RSI and MA divergence

        # Pad to expected feature size
        while len(features) < 15:
            features.append(0)

        return np.array(features).reshape(1, -1)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""

        if len(prices) < period + 1:
            return 50.0  # Neutral RSI

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

    def predict_return(
            self,
            features: np.ndarray,
            model_type: str = 'ensemble'
    ) -> Tuple[float, float]:
        """Predict expected return and confidence"""

        if not self.initialized:
            raise RuntimeError("Models not initialized")

        # Scale features
        features_scaled = self.scalers['features'].transform(features)

        if model_type == 'rf':
            prediction = self.models['rf'].predict(features_scaled)[0]
            # Use prediction variance from trees as uncertainty
            tree_predictions = np.array([
                tree.predict(features_scaled)[0]
                for tree in self.models['rf'].estimators_
            ])
            confidence = 1 / (1 + np.std(tree_predictions))

        elif model_type == 'gb':
            prediction = self.models['gb'].predict(features_scaled)[0]
            confidence = 0.7  # Fixed confidence for GB

        else:  # ensemble
            rf_pred = self.models['rf'].predict(features_scaled)[0]
            gb_pred = self.models['gb'].predict(features_scaled)[0]
            prediction = (rf_pred + gb_pred) / 2

            # Confidence based on agreement
            agreement = 1 - abs(rf_pred - gb_pred) / (abs(rf_pred) + abs(gb_pred) + 1e-6)
            confidence = 0.5 + 0.4 * agreement

        return prediction, confidence

    def generate_scenarios(
            self,
            expected_return: float,
            confidence: float,
            current_price: float,
            time_horizon: int,
            num_scenarios: int = 1000
    ) -> List[PriceScenario]:
        """Generate price scenarios using classical methods"""

        scenarios = []

        # Calculate parameters
        daily_return = expected_return / time_horizon

        # Volatility based on confidence (lower confidence = higher vol)
        base_volatility = 0.02  # 2% daily
        volatility = base_volatility * (2 - confidence)

        for i in range(num_scenarios):
            price_path = [current_price]
            returns_path = []

            for t in range(time_horizon):
                # Generate return with mean reversion
                mean_reversion_strength = 0.1
                current_deviation = (price_path[-1] - current_price) / current_price

                # Mean reverting component
                mean_reversion = -mean_reversion_strength * current_deviation

                # Random component
                random_shock = np.random.normal(0, volatility)

                # Total return
                period_return = daily_return + mean_reversion + random_shock
                returns_path.append(period_return)

                # Update price
                new_price = price_path[-1] * (1 + period_return)
                price_path.append(new_price)

            # Calculate path volatility
            path_vol = [volatility * (1 + 0.1 * np.sin(t * np.pi / time_horizon))
                        for t in range(time_horizon)]

            scenarios.append(PriceScenario(
                scenario_id=i,
                price_path=price_path,
                returns_path=returns_path,
                volatility_path=path_vol,
                probability_weight=1.0 / num_scenarios
            ))

        return scenarios

    def calculate_feature_importance(self) -> Dict[str, List[Tuple[str, float]]]:
        """Get feature importance from models"""

        feature_names = [
            "sentiment_score", "sentiment_confidence", "quantum_1", "quantum_2", "quantum_3",
            "volatility", "log_volume", "trend", "day_of_week", "hour",
            "rsi", "ma_divergence", "feature_12", "feature_13", "feature_14"
        ]

        importance_dict = {}

        for model_name, importances in self.feature_importance.items():
            # Sort by importance
            feature_importance_pairs = [
                (name, float(imp))
                for name, imp in zip(feature_names, importances)
            ]
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

            importance_dict[model_name] = feature_importance_pairs[:10]  # Top 10

        return importance_dict
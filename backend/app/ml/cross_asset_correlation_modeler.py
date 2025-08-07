"""
PHASE 4.4: CROSS-ASSET CORRELATION MODELING
Advanced quantum-enhanced correlation analysis for multi-asset market modeling
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize

try:
    from classiq import qfunc, QBit, QArray, Output, H, RY, RZ, CX, control
    CLASSIQ_AVAILABLE = True
except ImportError:
    CLASSIQ_AVAILABLE = False

from app.quantum.classiq_client import ClassiqClient
from app.quantum.quantum_finance import QuantumFinanceAlgorithms
from app.models.schemas import MarketPrediction
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CorrelationMetrics:
    """Comprehensive correlation metrics between assets"""
    pearson_correlation: float
    spearman_correlation: float
    quantum_correlation: Optional[float]
    time_varying_correlation: List[float]
    volatility_correlation: float
    tail_correlation: float
    correlation_stability: float
    confidence_interval: Tuple[float, float]


@dataclass
class CrossAssetRelationship:
    """Detailed relationship analysis between asset pairs"""
    asset_pair: Tuple[str, str]
    correlation_metrics: CorrelationMetrics
    lead_lag_relationship: Dict[str, float]
    causal_strength: float
    market_regime_correlations: Dict[str, float]
    sector_coupling: float
    quantum_entanglement_measure: Optional[float]
    

class CrossAssetCorrelationModeler:
    """
    PHASE 4.4: Advanced correlation modeling with quantum enhancement
    for understanding complex multi-asset relationships
    """
    
    def __init__(self, classiq_client: ClassiqClient):
        self.classiq_client = classiq_client
        self.quantum_finance = QuantumFinanceAlgorithms(classiq_client) if classiq_client else None
        
        # Correlation analysis parameters
        self.correlation_window = 252  # 1 year of daily data
        self.min_correlation_threshold = 0.1
        self.stability_threshold = 0.8
        
        # Market regime definitions
        self.market_regimes = {
            "bull_market": {"volatility": (0.0, 0.15), "trend": (0.05, 1.0)},
            "bear_market": {"volatility": (0.0, 0.25), "trend": (-1.0, -0.05)},
            "high_volatility": {"volatility": (0.25, 1.0), "trend": (-0.5, 0.5)},
            "low_volatility": {"volatility": (0.0, 0.10), "trend": (-0.1, 0.1)}
        }
        
        # Historical correlation data
        self.correlation_history = {}
        self.regime_correlations = {}
        
        # Quantum correlation circuits cache
        self.quantum_correlation_cache = {}
        
    async def initialize(self):
        """Initialize the cross-asset correlation modeler"""
        logger.info("🔄 Initializing Cross-Asset Correlation Modeler")
        
        try:
            if self.quantum_finance:
                # Quantum finance algorithms don't need separate initialization
                logger.info("✅ Quantum finance algorithms available")
            
            # Initialize correlation tracking structures
            self.correlation_history = {}
            self.regime_correlations = {}
            
            logger.info("🎯 Cross-asset correlation modeler ready")
            
        except Exception as e:
            logger.error(f"❌ Cross-asset correlation modeler initialization failed: {e}")
            raise

    async def analyze_cross_asset_correlations(
        self,
        market_data: Dict[str, Any],
        target_assets: List[str],
        historical_data: Optional[Dict[str, List[float]]] = None,
        quantum_enhanced: bool = True
    ) -> Dict[str, Any]:
        """
        PHASE 4.4.1: Comprehensive cross-asset correlation analysis
        """
        try:
            logger.info(f"🔗 Analyzing cross-asset correlations for {len(target_assets)} assets")
            
            if len(target_assets) < 2:
                logger.warning("Need at least 2 assets for correlation analysis")
                return {"error": "Insufficient assets for correlation analysis"}
            
            start_time = datetime.now()
            
            # Step 1: Prepare correlation data
            correlation_data = await self._prepare_correlation_data(
                market_data, target_assets, historical_data
            )
            
            if not correlation_data:
                return {"error": "Insufficient data for correlation analysis"}
            
            # Step 2: Calculate pairwise correlations
            pairwise_correlations = await self._calculate_pairwise_correlations(
                correlation_data, target_assets, quantum_enhanced
            )
            
            # Step 3: Analyze market regime correlations
            regime_correlations = await self._analyze_regime_correlations(
                correlation_data, target_assets
            )
            
            # Step 4: Calculate time-varying correlations
            time_varying_correlations = await self._calculate_time_varying_correlations(
                correlation_data, target_assets
            )
            
            # Step 5: Detect correlation clusters and groups
            correlation_clusters = await self._detect_correlation_clusters(
                pairwise_correlations, target_assets
            )
            
            # Step 6: Analyze lead-lag relationships
            lead_lag_relationships = await self._analyze_lead_lag_relationships(
                correlation_data, target_assets
            )
            
            # Step 7: Calculate systemic risk metrics
            systemic_risk_metrics = await self._calculate_systemic_risk_metrics(
                pairwise_correlations, regime_correlations
            )
            
            # Step 8: Generate quantum correlation insights (if available)
            quantum_insights = {}
            if quantum_enhanced and self.quantum_finance:
                quantum_insights = await self._generate_quantum_correlation_insights(
                    correlation_data, target_assets
                )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "correlation_analysis": {
                    "pairwise_correlations": pairwise_correlations,
                    "regime_correlations": regime_correlations,
                    "time_varying_correlations": time_varying_correlations,
                    "correlation_clusters": correlation_clusters,
                    "lead_lag_relationships": lead_lag_relationships,
                    "systemic_risk_metrics": systemic_risk_metrics
                },
                "quantum_insights": quantum_insights,
                "analysis_metadata": {
                    "assets_analyzed": target_assets,
                    "correlation_pairs": len(pairwise_correlations),
                    "quantum_enhanced": quantum_enhanced and self.quantum_finance is not None,
                    "execution_time": execution_time,
                    "data_quality_score": self._calculate_data_quality_score(correlation_data)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Cross-asset correlation analysis failed: {e}")
            return {"error": str(e)}

    async def _prepare_correlation_data(
        self,
        market_data: Dict[str, Any],
        target_assets: List[str],
        historical_data: Optional[Dict[str, List[float]]]
    ) -> Dict[str, Any]:
        """
        PHASE 4.4.2: Prepare and validate data for correlation analysis
        """
        try:
            correlation_data = {}
            
            # Use historical data if provided, otherwise generate from market data
            if historical_data:
                for asset in target_assets:
                    if asset in historical_data and len(historical_data[asset]) >= 30:
                        correlation_data[asset] = {
                            "prices": historical_data[asset],
                            "returns": self._calculate_returns(historical_data[asset]),
                            "volatility": np.std(self._calculate_returns(historical_data[asset])) * np.sqrt(252)
                        }
            else:
                # Generate synthetic historical data from current market data
                for asset in target_assets:
                    asset_data = market_data.get(asset, {})
                    if asset_data:
                        synthetic_data = self._generate_synthetic_historical_data(asset_data)
                        correlation_data[asset] = synthetic_data
            
            # Validate data quality
            valid_assets = []
            for asset, data in correlation_data.items():
                if (len(data.get("prices", [])) >= 30 and 
                    len(data.get("returns", [])) >= 29 and
                    not np.isnan(data.get("volatility", np.nan))):
                    valid_assets.append(asset)
            
            # Keep only valid assets
            correlation_data = {asset: correlation_data[asset] for asset in valid_assets}
            
            logger.info(f"📊 Prepared correlation data for {len(correlation_data)} assets")
            return correlation_data
            
        except Exception as e:
            logger.error(f"Correlation data preparation failed: {e}")
            return {}

    async def _calculate_pairwise_correlations(
        self,
        correlation_data: Dict[str, Any],
        target_assets: List[str],
        quantum_enhanced: bool
    ) -> List[CrossAssetRelationship]:
        """
        PHASE 4.4.3: Calculate comprehensive pairwise correlations
        """
        try:
            pairwise_correlations = []
            assets = list(correlation_data.keys())
            
            for i in range(len(assets)):
                for j in range(i + 1, len(assets)):
                    asset1, asset2 = assets[i], assets[j]
                    
                    returns1 = correlation_data[asset1]["returns"]
                    returns2 = correlation_data[asset2]["returns"]
                    
                    if len(returns1) != len(returns2):
                        min_len = min(len(returns1), len(returns2))
                        returns1 = returns1[:min_len]
                        returns2 = returns2[:min_len]
                    
                    # Calculate classical correlations
                    pearson_corr, _ = pearsonr(returns1, returns2)
                    spearman_corr, _ = spearmanr(returns1, returns2)
                    
                    # Calculate time-varying correlation
                    time_varying_corr = self._calculate_rolling_correlation(returns1, returns2)
                    
                    # Calculate volatility correlation
                    vol1 = self._calculate_rolling_volatility(returns1)
                    vol2 = self._calculate_rolling_volatility(returns2)
                    vol_corr, _ = pearsonr(vol1, vol2) if len(vol1) == len(vol2) else (0.0, 1.0)
                    
                    # Calculate tail correlation
                    tail_corr = self._calculate_tail_correlation(returns1, returns2)
                    
                    # Calculate correlation stability
                    corr_stability = self._calculate_correlation_stability(time_varying_corr)
                    
                    # Calculate confidence interval
                    conf_interval = self._calculate_correlation_confidence_interval(
                        pearson_corr, len(returns1)
                    )
                    
                    # Quantum correlation (if enabled)
                    quantum_corr = None
                    quantum_entanglement = None
                    if quantum_enhanced and self.quantum_finance:
                        quantum_results = await self._calculate_quantum_correlation(
                            returns1, returns2, asset1, asset2
                        )
                        quantum_corr = quantum_results.get("quantum_correlation")
                        quantum_entanglement = quantum_results.get("entanglement_measure")
                    
                    # Create correlation metrics
                    corr_metrics = CorrelationMetrics(
                        pearson_correlation=pearson_corr,
                        spearman_correlation=spearman_corr,
                        quantum_correlation=quantum_corr,
                        time_varying_correlation=time_varying_corr,
                        volatility_correlation=vol_corr,
                        tail_correlation=tail_corr,
                        correlation_stability=corr_stability,
                        confidence_interval=conf_interval
                    )
                    
                    # Calculate lead-lag relationship
                    lead_lag = self._calculate_lead_lag_relationship(returns1, returns2)
                    
                    # Calculate causal strength
                    causal_strength = self._calculate_causal_strength(returns1, returns2)
                    
                    # Market regime correlations
                    regime_corrs = await self._calculate_regime_specific_correlations(
                        returns1, returns2, correlation_data[asset1], correlation_data[asset2]
                    )
                    
                    # Sector coupling
                    sector_coupling = self._calculate_sector_coupling(asset1, asset2)
                    
                    # Create cross-asset relationship
                    relationship = CrossAssetRelationship(
                        asset_pair=(asset1, asset2),
                        correlation_metrics=corr_metrics,
                        lead_lag_relationship=lead_lag,
                        causal_strength=causal_strength,
                        market_regime_correlations=regime_corrs,
                        sector_coupling=sector_coupling,
                        quantum_entanglement_measure=quantum_entanglement
                    )
                    
                    pairwise_correlations.append(relationship)
            
            logger.info(f"📈 Calculated {len(pairwise_correlations)} pairwise correlations")
            return pairwise_correlations
            
        except Exception as e:
            logger.error(f"Pairwise correlation calculation failed: {e}")
            return []

    async def _analyze_regime_correlations(
        self,
        correlation_data: Dict[str, Any],
        target_assets: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        PHASE 4.4.4: Analyze correlations under different market regimes
        """
        try:
            regime_correlations = {}
            
            # Identify market regimes in the data
            regime_periods = self._identify_market_regimes(correlation_data)
            
            for regime, periods in regime_periods.items():
                regime_correlations[regime] = {}
                
                assets = list(correlation_data.keys())
                
                for i in range(len(assets)):
                    for j in range(i + 1, len(assets)):
                        asset1, asset2 = assets[i], assets[j]
                        
                        # Extract returns for this regime
                        regime_returns1 = []
                        regime_returns2 = []
                        
                        for start_idx, end_idx in periods:
                            returns1 = correlation_data[asset1]["returns"][start_idx:end_idx]
                            returns2 = correlation_data[asset2]["returns"][start_idx:end_idx]
                            regime_returns1.extend(returns1)
                            regime_returns2.extend(returns2)
                        
                        if len(regime_returns1) >= 10 and len(regime_returns2) >= 10:
                            corr, _ = pearsonr(regime_returns1, regime_returns2)
                            regime_correlations[regime][f"{asset1}_{asset2}"] = corr
                        else:
                            regime_correlations[regime][f"{asset1}_{asset2}"] = 0.0
            
            return regime_correlations
            
        except Exception as e:
            logger.error(f"Regime correlation analysis failed: {e}")
            return {}

    async def _calculate_time_varying_correlations(
        self,
        correlation_data: Dict[str, Any],
        target_assets: List[str]
    ) -> Dict[str, List[float]]:
        """
        PHASE 4.4.5: Calculate time-varying correlations using rolling windows
        """
        try:
            time_varying_correlations = {}
            assets = list(correlation_data.keys())
            window_size = 60  # 60-day rolling window
            
            for i in range(len(assets)):
                for j in range(i + 1, len(assets)):
                    asset1, asset2 = assets[i], assets[j]
                    
                    returns1 = correlation_data[asset1]["returns"]
                    returns2 = correlation_data[asset2]["returns"]
                    
                    # Calculate rolling correlations
                    rolling_corrs = []
                    for k in range(window_size, min(len(returns1), len(returns2))):
                        window_returns1 = returns1[k-window_size:k]
                        window_returns2 = returns2[k-window_size:k]
                        
                        if len(window_returns1) == len(window_returns2) and len(window_returns1) > 10:
                            corr, _ = pearsonr(window_returns1, window_returns2)
                            rolling_corrs.append(corr if not np.isnan(corr) else 0.0)
                        else:
                            rolling_corrs.append(0.0)
                    
                    time_varying_correlations[f"{asset1}_{asset2}"] = rolling_corrs
            
            return time_varying_correlations
            
        except Exception as e:
            logger.error(f"Time-varying correlation calculation failed: {e}")
            return {}

    async def _detect_correlation_clusters(
        self,
        pairwise_correlations: List[CrossAssetRelationship],
        target_assets: List[str]
    ) -> Dict[str, Any]:
        """
        PHASE 4.4.6: Detect correlation clusters and asset groups
        """
        try:
            # Create correlation matrix
            n_assets = len(target_assets)
            corr_matrix = np.eye(n_assets)
            asset_indices = {asset: i for i, asset in enumerate(target_assets)}
            
            for relationship in pairwise_correlations:
                asset1, asset2 = relationship.asset_pair
                if asset1 in asset_indices and asset2 in asset_indices:
                    i, j = asset_indices[asset1], asset_indices[asset2]
                    corr = relationship.correlation_metrics.pearson_correlation
                    corr_matrix[i, j] = corr_matrix[j, i] = corr
            
            # Simple clustering based on correlation thresholds
            high_corr_threshold = 0.7
            medium_corr_threshold = 0.4
            
            clusters = {
                "high_correlation": [],
                "medium_correlation": [],
                "low_correlation": [],
                "cluster_statistics": {}
            }
            
            # Find highly correlated pairs
            for relationship in pairwise_correlations:
                corr = relationship.correlation_metrics.pearson_correlation
                if abs(corr) >= high_corr_threshold:
                    clusters["high_correlation"].append({
                        "assets": relationship.asset_pair,
                        "correlation": corr,
                        "stability": relationship.correlation_metrics.correlation_stability
                    })
                elif abs(corr) >= medium_corr_threshold:
                    clusters["medium_correlation"].append({
                        "assets": relationship.asset_pair,
                        "correlation": corr,
                        "stability": relationship.correlation_metrics.correlation_stability
                    })
                else:
                    clusters["low_correlation"].append({
                        "assets": relationship.asset_pair,
                        "correlation": corr,
                        "stability": relationship.correlation_metrics.correlation_stability
                    })
            
            # Calculate cluster statistics
            all_corrs = [rel.correlation_metrics.pearson_correlation for rel in pairwise_correlations]
            clusters["cluster_statistics"] = {
                "avg_correlation": np.mean(all_corrs),
                "correlation_std": np.std(all_corrs),
                "high_corr_count": len(clusters["high_correlation"]),
                "medium_corr_count": len(clusters["medium_correlation"]),
                "low_corr_count": len(clusters["low_correlation"]),
                "diversification_ratio": len(clusters["low_correlation"]) / len(pairwise_correlations)
            }
            
            return clusters
            
        except Exception as e:
            logger.error(f"Correlation cluster detection failed: {e}")
            return {}

    async def _analyze_lead_lag_relationships(
        self,
        correlation_data: Dict[str, Any],
        target_assets: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        PHASE 4.4.7: Analyze lead-lag relationships between assets
        """
        try:
            lead_lag_relationships = {}
            assets = list(correlation_data.keys())
            max_lag = 5  # Maximum lag periods to consider
            
            for i in range(len(assets)):
                for j in range(len(assets)):
                    if i != j:
                        asset1, asset2 = assets[i], assets[j]
                        
                        returns1 = correlation_data[asset1]["returns"]
                        returns2 = correlation_data[asset2]["returns"]
                        
                        # Calculate cross-correlations at different lags
                        cross_correlations = []
                        for lag in range(-max_lag, max_lag + 1):
                            if lag == 0:
                                corr, _ = pearsonr(returns1, returns2)
                            elif lag > 0:
                                # asset1 leads asset2
                                if len(returns1) > lag and len(returns2) > lag:
                                    corr, _ = pearsonr(returns1[:-lag], returns2[lag:])
                                else:
                                    corr = 0.0
                            else:
                                # asset2 leads asset1
                                lag_abs = abs(lag)
                                if len(returns1) > lag_abs and len(returns2) > lag_abs:
                                    corr, _ = pearsonr(returns1[lag_abs:], returns2[:-lag_abs])
                                else:
                                    corr = 0.0
                            
                            cross_correlations.append(corr if not np.isnan(corr) else 0.0)
                        
                        # Find optimal lag
                        max_corr_idx = np.argmax(np.abs(cross_correlations))
                        optimal_lag = max_corr_idx - max_lag
                        max_corr = cross_correlations[max_corr_idx]
                        
                        lead_lag_relationships[f"{asset1}_to_{asset2}"] = {
                            "optimal_lag": optimal_lag,
                            "max_correlation": max_corr,
                            "lead_strength": max_corr if optimal_lag < 0 else 0.0,
                            "lag_strength": max_corr if optimal_lag > 0 else 0.0,
                            "cross_correlations": cross_correlations
                        }
            
            return lead_lag_relationships
            
        except Exception as e:
            logger.error(f"Lead-lag relationship analysis failed: {e}")
            return {}

    async def _calculate_systemic_risk_metrics(
        self,
        pairwise_correlations: List[CrossAssetRelationship],
        regime_correlations: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        PHASE 4.4.8: Calculate systemic risk metrics from correlation analysis
        """
        try:
            # Extract correlation values
            all_correlations = [
                rel.correlation_metrics.pearson_correlation 
                for rel in pairwise_correlations
            ]
            
            if not all_correlations:
                return {"error": "No correlations available for systemic risk calculation"}
            
            # Calculate systemic risk metrics
            systemic_risk_metrics = {
                "average_correlation": np.mean(all_correlations),
                "correlation_concentration": self._calculate_correlation_concentration(all_correlations),
                "tail_risk_correlation": self._calculate_tail_risk_correlation(pairwise_correlations),
                "regime_correlation_instability": self._calculate_regime_instability(regime_correlations),
                "diversification_ratio": self._calculate_diversification_ratio(all_correlations),
                "contagion_risk": self._calculate_contagion_risk(pairwise_correlations),
                "market_fragility": self._calculate_market_fragility(all_correlations)
            }
            
            # Overall systemic risk score (0-1, higher is riskier)
            systemic_risk_metrics["overall_systemic_risk"] = self._calculate_overall_systemic_risk(
                systemic_risk_metrics
            )
            
            return systemic_risk_metrics
            
        except Exception as e:
            logger.error(f"Systemic risk metrics calculation failed: {e}")
            return {"error": str(e)}

    async def _generate_quantum_correlation_insights(
        self,
        correlation_data: Dict[str, Any],
        target_assets: List[str]
    ) -> Dict[str, Any]:
        """
        PHASE 4.4.9: Generate quantum-enhanced correlation insights
        """
        try:
            if not self.quantum_finance:
                return {"quantum_available": False}
            
            quantum_insights = {
                "quantum_available": True,
                "quantum_correlations": {},
                "entanglement_measures": {},
                "quantum_advantage": {},
                "coherence_analysis": {}
            }
            
            assets = list(correlation_data.keys())
            
            # Calculate quantum correlations for selected pairs
            for i in range(min(len(assets), 3)):  # Limit for performance
                for j in range(i + 1, min(len(assets), 3)):
                    asset1, asset2 = assets[i], assets[j]
                    
                    returns1 = correlation_data[asset1]["returns"][:50]  # Limit data for quantum
                    returns2 = correlation_data[asset2]["returns"][:50]
                    
                    # Generate quantum correlation
                    quantum_result = await self._calculate_quantum_correlation(
                        returns1, returns2, asset1, asset2
                    )
                    
                    if quantum_result:
                        pair_key = f"{asset1}_{asset2}"
                        quantum_insights["quantum_correlations"][pair_key] = quantum_result.get("quantum_correlation", 0.0)
                        quantum_insights["entanglement_measures"][pair_key] = quantum_result.get("entanglement_measure", 0.0)
                        
                        # Calculate quantum advantage
                        classical_corr = pearsonr(returns1, returns2)[0] if len(returns1) == len(returns2) else 0.0
                        quantum_corr = quantum_result.get("quantum_correlation", 0.0)
                        quantum_advantage = abs(quantum_corr) - abs(classical_corr)
                        quantum_insights["quantum_advantage"][pair_key] = quantum_advantage
            
            # Calculate overall quantum coherence
            if quantum_insights["entanglement_measures"]:
                avg_entanglement = np.mean(list(quantum_insights["entanglement_measures"].values()))
                quantum_insights["coherence_analysis"] = {
                    "average_entanglement": avg_entanglement,
                    "quantum_coherence_strength": min(avg_entanglement * 2, 1.0),
                    "quantum_enhancement_factor": 1.0 + avg_entanglement * 0.2
                }
            
            return quantum_insights
            
        except Exception as e:
            logger.error(f"Quantum correlation insights generation failed: {e}")
            return {"quantum_available": False, "error": str(e)}

    # Helper methods for correlation calculations
    def _calculate_returns(self, prices: List[float]) -> List[float]:
        """Calculate returns from price series"""
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
            else:
                returns.append(0.0)
        
        return returns

    def _generate_synthetic_historical_data(self, asset_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate synthetic historical data for correlation analysis"""
        current_price = asset_data.get("current_price", 100.0)
        volatility = asset_data.get("volatility", 0.25)
        drift = asset_data.get("expected_return", 0.05) / 252  # Daily drift
        
        # Generate 100 days of synthetic data
        prices = [current_price]
        for _ in range(100):
            shock = np.random.normal(0, volatility / np.sqrt(252))
            new_price = prices[-1] * (1 + drift + shock)
            prices.append(max(new_price, 0.01))  # Ensure positive prices
        
        returns = self._calculate_returns(prices)
        
        return {
            "prices": prices,
            "returns": returns,
            "volatility": np.std(returns) * np.sqrt(252)
        }

    def _calculate_rolling_correlation(self, returns1: List[float], returns2: List[float], window: int = 30) -> List[float]:
        """Calculate rolling correlation between two return series"""
        rolling_corrs = []
        
        for i in range(window, min(len(returns1), len(returns2))):
            window_returns1 = returns1[i-window:i]
            window_returns2 = returns2[i-window:i]
            
            if len(window_returns1) == len(window_returns2) and len(window_returns1) > 5:
                corr, _ = pearsonr(window_returns1, window_returns2)
                rolling_corrs.append(corr if not np.isnan(corr) else 0.0)
            else:
                rolling_corrs.append(0.0)
        
        return rolling_corrs

    def _calculate_rolling_volatility(self, returns: List[float], window: int = 30) -> List[float]:
        """Calculate rolling volatility"""
        rolling_vols = []
        
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            vol = np.std(window_returns) * np.sqrt(252)
            rolling_vols.append(vol)
        
        return rolling_vols

    def _calculate_tail_correlation(self, returns1: List[float], returns2: List[float], percentile: float = 0.05) -> float:
        """Calculate correlation in tail events"""
        if len(returns1) != len(returns2) or len(returns1) < 20:
            return 0.0
        
        # Find tail events (extreme negative returns)
        threshold1 = np.percentile(returns1, percentile * 100)
        threshold2 = np.percentile(returns2, percentile * 100)
        
        tail_returns1 = [r for r in returns1 if r <= threshold1]
        tail_returns2 = [returns2[i] for i, r in enumerate(returns1) if r <= threshold1 and i < len(returns2)]
        
        if len(tail_returns1) >= 5 and len(tail_returns2) >= 5:
            corr, _ = pearsonr(tail_returns1, tail_returns2)
            return corr if not np.isnan(corr) else 0.0
        
        return 0.0

    def _calculate_correlation_stability(self, time_varying_corr: List[float]) -> float:
        """Calculate stability of correlation over time"""
        if len(time_varying_corr) < 10:
            return 0.5
        
        # Stability is inverse of volatility of correlations
        corr_volatility = np.std(time_varying_corr)
        stability = max(0.0, 1.0 - corr_volatility)
        
        return stability

    def _calculate_correlation_confidence_interval(self, correlation: float, sample_size: int) -> Tuple[float, float]:
        """Calculate confidence interval for correlation"""
        if sample_size < 3:
            return (correlation - 0.5, correlation + 0.5)
        
        # Fisher z-transformation
        z = 0.5 * np.log((1 + correlation) / (1 - correlation)) if abs(correlation) < 0.99 else 0.0
        se = 1 / np.sqrt(sample_size - 3)
        
        # 95% confidence interval
        z_lower = z - 1.96 * se
        z_upper = z + 1.96 * se
        
        # Transform back
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return (r_lower, r_upper)

    async def _calculate_quantum_correlation(
        self, 
        returns1: List[float], 
        returns2: List[float], 
        asset1: str, 
        asset2: str
    ) -> Optional[Dict[str, float]]:
        """Calculate quantum-enhanced correlation"""
        try:
            if not self.quantum_finance:
                return None
            
            # Use quantum correlation circuit
            correlation_result = await self.quantum_finance.quantum_correlation_circuit(
                returns1[:20], returns2[:20]  # Limit data size for quantum processing
            )
            
            if correlation_result:
                return {
                    "quantum_correlation": correlation_result.get("quantum_correlation", 0.0),
                    "entanglement_measure": correlation_result.get("entanglement_measure", 0.0),
                    "quantum_advantage": correlation_result.get("quantum_advantage", 0.0)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Quantum correlation calculation failed: {e}")
            return None

    def _calculate_lead_lag_relationship(self, returns1: List[float], returns2: List[float]) -> Dict[str, float]:
        """Calculate lead-lag relationship between two assets"""
        try:
            if len(returns1) < 10 or len(returns2) < 10:
                return {"lead_strength": 0.0, "lag_strength": 0.0, "optimal_lag": 0}
            
            max_lag = min(5, len(returns1) // 4)
            max_corr = 0.0
            optimal_lag = 0
            
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    corr, _ = pearsonr(returns1, returns2)
                elif lag > 0:
                    if len(returns1) > lag:
                        corr, _ = pearsonr(returns1[:-lag], returns2[lag:])
                    else:
                        corr = 0.0
                else:
                    lag_abs = abs(lag)
                    if len(returns2) > lag_abs:
                        corr, _ = pearsonr(returns1[lag_abs:], returns2[:-lag_abs])
                    else:
                        corr = 0.0
                
                if abs(corr) > abs(max_corr):
                    max_corr = corr
                    optimal_lag = lag
            
            return {
                "lead_strength": max_corr if optimal_lag < 0 else 0.0,
                "lag_strength": max_corr if optimal_lag > 0 else 0.0,
                "optimal_lag": optimal_lag,
                "max_correlation": max_corr
            }
            
        except Exception as e:
            logger.error(f"Lead-lag relationship calculation failed: {e}")
            return {"lead_strength": 0.0, "lag_strength": 0.0, "optimal_lag": 0}

    def _calculate_causal_strength(self, returns1: List[float], returns2: List[float]) -> float:
        """Calculate Granger causality-inspired causal strength"""
        try:
            # Simplified causal strength based on predictive power
            if len(returns1) < 20 or len(returns2) < 20:
                return 0.0
            
            # Use simple linear prediction
            X = np.array(returns1[:-1]).reshape(-1, 1)
            y = returns2[1:]
            
            if len(X) != len(y):
                min_len = min(len(X), len(y))
                X = X[:min_len]
                y = y[:min_len]
            
            # Calculate R-squared for predictive relationship
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            try:
                model = LinearRegression().fit(X, y)
                predictions = model.predict(X)
                r2 = r2_score(y, predictions)
                return max(0.0, r2)
            except:
                return 0.0
                
        except Exception as e:
            logger.error(f"Causal strength calculation failed: {e}")
            return 0.0

    async def _calculate_regime_specific_correlations(
        self,
        returns1: List[float],
        returns2: List[float],
        asset1_data: Dict[str, Any],
        asset2_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate correlations for different market regimes"""
        try:
            regime_correlations = {}
            
            # Simple regime identification based on volatility
            vol1 = asset1_data.get("volatility", 0.25)
            vol2 = asset2_data.get("volatility", 0.25)
            avg_vol = (vol1 + vol2) / 2
            
            if avg_vol > 0.3:
                regime = "high_volatility"
            elif avg_vol < 0.15:
                regime = "low_volatility"
            else:
                regime = "normal_volatility"
            
            # Calculate correlation for the identified regime
            if len(returns1) == len(returns2) and len(returns1) >= 10:
                corr, _ = pearsonr(returns1, returns2)
                regime_correlations[regime] = corr if not np.isnan(corr) else 0.0
            else:
                regime_correlations[regime] = 0.0
            
            # Add default values for other regimes
            for regime_name in ["bull_market", "bear_market", "high_volatility", "low_volatility"]:
                if regime_name not in regime_correlations:
                    regime_correlations[regime_name] = regime_correlations.get(regime, 0.0)
            
            return regime_correlations
            
        except Exception as e:
            logger.error(f"Regime-specific correlation calculation failed: {e}")
            return {"normal_volatility": 0.0}

    def _calculate_sector_coupling(self, asset1: str, asset2: str) -> float:
        """Calculate sector coupling strength between assets"""
        # Simplified sector mapping
        sector_map = {
            "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology",
            "JPM": "finance", "BAC": "finance", "WFC": "finance",
            "JNJ": "healthcare", "PFE": "healthcare", "UNH": "healthcare"
        }
        
        sector1 = sector_map.get(asset1, "unknown")
        sector2 = sector_map.get(asset2, "unknown")
        
        if sector1 == sector2 and sector1 != "unknown":
            return 0.8  # High coupling within same sector
        elif sector1 != "unknown" and sector2 != "unknown":
            return 0.3  # Medium coupling across sectors
        else:
            return 0.1  # Low coupling for unknown sectors

    def _identify_market_regimes(self, correlation_data: Dict[str, Any]) -> Dict[str, List[Tuple[int, int]]]:
        """Identify market regime periods in the data"""
        try:
            # Simple regime identification based on average volatility
            all_returns = []
            for asset_data in correlation_data.values():
                all_returns.extend(asset_data.get("returns", []))
            
            if not all_returns:
                return {"normal": [(0, 100)]}
            
            # Calculate rolling volatility
            window = 30
            regime_periods = {"high_volatility": [], "low_volatility": [], "normal": []}
            
            for i in range(window, len(all_returns), window):
                window_returns = all_returns[i-window:i]
                vol = np.std(window_returns) * np.sqrt(252) if window_returns else 0.25
                
                start_idx = i - window
                end_idx = i
                
                if vol > 0.3:
                    regime_periods["high_volatility"].append((start_idx, end_idx))
                elif vol < 0.15:
                    regime_periods["low_volatility"].append((start_idx, end_idx))
                else:
                    regime_periods["normal"].append((start_idx, end_idx))
            
            return regime_periods
            
        except Exception as e:
            logger.error(f"Market regime identification failed: {e}")
            return {"normal": [(0, 100)]}

    def _calculate_data_quality_score(self, correlation_data: Dict[str, Any]) -> float:
        """Calculate data quality score for correlation analysis"""
        try:
            if not correlation_data:
                return 0.0
            
            quality_scores = []
            
            for asset, data in correlation_data.items():
                prices = data.get("prices", [])
                returns = data.get("returns", [])
                
                # Data completeness
                completeness = min(len(prices) / 100.0, 1.0)
                
                # Data consistency (no extreme outliers)
                if returns:
                    outlier_threshold = 3 * np.std(returns)
                    outliers = sum(1 for r in returns if abs(r) > outlier_threshold)
                    consistency = max(0.0, 1.0 - outliers / len(returns))
                else:
                    consistency = 0.0
                
                # Overall quality for this asset
                asset_quality = 0.7 * completeness + 0.3 * consistency
                quality_scores.append(asset_quality)
            
            return np.mean(quality_scores) if quality_scores else 0.0
            
        except Exception as e:
            logger.error(f"Data quality score calculation failed: {e}")
            return 0.5

    # Systemic risk calculation helpers
    def _calculate_correlation_concentration(self, correlations: List[float]) -> float:
        """Calculate correlation concentration (higher = more concentrated)"""
        if not correlations:
            return 0.0
        
        abs_corrs = [abs(c) for c in correlations]
        high_corr_count = sum(1 for c in abs_corrs if c > 0.7)
        return high_corr_count / len(correlations)

    def _calculate_tail_risk_correlation(self, pairwise_correlations: List[CrossAssetRelationship]) -> float:
        """Calculate average tail correlation"""
        tail_corrs = [rel.correlation_metrics.tail_correlation for rel in pairwise_correlations]
        return np.mean(tail_corrs) if tail_corrs else 0.0

    def _calculate_regime_instability(self, regime_correlations: Dict[str, Dict[str, float]]) -> float:
        """Calculate instability across market regimes"""
        if not regime_correlations:
            return 0.5
        
        instabilities = []
        
        for pair in regime_correlations.get("high_volatility", {}).keys():
            regime_corrs = []
            for regime in regime_correlations.values():
                if pair in regime:
                    regime_corrs.append(regime[pair])
            
            if len(regime_corrs) > 1:
                instability = np.std(regime_corrs)
                instabilities.append(instability)
        
        return np.mean(instabilities) if instabilities else 0.5

    def _calculate_diversification_ratio(self, correlations: List[float]) -> float:
        """Calculate diversification ratio (higher = better diversified)"""
        if not correlations:
            return 1.0
        
        avg_corr = np.mean([abs(c) for c in correlations])
        return max(0.0, 1.0 - avg_corr)

    def _calculate_contagion_risk(self, pairwise_correlations: List[CrossAssetRelationship]) -> float:
        """Calculate contagion risk based on correlation patterns"""
        high_corr_count = sum(
            1 for rel in pairwise_correlations 
            if abs(rel.correlation_metrics.pearson_correlation) > 0.8
        )
        
        total_pairs = len(pairwise_correlations)
        return high_corr_count / total_pairs if total_pairs > 0 else 0.0

    def _calculate_market_fragility(self, correlations: List[float]) -> float:
        """Calculate market fragility score"""
        if not correlations:
            return 0.5
        
        # Fragility increases with high positive correlations
        positive_corrs = [c for c in correlations if c > 0]
        if not positive_corrs:
            return 0.3
        
        avg_positive_corr = np.mean(positive_corrs)
        return min(avg_positive_corr, 1.0)

    def _calculate_overall_systemic_risk(self, metrics: Dict[str, float]) -> float:
        """Calculate overall systemic risk score"""
        try:
            # Weight different risk components
            weights = {
                "correlation_concentration": 0.25,
                "tail_risk_correlation": 0.20,
                "regime_correlation_instability": 0.15,
                "contagion_risk": 0.25,
                "market_fragility": 0.15
            }
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in metrics and not np.isnan(metrics[metric]):
                    weighted_score += metrics[metric] * weight
                    total_weight += weight
            
            return weighted_score / total_weight if total_weight > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Overall systemic risk calculation failed: {e}")
            return 0.5

    def get_correlation_summary(self) -> Dict[str, Any]:
        """Get summary of correlation analysis capabilities and status"""
        return {
            "correlation_modeler_status": {
                "quantum_enhanced": self.quantum_finance is not None,
                "correlation_window": self.correlation_window,
                "min_correlation_threshold": self.min_correlation_threshold,
                "supported_regimes": list(self.market_regimes.keys())
            },
            "analysis_capabilities": [
                "pairwise_correlations",
                "regime_correlations", 
                "time_varying_correlations",
                "correlation_clustering",
                "lead_lag_analysis",
                "systemic_risk_metrics",
                "quantum_correlation_insights"
            ],
            "historical_data": {
                "correlation_history_size": len(self.correlation_history),
                "regime_correlations_tracked": len(self.regime_correlations)
            }
        }

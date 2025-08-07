"""
PHASE 6: PRODUCTION ENHANCED UNIFIED MARKET SIMULATOR
Production-ready market simulator with optimization, monitoring, and scaling
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

# Core imports
from app.services.unified_market_simulator import UnifiedMarketSimulator
from app.services.production_optimizer import (
    PerformanceMonitor, CacheManager, LoadBalancer, 
    ProductionOptimizer, HealthChecker, SystemHealth
)
from app.models.schemas import MarketPrediction
from app.config import settings

logger = logging.getLogger(__name__)


class ProductionMarketSimulator:
    """
    PHASE 6: Production-Enhanced Market Simulator
    
    Enhanced version of UnifiedMarketSimulator with:
    - Performance monitoring and optimization
    - Intelligent caching and load balancing  
    - Auto-scaling and resource management
    - Comprehensive health monitoring
    - Production deployment optimizations
    """
    
    def __init__(self, classiq_client=None):
        # Core simulator
        self.core_simulator = UnifiedMarketSimulator(classiq_client)
        
        # Production components
        self.performance_monitor = PerformanceMonitor(window_size=200)
        self.cache_manager = CacheManager(max_size=2000, ttl_seconds=300)
        self.load_balancer = LoadBalancer()
        
        # Initialize load balancer with workers
        self._initialize_workers()
        
        # Production optimizer
        self.optimizer = ProductionOptimizer(
            self.performance_monitor,
            self.cache_manager, 
            self.load_balancer
        )
        
        # Health checker
        self.health_checker = HealthChecker({
            "core_simulator": self.core_simulator,
            "performance_monitor": self.performance_monitor,
            "cache_manager": self.cache_manager,
            "load_balancer": self.load_balancer,
            "optimizer": self.optimizer
        })
        
        # Production state
        self.is_production_ready = False
        self.startup_time = None
        self.total_requests = 0
        self.optimization_interval = 300  # 5 minutes
        
        # Background tasks
        self.monitoring_task = None
        self.optimization_task = None
        
    def _initialize_workers(self):
        """Initialize load balancer workers"""
        try:
            # Add quantum workers (limited capacity)
            self.load_balancer.add_quantum_worker("quantum_primary", capacity=50)
            self.load_balancer.add_quantum_worker("quantum_secondary", capacity=30)
            
            # Add classical workers (higher capacity)
            self.load_balancer.add_classical_worker("classical_primary", capacity=200)
            self.load_balancer.add_classical_worker("classical_secondary", capacity=150)
            self.load_balancer.add_classical_worker("classical_tertiary", capacity=100)
            
            logger.info("🔧 Load balancer workers initialized")
            
        except Exception as e:
            logger.error(f"Worker initialization failed: {e}")
            
    async def start_production_services(self):
        """Start all production services"""
        try:
            logger.info("🚀 Starting production market simulator services")
            
            # Initialize core simulator
            await self.core_simulator.initialize()
            
            # Start performance monitoring
            self.monitoring_task = asyncio.create_task(
                self.performance_monitor.start_monitoring(interval_seconds=30)
            )
            
            # Start optimization cycle
            self.optimization_task = asyncio.create_task(
                self._run_optimization_loop()
            )
            
            # Mark as production ready
            self.is_production_ready = True
            self.startup_time = datetime.now()
            
            logger.info("✅ Production services started successfully")
            
        except Exception as e:
            logger.error(f"Production service startup failed: {e}")
            raise
            
    async def stop_production_services(self):
        """Stop all production services"""
        try:
            logger.info("🛑 Stopping production services")
            
            # Stop monitoring
            if self.monitoring_task:
                await self.performance_monitor.stop_monitoring()
                self.monitoring_task.cancel()
                
            # Stop optimization
            if self.optimization_task:
                self.optimization_task.cancel()
                
            self.is_production_ready = False
            
            logger.info("✅ Production services stopped")
            
        except Exception as e:
            logger.error(f"Production service shutdown failed: {e}")
            
    async def _run_optimization_loop(self):
        """Run continuous optimization loop"""
        try:
            while self.is_production_ready:
                await self.optimizer.run_optimization_cycle()
                await asyncio.sleep(self.optimization_interval)
                
        except asyncio.CancelledError:
            logger.info("Optimization loop cancelled")
        except Exception as e:
            logger.error(f"Optimization loop error: {e}")
            
    @asynccontextmanager
    async def production_request_context(self, request_type: str = "simulation"):
        """Context manager for production request handling"""
        start_time = time.time()
        worker = None
        
        try:
            # Route request to optimal worker
            complexity_score = self._calculate_request_complexity(request_type)
            worker = await self.load_balancer.route_request(request_type, complexity_score)
            
            if not worker:
                logger.warning("No workers available - using fallback")
                
            yield worker
            
            # Record successful request
            response_time = (time.time() - start_time) * 1000  # ms
            self.performance_monitor.record_request(response_time, success=True)
            
            if worker:
                self.load_balancer.complete_request(worker["id"], response_time, success=True)
                
        except Exception as e:
            # Record failed request
            response_time = (time.time() - start_time) * 1000
            self.performance_monitor.record_request(response_time, success=False)
            
            if worker:
                self.load_balancer.complete_request(worker["id"], response_time, success=False)
                
            raise
        finally:
            self.total_requests += 1
            
    def _calculate_request_complexity(self, request_type: str) -> float:
        """Calculate complexity score for request routing"""
        complexity_map = {
            "enhanced_simulation": 0.8,
            "ensemble_prediction": 0.9,
            "standard_simulation": 0.4,
            "sentiment_analysis": 0.6,
            "correlation_analysis": 0.7,
            "portfolio_optimization": 0.8
        }
        
        return complexity_map.get(request_type, 0.5)
        
    async def run_enhanced_simulation_production(
        self,
        news_data: List[str],
        target_assets: List[str] = None,
        prediction_horizon: int = 7,
        use_phase4: bool = True,
        market_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        PHASE 6.1: Production-optimized enhanced simulation
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(news_data, target_assets, prediction_horizon)
            cached_result = self.cache_manager.get(cache_key, "predictions")
            
            if cached_result:
                logger.info("📦 Returning cached prediction result")
                cached_result["from_cache"] = True
                return cached_result
            
            # Run simulation with production context
            async with self.production_request_context("enhanced_simulation") as worker:
                # Add worker information to context
                if worker:
                    if not market_context:
                        market_context = {}
                    market_context["assigned_worker"] = worker
                    
                # Run core simulation
                results = await self.core_simulator.run_enhanced_simulation(
                    news_data=news_data,
                    target_assets=target_assets,
                    prediction_horizon=prediction_horizon,
                    use_phase4=use_phase4,
                    market_context=market_context
                )
                
                # Add production metadata
                results["production_metadata"] = {
                    "worker_assigned": worker["id"] if worker else "fallback",
                    "worker_type": worker["type"] if worker else "none",
                    "request_timestamp": datetime.now().isoformat(),
                    "cache_key": cache_key,
                    "total_requests": self.total_requests
                }
                
                # Cache successful results
                if results.get("predictions"):
                    self.cache_manager.set(cache_key, results, "predictions")
                    
                return results
                
        except Exception as e:
            logger.error(f"Production enhanced simulation failed: {e}")
            
            # Return fallback result
            return await self._production_fallback_simulation(
                news_data, target_assets, prediction_horizon
            )
            
    async def run_standard_simulation_production(
        self,
        news_data: List[str],
        target_assets: List[str] = None,
        quantum_enhanced: bool = True
    ) -> Dict[str, Any]:
        """
        PHASE 6.2: Production-optimized standard simulation
        """
        try:
            # Check cache
            cache_key = self._generate_cache_key(news_data, target_assets, 7, "standard")
            cached_result = self.cache_manager.get(cache_key, "predictions")
            
            if cached_result:
                cached_result["from_cache"] = True
                return cached_result
                
            # Run with production context
            async with self.production_request_context("standard_simulation") as worker:
                results = await self.core_simulator.run_simulation(
                    news_data=news_data,
                    target_assets=target_assets,
                    quantum_enhanced=quantum_enhanced
                )
                
                # Add production metadata
                results["production_metadata"] = {
                    "worker_assigned": worker["id"] if worker else "fallback",
                    "worker_type": worker["type"] if worker else "none",
                    "simulation_type": "standard_production"
                }
                
                # Cache result
                self.cache_manager.set(cache_key, results, "predictions")
                
                return results
                
        except Exception as e:
            logger.error(f"Production standard simulation failed: {e}")
            return await self._production_fallback_simulation(news_data, target_assets, 7)
            
    async def _production_fallback_simulation(
        self, 
        news_data: List[str], 
        target_assets: List[str], 
        prediction_horizon: int
    ) -> Dict[str, Any]:
        """Production fallback when main simulation fails"""
        try:
            logger.warning("🔄 Using production fallback simulation")
            
            if not target_assets:
                target_assets = ["AAPL", "GOOGL", "TSLA", "MSFT", "NVDA"]
                
            # Generate basic predictions
            fallback_predictions = []
            
            for asset in target_assets:
                # Simple market prediction
                base_price = 100.0 + np.random.uniform(-50, 200)
                expected_return = np.random.uniform(-0.05, 0.1)
                
                prediction = {
                    "asset": asset,
                    "predicted_price": base_price * (1 + expected_return),
                    "confidence": 0.6,
                    "fallback": True,
                    "prediction_method": "production_fallback"
                }
                
                fallback_predictions.append(prediction)
                
            return {
                "predictions": fallback_predictions,
                "production_metadata": {
                    "fallback_used": True,
                    "fallback_reason": "main_simulation_failed",
                    "timestamp": datetime.now().isoformat()
                },
                "simulation_type": "fallback"
            }
            
        except Exception as e:
            logger.error(f"Fallback simulation failed: {e}")
            return {
                "predictions": [],
                "error": "all_simulation_methods_failed",
                "timestamp": datetime.now().isoformat()
            }
            
    def _generate_cache_key(self, news_data: List[str], target_assets: List[str], 
                           prediction_horizon: int, sim_type: str = "enhanced") -> str:
        """Generate cache key for requests"""
        try:
            # Create hash from request parameters
            import hashlib
            
            content = f"{sim_type}:{prediction_horizon}:"
            content += ":".join(sorted(target_assets or []))
            content += ":" + ":".join(sorted(news_data[:5]))  # First 5 news items
            
            return hashlib.md5(content.encode()).hexdigest()[:16]
            
        except Exception as e:
            logger.error(f"Cache key generation failed: {e}")
            return f"fallback_{int(time.time())}"
            
    async def get_production_health(self) -> SystemHealth:
        """Get comprehensive production health status"""
        try:
            return await self.health_checker.perform_health_check()
            
        except Exception as e:
            logger.error(f"Production health check failed: {e}")
            return SystemHealth(
                overall_status="error",
                uptime_seconds=0,
                last_health_check=datetime.now(),
                component_status={"health_check": f"failed: {e}"},
                performance_summary={},
                alerts=[f"Health check system failure: {e}"],
                recommendations=["Investigate health monitoring system"]
            )
            
    async def get_production_metrics(self) -> Dict[str, Any]:
        """Get comprehensive production metrics"""
        try:
            # Performance metrics
            performance_summary = self.performance_monitor.get_metrics_summary()
            
            # Cache metrics
            cache_stats = self.cache_manager.get_cache_stats()
            
            # Load balancer metrics
            lb_stats = self.load_balancer.get_load_balancer_stats()
            
            # Optimization metrics
            optimization_summary = self.optimizer.get_optimization_summary()
            
            # Health trend
            health_trend = self.health_checker.get_health_trend()
            
            # Core simulator metrics
            core_metrics = self.core_simulator.get_simulation_metrics()
            
            return {
                "production_status": {
                    "is_ready": self.is_production_ready,
                    "uptime_hours": (datetime.now() - self.startup_time).total_seconds() / 3600 if self.startup_time else 0,
                    "total_requests": self.total_requests
                },
                "performance": performance_summary,
                "caching": cache_stats,
                "load_balancing": lb_stats,
                "optimization": optimization_summary,
                "health_trend": health_trend,
                "core_simulator": core_metrics
            }
            
        except Exception as e:
            logger.error(f"Production metrics collection failed: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
            
    async def optimize_production_performance(self) -> Dict[str, Any]:
        """Manually trigger production optimization"""
        try:
            logger.info("🔧 Manual production optimization triggered")
            
            await self.optimizer.run_optimization_cycle()
            
            return {
                "optimization_completed": True,
                "timestamp": datetime.now().isoformat(),
                "optimization_summary": self.optimizer.get_optimization_summary()
            }
            
        except Exception as e:
            logger.error(f"Manual optimization failed: {e}")
            return {"error": str(e), "optimization_completed": False}
            
    def configure_production_settings(self, settings: Dict[str, Any]):
        """Configure production-specific settings"""
        try:
            # Performance monitor settings
            if "performance_thresholds" in settings:
                self.performance_monitor.performance_thresholds.update(
                    settings["performance_thresholds"]
                )
                
            # Cache settings
            if "cache_max_size" in settings:
                self.cache_manager.max_size = settings["cache_max_size"]
                
            if "cache_ttl_seconds" in settings:
                self.cache_manager.ttl_seconds = settings["cache_ttl_seconds"]
                
            # Load balancer settings
            if "routing_strategy" in settings:
                self.load_balancer.routing_strategy = settings["routing_strategy"]
                
            # Optimization settings
            if "optimization_interval" in settings:
                self.optimization_interval = settings["optimization_interval"]
                
            if "auto_scaling_enabled" in settings:
                self.optimizer.auto_scaling_enabled = settings["auto_scaling_enabled"]
                
            logger.info("⚙️ Production settings updated")
            
        except Exception as e:
            logger.error(f"Production settings configuration failed: {e}")
            
    def get_production_status(self) -> Dict[str, Any]:
        """Get current production status"""
        try:
            return {
                "production_ready": self.is_production_ready,
                "startup_time": self.startup_time.isoformat() if self.startup_time else None,
                "total_requests": self.total_requests,
                "services": {
                    "monitoring": self.monitoring_task is not None and not self.monitoring_task.done(),
                    "optimization": self.optimization_task is not None and not self.optimization_task.done(),
                    "core_simulator": hasattr(self.core_simulator, 'quantum_simulator'),
                    "load_balancer": len(self.load_balancer.quantum_workers + self.load_balancer.classical_workers) > 0
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Production status check failed: {e}")
            return {"error": str(e)}


# Production deployment helpers

async def create_production_simulator(classiq_client=None) -> ProductionMarketSimulator:
    """Create and initialize production market simulator"""
    try:
        logger.info("🏭 Creating production market simulator")
        
        simulator = ProductionMarketSimulator(classiq_client)
        await simulator.start_production_services()
        
        logger.info("✅ Production simulator ready")
        return simulator
        
    except Exception as e:
        logger.error(f"Production simulator creation failed: {e}")
        raise


async def deploy_market_simulator_production():
    """Deploy market simulator in production mode"""
    try:
        logger.info("🚀 DEPLOYING QUANTUM MARKET SIMULATOR - PRODUCTION MODE")
        
        # Create production simulator
        simulator = await create_production_simulator()
        
        # Run initial health check
        health = await simulator.get_production_health()
        logger.info(f"🏥 Initial health status: {health.overall_status}")
        
        # Log production readiness
        status = simulator.get_production_status()
        logger.info(f"📊 Production status: {status}")
        
        return simulator
        
    except Exception as e:
        logger.error(f"Production deployment failed: {e}")
        raise


if __name__ == "__main__":
    # Production deployment example
    async def main():
        try:
            simulator = await deploy_market_simulator_production()
            
            # Run a test simulation
            test_news = [
                "Market shows strong growth potential",
                "Technology sector leads gains",
                "Economic indicators remain positive"
            ]
            
            results = await simulator.run_enhanced_simulation_production(
                news_data=test_news,
                target_assets=["AAPL", "GOOGL", "TSLA"],
                prediction_horizon=7
            )
            
            print("Production simulation results:", results)
            
            # Get production metrics
            metrics = await simulator.get_production_metrics()
            print("Production metrics:", metrics)
            
        except Exception as e:
            logger.error(f"Production test failed: {e}")
        
    asyncio.run(main())

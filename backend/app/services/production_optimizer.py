"""
PHASE 6: PRODUCTION DEPLOYMENT & OPTIMIZATION
Enhanced deployment infrastructure with performance monitoring and optimization
"""

import asyncio
import logging
import time
import psutil
import threading
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import os
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for production monitoring"""
    timestamp: datetime
    response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    active_connections: int
    requests_per_second: float
    error_rate: float
    quantum_circuit_execution_time_ms: float
    prediction_accuracy: float
    cache_hit_rate: float


@dataclass
class SystemHealth:
    """System health status"""
    overall_status: str  # "healthy", "warning", "critical"
    uptime_seconds: float
    last_health_check: datetime
    component_status: Dict[str, str]
    performance_summary: Dict[str, float]
    alerts: List[str]
    recommendations: List[str]


class PerformanceMonitor:
    """
    PHASE 6.1: Real-time performance monitoring and optimization
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.performance_thresholds = {
            "response_time_ms": 5000,  # 5 seconds
            "memory_usage_mb": 2048,   # 2GB
            "cpu_usage_percent": 80,   # 80%
            "error_rate": 0.05,        # 5%
            "cache_hit_rate": 0.8      # 80%
        }
        
        self.start_time = time.time()
        self.request_counter = 0
        self.error_counter = 0
        self.last_metrics_time = time.time()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
        
    async def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous performance monitoring"""
        try:
            self.monitoring_active = True
            logger.info("🔍 Starting production performance monitoring")
            
            while self.monitoring_active:
                await self._collect_system_metrics()
                await asyncio.sleep(interval_seconds)
                
        except Exception as e:
            logger.error(f"Performance monitoring error: {e}")
            
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        logger.info("🛑 Stopping performance monitoring")
        
    async def _collect_system_metrics(self):
        """Collect current system performance metrics"""
        try:
            current_time = time.time()
            
            # System metrics
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Request metrics
            time_diff = current_time - self.last_metrics_time
            rps = self.request_counter / time_diff if time_diff > 0 else 0
            error_rate = self.error_counter / max(self.request_counter, 1)
            
            # Create metrics object
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                response_time_ms=0,  # Will be updated per request
                memory_usage_mb=memory_info.used / (1024 * 1024),
                cpu_usage_percent=cpu_percent,
                active_connections=0,  # Will be updated by connection manager
                requests_per_second=rps,
                error_rate=error_rate,
                quantum_circuit_execution_time_ms=0,  # Updated by quantum components
                prediction_accuracy=0,  # Updated by accuracy tracker
                cache_hit_rate=0  # Updated by cache manager
            )
            
            self.metrics_history.append(metrics)
            
            # Reset counters
            self.request_counter = 0
            self.error_counter = 0
            self.last_metrics_time = current_time
            
            # Check for performance issues
            await self._check_performance_thresholds(metrics)
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            
    async def _check_performance_thresholds(self, metrics: PerformanceMetrics):
        """Check if metrics exceed performance thresholds"""
        alerts = []
        
        if metrics.memory_usage_mb > self.performance_thresholds["memory_usage_mb"]:
            alerts.append(f"High memory usage: {metrics.memory_usage_mb:.0f}MB")
            
        if metrics.cpu_usage_percent > self.performance_thresholds["cpu_usage_percent"]:
            alerts.append(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")
            
        if metrics.error_rate > self.performance_thresholds["error_rate"]:
            alerts.append(f"High error rate: {metrics.error_rate:.2%}")
            
        if alerts:
            logger.warning(f"⚠️ Performance alerts: {'; '.join(alerts)}")
            
    def record_request(self, response_time_ms: float, success: bool = True):
        """Record a request for metrics tracking"""
        self.request_counter += 1
        if not success:
            self.error_counter += 1
            
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recent metrics"""
        if not self.metrics_history:
            return {}
            
        recent_metrics = list(self.metrics_history)
        
        return {
            "avg_response_time_ms": np.mean([m.response_time_ms for m in recent_metrics]),
            "avg_memory_usage_mb": np.mean([m.memory_usage_mb for m in recent_metrics]),
            "avg_cpu_usage_percent": np.mean([m.cpu_usage_percent for m in recent_metrics]),
            "avg_requests_per_second": np.mean([m.requests_per_second for m in recent_metrics]),
            "current_error_rate": recent_metrics[-1].error_rate,
            "uptime_hours": (time.time() - self.start_time) / 3600,
            "total_requests": sum(self.request_counter for _ in recent_metrics),
            "window_size": len(recent_metrics)
        }


class CacheManager:
    """
    PHASE 6.2: Intelligent caching for performance optimization
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
        # Cache storage
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        
        # Cache types
        self.prediction_cache = {}
        self.sentiment_cache = {}
        self.market_data_cache = {}
        
    def get(self, key: str, cache_type: str = "general") -> Optional[Any]:
        """Get item from cache"""
        try:
            cache_dict = self._get_cache_dict(cache_type)
            
            if key in cache_dict:
                item = cache_dict[key]
                
                # Check TTL
                if time.time() - item["timestamp"] <= self.ttl_seconds:
                    self.hit_count += 1
                    self.access_times[key] = time.time()
                    return item["data"]
                else:
                    # Expired item
                    del cache_dict[key]
                    if key in self.access_times:
                        del self.access_times[key]
            
            self.miss_count += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.miss_count += 1
            return None
            
    def set(self, key: str, value: Any, cache_type: str = "general"):
        """Set item in cache"""
        try:
            cache_dict = self._get_cache_dict(cache_type)
            
            # Check if cache is full
            if len(cache_dict) >= self.max_size:
                self._evict_oldest_item(cache_type)
            
            cache_dict[key] = {
                "data": value,
                "timestamp": time.time()
            }
            self.access_times[key] = time.time()
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            
    def _get_cache_dict(self, cache_type: str) -> Dict:
        """Get the appropriate cache dictionary"""
        cache_map = {
            "general": self.cache,
            "predictions": self.prediction_cache,
            "sentiment": self.sentiment_cache,
            "market_data": self.market_data_cache
        }
        return cache_map.get(cache_type, self.cache)
        
    def _evict_oldest_item(self, cache_type: str):
        """Evict the oldest item from cache"""
        cache_dict = self._get_cache_dict(cache_type)
        
        if not cache_dict:
            return
            
        # Find oldest item by access time
        oldest_key = min(cache_dict.keys(), 
                        key=lambda k: self.access_times.get(k, 0))
        
        del cache_dict[oldest_key]
        if oldest_key in self.access_times:
            del self.access_times[oldest_key]
            
    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0.0
        
    def clear_cache(self, cache_type: str = "all"):
        """Clear cache"""
        if cache_type == "all":
            self.cache.clear()
            self.prediction_cache.clear()
            self.sentiment_cache.clear()
            self.market_data_cache.clear()
        else:
            cache_dict = self._get_cache_dict(cache_type)
            cache_dict.clear()
            
        # Clear access times for cleared keys
        self.access_times.clear()
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "hit_rate": self.get_hit_rate(),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "total_items": len(self.cache),
            "prediction_cache_size": len(self.prediction_cache),
            "sentiment_cache_size": len(self.sentiment_cache),
            "market_data_cache_size": len(self.market_data_cache),
            "cache_utilization": (len(self.cache) / self.max_size) * 100
        }


class LoadBalancer:
    """
    PHASE 6.3: Intelligent load balancing for quantum and classical resources
    """
    
    def __init__(self):
        self.quantum_workers = []
        self.classical_workers = []
        self.worker_stats = defaultdict(lambda: {
            "requests_handled": 0,
            "avg_response_time": 0.0,
            "error_count": 0,
            "current_load": 0,
            "health_status": "healthy"
        })
        
        self.routing_strategy = "intelligent"  # "round_robin", "least_connections", "intelligent"
        
    def add_quantum_worker(self, worker_id: str, capacity: int = 100):
        """Add a quantum processing worker"""
        self.quantum_workers.append({
            "id": worker_id,
            "type": "quantum",
            "capacity": capacity,
            "current_load": 0
        })
        logger.info(f"Added quantum worker: {worker_id}")
        
    def add_classical_worker(self, worker_id: str, capacity: int = 200):
        """Add a classical processing worker"""
        self.classical_workers.append({
            "id": worker_id,
            "type": "classical",
            "capacity": capacity,
            "current_load": 0
        })
        logger.info(f"Added classical worker: {worker_id}")
        
    async def route_request(self, request_type: str, complexity_score: float = 0.5) -> Optional[Dict]:
        """Route request to optimal worker"""
        try:
            # Determine if quantum processing is beneficial
            use_quantum = await self._should_use_quantum(request_type, complexity_score)
            
            if use_quantum and self.quantum_workers:
                worker = self._select_quantum_worker()
            else:
                worker = self._select_classical_worker()
                
            if worker:
                self._update_worker_load(worker["id"], 1)
                return worker
                
            return None
            
        except Exception as e:
            logger.error(f"Request routing failed: {e}")
            return None
            
    async def _should_use_quantum(self, request_type: str, complexity_score: float) -> bool:
        """Determine if quantum processing should be used"""
        try:
            # Quantum advantage heuristics
            quantum_beneficial_types = [
                "ensemble_prediction",
                "correlation_analysis", 
                "portfolio_optimization",
                "sentiment_analysis"
            ]
            
            # Use quantum if:
            # 1. Request type benefits from quantum
            # 2. Complexity score is high enough
            # 3. Quantum workers are available and not overloaded
            
            type_benefits = any(qtype in request_type for qtype in quantum_beneficial_types)
            complexity_threshold = complexity_score > 0.6
            quantum_available = any(w["current_load"] < w["capacity"] * 0.8 for w in self.quantum_workers)
            
            return type_benefits and complexity_threshold and quantum_available
            
        except Exception as e:
            logger.error(f"Quantum routing decision failed: {e}")
            return False
            
    def _select_quantum_worker(self) -> Optional[Dict]:
        """Select optimal quantum worker"""
        available_workers = [w for w in self.quantum_workers 
                           if w["current_load"] < w["capacity"]]
        
        if not available_workers:
            return None
            
        if self.routing_strategy == "least_connections":
            return min(available_workers, key=lambda w: w["current_load"])
        elif self.routing_strategy == "intelligent":
            # Consider both load and historical performance
            return min(available_workers, key=lambda w: 
                      w["current_load"] + self.worker_stats[w["id"]]["error_count"])
        else:  # round_robin
            return available_workers[0]
            
    def _select_classical_worker(self) -> Optional[Dict]:
        """Select optimal classical worker"""
        available_workers = [w for w in self.classical_workers 
                           if w["current_load"] < w["capacity"]]
        
        if not available_workers:
            return None
            
        if self.routing_strategy == "least_connections":
            return min(available_workers, key=lambda w: w["current_load"])
        elif self.routing_strategy == "intelligent":
            return min(available_workers, key=lambda w: 
                      w["current_load"] + self.worker_stats[w["id"]]["error_count"])
        else:  # round_robin
            return available_workers[0]
            
    def _update_worker_load(self, worker_id: str, change: int):
        """Update worker load"""
        # Update quantum workers
        for worker in self.quantum_workers:
            if worker["id"] == worker_id:
                worker["current_load"] = max(0, worker["current_load"] + change)
                break
                
        # Update classical workers
        for worker in self.classical_workers:
            if worker["id"] == worker_id:
                worker["current_load"] = max(0, worker["current_load"] + change)
                break
                
    def complete_request(self, worker_id: str, response_time: float, success: bool = True):
        """Mark request as completed"""
        self._update_worker_load(worker_id, -1)
        
        stats = self.worker_stats[worker_id]
        stats["requests_handled"] += 1
        
        # Update average response time
        current_avg = stats["avg_response_time"]
        request_count = stats["requests_handled"]
        stats["avg_response_time"] = (current_avg * (request_count - 1) + response_time) / request_count
        
        if not success:
            stats["error_count"] += 1
            
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        return {
            "quantum_workers": len(self.quantum_workers),
            "classical_workers": len(self.classical_workers),
            "total_capacity": sum(w["capacity"] for w in self.quantum_workers + self.classical_workers),
            "current_load": sum(w["current_load"] for w in self.quantum_workers + self.classical_workers),
            "worker_stats": dict(self.worker_stats),
            "routing_strategy": self.routing_strategy
        }


class ProductionOptimizer:
    """
    PHASE 6.4: Production optimization engine
    """
    
    def __init__(self, performance_monitor: PerformanceMonitor, 
                 cache_manager: CacheManager, 
                 load_balancer: LoadBalancer):
        self.performance_monitor = performance_monitor
        self.cache_manager = cache_manager
        self.load_balancer = load_balancer
        
        # Optimization settings
        self.optimization_enabled = True
        self.auto_scaling_enabled = True
        self.cache_optimization_enabled = True
        
        # Optimization history
        self.optimization_history = deque(maxlen=100)
        
    async def run_optimization_cycle(self):
        """Run a complete optimization cycle"""
        try:
            logger.info("🔧 Running production optimization cycle")
            
            optimization_actions = []
            
            # Get current metrics
            current_metrics = self.performance_monitor.get_current_metrics()
            
            if current_metrics:
                # Memory optimization
                if current_metrics.memory_usage_mb > 1500:  # 1.5GB threshold
                    await self._optimize_memory_usage()
                    optimization_actions.append("memory_optimization")
                
                # Cache optimization
                if current_metrics.response_time_ms > 3000:  # 3 second threshold
                    await self._optimize_cache_strategy()
                    optimization_actions.append("cache_optimization")
                
                # Load balancing optimization
                if current_metrics.error_rate > 0.03:  # 3% error rate
                    await self._optimize_load_balancing()
                    optimization_actions.append("load_balancing_optimization")
                
                # Auto-scaling decisions
                if self.auto_scaling_enabled:
                    scaling_action = await self._auto_scale_resources(current_metrics)
                    if scaling_action:
                        optimization_actions.append(f"auto_scaling_{scaling_action}")
            
            # Record optimization actions
            self.optimization_history.append({
                "timestamp": datetime.now(),
                "actions": optimization_actions,
                "metrics_before": asdict(current_metrics) if current_metrics else {}
            })
            
            logger.info(f"✅ Optimization cycle completed: {len(optimization_actions)} actions taken")
            
        except Exception as e:
            logger.error(f"Optimization cycle failed: {e}")
            
    async def _optimize_memory_usage(self):
        """Optimize memory usage"""
        try:
            # Clear old cache entries
            self.cache_manager.clear_cache("sentiment")  # Clear least critical cache first
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("🧹 Memory optimization completed")
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            
    async def _optimize_cache_strategy(self):
        """Optimize caching strategy"""
        try:
            cache_stats = self.cache_manager.get_cache_stats()
            
            # If hit rate is low, increase cache size
            if cache_stats["hit_rate"] < 0.6:
                self.cache_manager.max_size = min(self.cache_manager.max_size * 1.2, 2000)
                logger.info(f"📈 Increased cache size to {self.cache_manager.max_size}")
            
            # If cache utilization is low, decrease TTL
            elif cache_stats["cache_utilization"] < 50:
                self.cache_manager.ttl_seconds = max(self.cache_manager.ttl_seconds * 0.8, 60)
                logger.info(f"⏰ Decreased cache TTL to {self.cache_manager.ttl_seconds}s")
                
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
            
    async def _optimize_load_balancing(self):
        """Optimize load balancing strategy"""
        try:
            lb_stats = self.load_balancer.get_load_balancer_stats()
            
            # Switch to intelligent routing if error rate is high
            if self.load_balancer.routing_strategy != "intelligent":
                self.load_balancer.routing_strategy = "intelligent"
                logger.info("🧠 Switched to intelligent load balancing")
                
            # Check if we need more workers
            total_load = lb_stats["current_load"]
            total_capacity = lb_stats["total_capacity"]
            
            if total_capacity > 0 and (total_load / total_capacity) > 0.8:
                logger.warning("⚠️ High load detected - consider adding more workers")
                
        except Exception as e:
            logger.error(f"Load balancing optimization failed: {e}")
            
    async def _auto_scale_resources(self, metrics: PerformanceMetrics) -> Optional[str]:
        """Auto-scale resources based on metrics"""
        try:
            # Scale up conditions
            if (metrics.cpu_usage_percent > 75 or 
                metrics.memory_usage_mb > 1800 or 
                metrics.requests_per_second > 100):
                
                # Add classical worker (easier to scale)
                worker_id = f"classical_worker_{int(time.time())}"
                self.load_balancer.add_classical_worker(worker_id, capacity=150)
                logger.info(f"📈 Auto-scaled up: Added {worker_id}")
                return "scale_up"
                
            # Scale down conditions
            elif (metrics.cpu_usage_percent < 30 and 
                  metrics.memory_usage_mb < 800 and 
                  len(self.load_balancer.classical_workers) > 2):
                
                # Remove least utilized classical worker
                least_utilized = min(self.load_balancer.classical_workers, 
                                   key=lambda w: w["current_load"])
                
                if least_utilized["current_load"] == 0:
                    self.load_balancer.classical_workers.remove(least_utilized)
                    logger.info(f"📉 Auto-scaled down: Removed {least_utilized['id']}")
                    return "scale_down"
                    
            return None
            
        except Exception as e:
            logger.error(f"Auto-scaling failed: {e}")
            return None
            
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary"""
        try:
            recent_optimizations = list(self.optimization_history)[-10:]  # Last 10
            
            action_counts = defaultdict(int)
            for opt in recent_optimizations:
                for action in opt["actions"]:
                    action_counts[action] += 1
                    
            return {
                "optimization_enabled": self.optimization_enabled,
                "auto_scaling_enabled": self.auto_scaling_enabled,
                "recent_optimizations": len(recent_optimizations),
                "most_common_actions": dict(action_counts),
                "last_optimization": recent_optimizations[-1]["timestamp"].isoformat() if recent_optimizations else None
            }
            
        except Exception as e:
            logger.error(f"Optimization summary failed: {e}")
            return {"error": str(e)}


class HealthChecker:
    """
    PHASE 6.5: Comprehensive system health monitoring
    """
    
    def __init__(self, components: Dict[str, Any]):
        self.components = components
        self.health_history = deque(maxlen=50)
        self.alert_thresholds = {
            "response_time_ms": 10000,
            "error_rate": 0.1,
            "memory_usage_mb": 3000,
            "cpu_usage_percent": 90
        }
        
    async def perform_health_check(self) -> SystemHealth:
        """Perform comprehensive health check"""
        try:
            start_time = time.time()
            component_status = {}
            alerts = []
            recommendations = []
            
            # Check each component
            for component_name, component in self.components.items():
                try:
                    status = await self._check_component_health(component_name, component)
                    component_status[component_name] = status
                    
                    if status not in ["healthy", "warning"]:
                        alerts.append(f"{component_name}: {status}")
                        
                except Exception as e:
                    component_status[component_name] = f"error: {e}"
                    alerts.append(f"{component_name}: health check failed")
            
            # Overall system assessment
            healthy_components = sum(1 for status in component_status.values() 
                                   if "healthy" in status)
            total_components = len(component_status)
            
            if healthy_components / total_components >= 0.8:
                overall_status = "healthy"
            elif healthy_components / total_components >= 0.6:
                overall_status = "warning"
            else:
                overall_status = "critical"
                
            # Performance summary
            performance_summary = {}
            if hasattr(self.components.get("performance_monitor"), "get_metrics_summary"):
                performance_summary = self.components["performance_monitor"].get_metrics_summary()
                
            # Generate recommendations
            recommendations = self._generate_health_recommendations(
                component_status, performance_summary, alerts
            )
            
            # Create health object
            health = SystemHealth(
                overall_status=overall_status,
                uptime_seconds=time.time() - start_time,
                last_health_check=datetime.now(),
                component_status=component_status,
                performance_summary=performance_summary,
                alerts=alerts,
                recommendations=recommendations
            )
            
            self.health_history.append(health)
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return SystemHealth(
                overall_status="error",
                uptime_seconds=0,
                last_health_check=datetime.now(),
                component_status={"error": str(e)},
                performance_summary={},
                alerts=[f"Health check system error: {e}"],
                recommendations=["Investigate health check system"]
            )
            
    async def _check_component_health(self, component_name: str, component: Any) -> str:
        """Check health of individual component"""
        try:
            # Universal health checks
            if hasattr(component, "health_check"):
                result = await component.health_check()
                if isinstance(result, dict):
                    return result.get("status", "unknown")
                return str(result)
                
            # Component-specific checks
            if component_name == "performance_monitor":
                metrics = component.get_current_metrics()
                if metrics:
                    if metrics.error_rate > 0.05:
                        return "warning_high_error_rate"
                    return "healthy"
                return "no_metrics"
                
            elif component_name == "cache_manager":
                stats = component.get_cache_stats()
                if stats["hit_rate"] < 0.5:
                    return "warning_low_hit_rate"
                return "healthy"
                
            elif component_name == "load_balancer":
                stats = component.get_load_balancer_stats()
                if stats["current_load"] / max(stats["total_capacity"], 1) > 0.9:
                    return "warning_high_load"
                return "healthy"
                
            else:
                # Generic availability check
                return "healthy" if component else "unavailable"
                
        except Exception as e:
            return f"error: {e}"
            
    def _generate_health_recommendations(self, 
                                       component_status: Dict[str, str], 
                                       performance_summary: Dict[str, Any],
                                       alerts: List[str]) -> List[str]:
        """Generate health-based recommendations"""
        recommendations = []
        
        try:
            # Component-specific recommendations
            for component, status in component_status.items():
                if "warning" in status:
                    if "cache" in component and "low_hit_rate" in status:
                        recommendations.append("Consider increasing cache size or adjusting TTL")
                    elif "load_balancer" in component and "high_load" in status:
                        recommendations.append("Scale up workers or optimize load distribution")
                    elif "performance_monitor" in component and "high_error_rate" in status:
                        recommendations.append("Investigate error sources and improve error handling")
                        
                elif "error" in status:
                    recommendations.append(f"Urgent: Fix {component} component errors")
            
            # Performance-based recommendations
            if performance_summary:
                if performance_summary.get("avg_memory_usage_mb", 0) > 1500:
                    recommendations.append("Monitor memory usage and consider memory optimization")
                    
                if performance_summary.get("avg_response_time_ms", 0) > 3000:
                    recommendations.append("Optimize response times through caching or code optimization")
                    
                if performance_summary.get("current_error_rate", 0) > 0.03:
                    recommendations.append("Investigate and fix sources of errors")
            
            # General recommendations
            if len(alerts) > 5:
                recommendations.append("Multiple alerts detected - consider comprehensive system review")
                
            if not recommendations:
                recommendations.append("System is operating within normal parameters")
                
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            recommendations.append("Unable to generate recommendations due to analysis error")
            
        return recommendations
        
    def get_health_trend(self) -> Dict[str, Any]:
        """Get health trend analysis"""
        try:
            if len(self.health_history) < 2:
                return {"insufficient_data": True}
                
            recent_health = list(self.health_history)
            
            # Calculate trends
            status_counts = defaultdict(int)
            alert_counts = defaultdict(int)
            
            for health in recent_health:
                status_counts[health.overall_status] += 1
                alert_counts[len(health.alerts)] += 1
                
            return {
                "health_checks_performed": len(recent_health),
                "status_distribution": dict(status_counts),
                "avg_alerts_per_check": np.mean([len(h.alerts) for h in recent_health]),
                "trend": "improving" if recent_health[-1].overall_status == "healthy" else "stable"
            }
            
        except Exception as e:
            logger.error(f"Health trend analysis failed: {e}")
            return {"error": str(e)}

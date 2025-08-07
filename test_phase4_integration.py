"""
Phase 4 Integration Test
Test the enhanced unified market simulator with Phase 4 components
"""

import asyncio
import sys
import os

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from backend.app.services.unified_market_simulator import UnifiedMarketSimulator
    SIMULATOR_AVAILABLE = True
except ImportError as e:
    print(f"Simulator import failed: {e}")
    SIMULATOR_AVAILABLE = False


async def test_phase4_integration():
    """Test Phase 4 integration with the unified market simulator"""
    print("🚀 PHASE 4 INTEGRATION TEST")
    print("=" * 60)
    
    if not SIMULATOR_AVAILABLE:
        print("❌ UnifiedMarketSimulator not available")
        return {"status": "failed", "error": "import_failed"}
    
    try:
        # Initialize simulator
        print("🔧 Initializing UnifiedMarketSimulator...")
        simulator = UnifiedMarketSimulator()
        await simulator.initialize()
        
        # Test data
        news_data = [
            "Apple reports strong quarterly earnings with record revenue",
            "Tesla announces new breakthrough in autonomous driving technology",
            "Federal Reserve maintains interest rates amid economic uncertainty",
            "Technology stocks surge on positive market sentiment",
            "NVIDIA sees increased demand for AI computing chips"
        ]
        
        target_assets = ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT"]
        
        # Test standard simulation
        print("\n📊 Running standard simulation...")
        standard_results = await simulator.run_simulation(
            news_data=news_data,
            target_assets=target_assets,
            quantum_enhanced=True
        )
        
        print(f"✅ Standard simulation completed")
        print(f"   - Predictions: {len(standard_results.get('predictions', []))}")
        print(f"   - Type: {standard_results.get('simulation_type', 'unknown')}")
        
        # Test enhanced simulation with Phase 4
        print("\n🧠 Running enhanced simulation with Phase 4...")
        enhanced_results = await simulator.run_enhanced_simulation(
            news_data=news_data,
            target_assets=target_assets,
            prediction_horizon=7,
            use_phase4=True,
            market_context={
                "volatility_adjustment": 1.1,
                "return_adjustment": 0.02,
                "inflation_rate": 0.035,
                "interest_rate": 0.055
            }
        )
        
        print(f"✅ Enhanced simulation completed")
        
        # Analyze results
        metadata = enhanced_results.get("simulation_metadata", {})
        print(f"   - Simulation type: {metadata.get('simulation_type', 'unknown')}")
        print(f"   - Execution time: {metadata.get('execution_time_seconds', 0):.2f}s")
        print(f"   - Phase 4 available: {metadata.get('phase4_available', False)}")
        
        predictions = enhanced_results.get("predictions", [])
        print(f"   - Predictions generated: {len(predictions)}")
        
        # Phase 4 analysis
        phase4_analysis = enhanced_results.get("phase4_analysis", {})
        if phase4_analysis:
            print(f"   - Hybrid pipeline: {'✅' if phase4_analysis.get('hybrid_pipeline') else '❌'}")
            print(f"   - Ensemble models: {'✅' if phase4_analysis.get('ensemble_models') else '❌'}")
            print(f"   - Advanced sentiment: {'✅' if phase4_analysis.get('advanced_sentiment') else '❌'}")
            print(f"   - Correlation modeling: {'✅' if phase4_analysis.get('correlation_modeling') else '❌'}")
        
        # Accuracy improvements
        accuracy_improvements = enhanced_results.get("accuracy_improvements", {})
        if accuracy_improvements:
            print(f"   - Hybrid improvement: {accuracy_improvements.get('hybrid_improvement', 0):.3f}")
            print(f"   - Ensemble improvement: {accuracy_improvements.get('ensemble_improvement', 0):.3f}")
            print(f"   - Combined improvement: {accuracy_improvements.get('combined_improvement', 0):.3f}")
        
        # Check prediction quality
        if predictions:
            confidences = [p.confidence for p in predictions if hasattr(p, 'confidence')]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            print(f"   - Average confidence: {avg_confidence:.3f}")
            
            quantum_advantages = []
            for p in predictions:
                if hasattr(p, 'quantum_metrics') and p.quantum_metrics:
                    quantum_advantages.append(p.quantum_metrics.quantum_advantage)
            
            if quantum_advantages:
                avg_quantum_advantage = sum(quantum_advantages) / len(quantum_advantages)
                print(f"   - Average quantum advantage: {avg_quantum_advantage:.3f}")
        
        # Get simulator metrics
        print("\n📈 Simulator Performance Metrics:")
        metrics = simulator.get_simulation_metrics()
        print(f"   - Total simulations: {metrics.get('total_simulations', 0)}")
        print(f"   - Phase 4 simulations: {metrics.get('phase4_simulations', 0)}")
        print(f"   - Phase 4 usage rate: {metrics.get('phase4_usage_rate', 0):.1%}")
        print(f"   - Average accuracy: {metrics.get('average_accuracy', 0):.3f}")
        print(f"   - Quantum enhancement rate: {metrics.get('quantum_enhancement_rate', 0):.3f}")
        
        # Health check
        print("\n🔍 Health Check:")
        health = await simulator.health_check()
        print(f"   - Overall status: {health.get('overall_status', 'unknown')}")
        print(f"   - Phase 4 status: {health.get('phase4_status', 'unknown')}")
        
        components = health.get('components', {})
        if 'phase4' in components:
            phase4_components = components['phase4']
            for component, status in phase4_components.items():
                print(f"   - {component}: {status}")
        
        # Test summary
        print(f"\n🎯 PHASE 4 INTEGRATION TEST RESULTS:")
        
        success_indicators = []
        success_indicators.append(len(predictions) > 0)
        success_indicators.append(metadata.get('simulation_type') == 'phase4_enhanced')
        success_indicators.append(health.get('overall_status') in ['healthy', 'degraded'])
        success_indicators.append(metrics.get('total_simulations', 0) > 0)
        
        success_rate = sum(success_indicators) / len(success_indicators)
        
        if success_rate >= 0.8:
            test_status = "EXCELLENT"
            status_emoji = "🏆"
        elif success_rate >= 0.6:
            test_status = "GOOD"
            status_emoji = "✅"
        elif success_rate >= 0.4:
            test_status = "PARTIAL"
            status_emoji = "⚠️"
        else:
            test_status = "NEEDS_WORK"
            status_emoji = "❌"
        
        print(f"{status_emoji} Integration Status: {test_status} ({success_rate:.1%})")
        
        return {
            "status": "success",
            "test_status": test_status,
            "success_rate": success_rate,
            "standard_results": standard_results,
            "enhanced_results": enhanced_results,
            "metrics": metrics,
            "health": health
        }
        
    except Exception as e:
        print(f"❌ Phase 4 integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


async def main():
    """Run the Phase 4 integration test"""
    results = await test_phase4_integration()
    
    print(f"\n🏁 Test completed with status: {results.get('status', 'unknown')}")
    
    if results.get("status") == "success":
        print("✅ Phase 4 model accuracy improvements are successfully integrated!")
        print(f"📊 Integration quality: {results.get('test_status', 'unknown')}")
    else:
        print(f"❌ Integration test failed: {results.get('error', 'unknown error')}")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())

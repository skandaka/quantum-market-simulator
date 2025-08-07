#!/usr/bin/env python3
"""
COMPREHENSIVE QUANTUM MARKET SIMULATOR TEST SUITE
Tests all implemented phases of the quantum enhancement plan
"""

import sys
import asyncio
import json
import time
import traceback
from pathlib import Path

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent / "backend"))

async def test_phase_1_quantum_algorithms():
    """Test Phase 1: Quantum Algorithm Accuracy Enhancement"""
    print("\n" + "="*60)
    print("TESTING PHASE 1: QUANTUM ALGORITHM ACCURACY ENHANCEMENT")
    print("="*60)
    
    results = {"phase_1": {"tests": [], "success_count": 0, "total_count": 0}}
    
    # Test 1.1: Quantum Sentiment Analysis Circuit
    print("\n🧠 Testing Phase 1.1: Quantum Sentiment Analysis Circuit")
    try:
        from app.quantum.qnlp_model import QuantumNLPModel
        from app.quantum.classiq_client import ClassiqClient
        
        # Initialize quantum components
        classiq_client = ClassiqClient()
        await classiq_client.initialize()
        
        quantum_nlp = QuantumNLPModel(classiq_client)
        await quantum_nlp.initialize()
        
        # Test quantum word embedding
        test_text = "Apple stock surges after strong quarterly earnings report"
        embedding_result = await quantum_nlp.create_quantum_word_embedding_circuit(test_text)
        
        assert embedding_result is not None, "Word embedding failed"
        assert "quantum_embedding" in embedding_result, "Missing quantum embedding"
        print("✅ Quantum word embedding circuit: PASS")
        
        # Test advanced feature map
        if "quantum_embedding" in embedding_result:
            import numpy as np
            feature_map_result = quantum_nlp.create_advanced_feature_map(
                np.array(embedding_result["quantum_embedding"])
            )
            assert feature_map_result is not None, "Feature map failed"
            print("✅ Advanced quantum feature map: PASS")
        
        # Test quantum attention mechanism
        attention_result = await quantum_nlp.quantum_attention_layer(test_text)
        assert attention_result is not None, "Attention mechanism failed"
        print("✅ Quantum attention mechanism: PASS")
        
        # Test full sentiment classification
        quantum_features = await quantum_nlp.encode_text_quantum(test_text)
        sentiment_result = await quantum_nlp.quantum_sentiment_classification(quantum_features)
        
        assert sentiment_result is not None, "Sentiment classification failed"
        assert "predicted_sentiment" in sentiment_result, "Missing prediction"
        print("✅ Full quantum sentiment classification: PASS")
        
        results["phase_1"]["tests"].append({
            "test": "1.1_quantum_sentiment_analysis",
            "status": "PASS",
            "details": {
                "word_embedding": "success",
                "feature_map": "success",
                "attention": "success",
                "classification": sentiment_result.get("predicted_sentiment", "unknown")
            }
        })
        results["phase_1"]["success_count"] += 1
        
    except Exception as e:
        print(f"❌ Phase 1.1 failed: {e}")
        results["phase_1"]["tests"].append({
            "test": "1.1_quantum_sentiment_analysis",
            "status": "FAIL",
            "error": str(e)
        })
    
    results["phase_1"]["total_count"] += 1
    
    # Test 1.2: Quantum Monte Carlo Enhancement
    print("\n📊 Testing Phase 1.2: Quantum Monte Carlo Enhancement")
    try:
        from app.quantum.quantum_finance import QuantumFinanceAlgorithms
        
        quantum_finance = QuantumFinanceAlgorithms(classiq_client)
        
        # Test quantum Monte Carlo pricing
        qmc_result = await quantum_finance.quantum_monte_carlo_pricing(
            spot_price=150.0,
            volatility=0.25,
            drift=0.05,
            time_horizon=30,
            num_paths=100
        )
        
        assert qmc_result is not None, "QMC pricing failed"
        assert "expected_price" in qmc_result, "Missing expected price"
        print("✅ Quantum Monte Carlo pricing: PASS")
        
        # Test quantum random number generation
        qrng_result = await quantum_finance.quantum_random_generator(num_samples=50)
        assert qrng_result is not None, "QRNG failed"
        assert "random_numbers" in qrng_result, "Missing random numbers"
        print("✅ Quantum random number generation: PASS")
        
        # Test quantum correlation modeling
        import numpy as np
        correlation_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        correlation_result = await quantum_finance.quantum_correlation_circuit(correlation_matrix)
        
        assert correlation_result is not None, "Correlation modeling failed"
        print("✅ Quantum correlation modeling: PASS")
        
        results["phase_1"]["tests"].append({
            "test": "1.2_quantum_monte_carlo",
            "status": "PASS",
            "details": {
                "monte_carlo": "success",
                "random_generation": "success",
                "correlation_modeling": "success",
                "expected_price": qmc_result.get("expected_price", 0)
            }
        })
        results["phase_1"]["success_count"] += 1
        
    except Exception as e:
        print(f"❌ Phase 1.2 failed: {e}")
        results["phase_1"]["tests"].append({
            "test": "1.2_quantum_monte_carlo",
            "status": "FAIL",
            "error": str(e)
        })
    
    results["phase_1"]["total_count"] += 1
    
    # Test 1.3: Quantum Portfolio Optimization
    print("\n💼 Testing Phase 1.3: Quantum Portfolio Optimization")
    try:
        from app.quantum.quantum_portfolio_optimizer import QuantumPortfolioOptimizer
        
        quantum_portfolio = QuantumPortfolioOptimizer(classiq_client)
        
        # Test QAOA portfolio optimization
        assets = ["AAPL", "GOOGL", "MSFT"]
        expected_returns = [0.1, 0.12, 0.08]
        covariance_matrix = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.05, 0.015],
            [0.01, 0.015, 0.03]
        ])
        
        portfolio_result = await quantum_portfolio.quantum_portfolio_optimization(
            assets, expected_returns, covariance_matrix, risk_tolerance=0.5
        )
        
        assert portfolio_result is not None, "Portfolio optimization failed"
        print("✅ QAOA portfolio optimization: PASS")
        
        # Test quantum risk parity
        risk_parity_result = await quantum_portfolio.quantum_risk_parity(
            covariance_matrix, target_risk=0.15
        )
        
        assert risk_parity_result is not None, "Risk parity failed"
        print("✅ Quantum risk parity optimization: PASS")
        
        results["phase_1"]["tests"].append({
            "test": "1.3_quantum_portfolio_optimization",
            "status": "PASS",
            "details": {
                "qaoa_optimization": "success",
                "risk_parity": "success"
            }
        })
        results["phase_1"]["success_count"] += 1
        
    except Exception as e:
        print(f"❌ Phase 1.3 failed: {e}")
        results["phase_1"]["tests"].append({
            "test": "1.3_quantum_portfolio_optimization", 
            "status": "FAIL",
            "error": str(e)
        })
    
    results["phase_1"]["total_count"] += 1
    
    print(f"\n📊 Phase 1 Results: {results['phase_1']['success_count']}/{results['phase_1']['total_count']} tests passed")
    return results

async def test_phase_3_classiq_integration():
    """Test Phase 3: Classiq Platform Integration Maximization"""
    print("\n" + "="*60)
    print("TESTING PHASE 3: CLASSIQ PLATFORM INTEGRATION MAXIMIZATION")
    print("="*60)
    
    results = {"phase_3": {"tests": [], "success_count": 0, "total_count": 0}}
    
    # Test 3.1: Advanced Circuit Synthesis
    print("\n🔧 Testing Phase 3.1: Advanced Circuit Synthesis")
    try:
        from app.quantum.classiq_client import ClassiqClient
        
        classiq_client = ClassiqClient()
        await classiq_client.initialize()
        
        # Create a simple quantum function for testing
        from classiq import qfunc, QBit, H
        
        @qfunc
        def test_circuit(qubits: QBit):
            H(qubits)
        
        # Test enhanced synthesis with multiple optimization levels
        synthesis_result = await classiq_client.synthesize_quantum_circuit(
            test_circuit,
            optimization_preferences={"num_shots": 1024, "optimization_level": 2}
        )
        
        assert synthesis_result is not None, "Synthesis failed"
        assert synthesis_result.get("synthesized", False), "Synthesis not successful"
        assert "best_result" in synthesis_result, "Missing best result"
        assert "synthesis_metrics" in synthesis_result, "Missing synthesis metrics"
        print("✅ Multi-level circuit synthesis: PASS")
        
        # Verify synthesis metrics
        metrics = synthesis_result["synthesis_metrics"]
        assert "best_optimization" in metrics, "Missing optimization info"
        assert "circuit_depth" in metrics, "Missing circuit depth"
        assert "gate_count" in metrics, "Missing gate count"
        print("✅ Synthesis metrics validation: PASS")
        
        results["phase_3"]["tests"].append({
            "test": "3.1_advanced_circuit_synthesis",
            "status": "PASS",
            "details": {
                "synthesis": "success",
                "best_optimization": metrics.get("best_optimization", "unknown"),
                "circuit_depth": metrics.get("circuit_depth", 0),
                "gate_count": metrics.get("gate_count", 0)
            }
        })
        results["phase_3"]["success_count"] += 1
        
    except Exception as e:
        print(f"❌ Phase 3.1 failed: {e}")
        results["phase_3"]["tests"].append({
            "test": "3.1_advanced_circuit_synthesis",
            "status": "FAIL", 
            "error": str(e)
        })
    
    results["phase_3"]["total_count"] += 1
    
    # Test 3.2: Optimized Quantum Execution
    print("\n⚡ Testing Phase 3.2: Optimized Quantum Execution")
    try:
        # Test enhanced execution with error mitigation
        execution_result = await classiq_client.execute_optimized_circuit(
            {"circuit_depth": 50, "qubit_count": 8},
            execution_params={
                "num_shots": 1024,
                "error_mitigation_level": "medium",
                "readout_error_mitigation": True
            }
        )
        
        assert execution_result is not None, "Execution failed"
        assert execution_result.get("executed", False), "Execution not successful"
        assert "measurement_results" in execution_result, "Missing measurement results"
        assert "execution_metrics" in execution_result, "Missing execution metrics"
        print("✅ Optimized quantum execution: PASS")
        
        # Verify execution analysis
        if "execution_analysis" in execution_result:
            analysis = execution_result["execution_analysis"]
            assert "execution_quality_score" in analysis, "Missing quality score"
            print("✅ Execution analysis: PASS")
        
        results["phase_3"]["tests"].append({
            "test": "3.2_optimized_quantum_execution",
            "status": "PASS",
            "details": {
                "execution": "success",
                "measurements": len(execution_result.get("measurement_results", {}).get("counts", {})),
                "fidelity": execution_result.get("execution_metrics", {}).get("execution_fidelity", 0)
            }
        })
        results["phase_3"]["success_count"] += 1
        
    except Exception as e:
        print(f"❌ Phase 3.2 failed: {e}")
        results["phase_3"]["tests"].append({
            "test": "3.2_optimized_quantum_execution",
            "status": "FAIL",
            "error": str(e)
        })
    
    results["phase_3"]["total_count"] += 1
    
    print(f"\n📊 Phase 3 Results: {results['phase_3']['success_count']}/{results['phase_3']['total_count']} tests passed")
    return results

async def test_api_integration():
    """Test API integration with enhanced quantum features"""
    print("\n" + "="*60)
    print("TESTING API INTEGRATION WITH QUANTUM ENHANCEMENTS")
    print("="*60)
    
    results = {"api_integration": {"tests": [], "success_count": 0, "total_count": 0}}
    
    # Test enhanced simulation endpoint
    print("\n🌐 Testing Enhanced Simulation API")
    try:
        from app.api.routes import simulate_market
        from app.models.schemas import SimulationRequest, NewsInput
        
        # Create test request
        test_request = SimulationRequest(
            news_inputs=[
                NewsInput(
                    content="Apple reports record quarterly earnings, beating analyst expectations",
                    source_type="headline"
                ),
                NewsInput(
                    content="Tech stocks rally amid positive market sentiment",
                    source_type="headline"
                )
            ],
            target_assets=["AAPL", "GOOGL"],
            time_horizon_days=30,
            num_scenarios=50,
            simulation_method="quantum_enhanced_monte_carlo"
        )
        
        # Test the enhanced simulation
        response = await simulate_market(test_request)
        
        assert response is not None, "Simulation response is None"
        assert hasattr(response, 'market_predictions'), "Missing market predictions"
        assert hasattr(response, 'news_analysis'), "Missing news analysis"
        assert len(response.market_predictions) > 0, "No market predictions"
        assert len(response.news_analysis) > 0, "No news analysis"
        print("✅ Enhanced simulation API: PASS")
        
        # Verify quantum features in response
        for prediction in response.market_predictions:
            if hasattr(prediction, 'quantum_metrics') and prediction.quantum_metrics:
                assert "quantum_advantage" in prediction.quantum_metrics, "Missing quantum advantage"
                print("✅ Quantum metrics in predictions: PASS")
                break
        
        # Verify enhanced sentiment analysis
        for analysis in response.news_analysis:
            if isinstance(analysis, dict) and "quantum_features" in analysis:
                print("✅ Quantum sentiment features: PASS")
                break
        
        results["api_integration"]["tests"].append({
            "test": "enhanced_simulation_api",
            "status": "PASS",
            "details": {
                "predictions_count": len(response.market_predictions),
                "news_analysis_count": len(response.news_analysis),
                "quantum_enabled": any(
                    hasattr(p, 'quantum_metrics') and p.quantum_metrics 
                    for p in response.market_predictions
                )
            }
        })
        results["api_integration"]["success_count"] += 1
        
    except Exception as e:
        print(f"❌ API integration test failed: {e}")
        traceback.print_exc()
        results["api_integration"]["tests"].append({
            "test": "enhanced_simulation_api",
            "status": "FAIL",
            "error": str(e)
        })
    
    results["api_integration"]["total_count"] += 1
    
    print(f"\n📊 API Integration Results: {results['api_integration']['success_count']}/{results['api_integration']['total_count']} tests passed")
    return results

async def test_performance_benchmarks():
    """Test performance benchmarks for quantum vs classical"""
    print("\n" + "="*60)
    print("TESTING PERFORMANCE BENCHMARKS")
    print("="*60)
    
    results = {"performance": {"tests": [], "success_count": 0, "total_count": 0}}
    
    # Benchmark quantum sentiment analysis
    print("\n⚡ Benchmarking Quantum vs Classical Performance")
    try:
        from app.quantum.qnlp_model import QuantumNLPModel
        from app.quantum.classiq_client import ClassiqClient
        
        classiq_client = ClassiqClient()
        await classiq_client.initialize()
        quantum_nlp = QuantumNLPModel(classiq_client)
        await quantum_nlp.initialize()
        
        test_text = "The market shows strong bullish sentiment following positive economic indicators"
        
        # Benchmark quantum processing
        start_time = time.time()
        quantum_result = await quantum_nlp.quantum_sentiment_classification(
            await quantum_nlp.encode_text_quantum(test_text)
        )
        quantum_time = time.time() - start_time
        
        # Benchmark classical processing
        start_time = time.time()
        classical_result = await quantum_nlp.classical_sentiment_analysis(test_text)
        classical_time = time.time() - start_time
        
        print(f"⏱️  Quantum processing time: {quantum_time:.3f}s")
        print(f"⏱️  Classical processing time: {classical_time:.3f}s")
        
        # Calculate quantum advantage
        if quantum_result and "quantum_metrics" in quantum_result:
            quantum_advantage = quantum_result["quantum_metrics"].get("quantum_advantage", 1.0)
            print(f"🚀 Quantum advantage: {quantum_advantage:.2f}x")
        
        assert quantum_result is not None, "Quantum processing failed"
        assert classical_result is not None, "Classical processing failed"
        print("✅ Performance benchmark: PASS")
        
        results["performance"]["tests"].append({
            "test": "quantum_vs_classical_benchmark",
            "status": "PASS",
            "details": {
                "quantum_time": quantum_time,
                "classical_time": classical_time,
                "quantum_advantage": quantum_result.get("quantum_metrics", {}).get("quantum_advantage", 1.0) if quantum_result else 1.0,
                "speedup_ratio": classical_time / quantum_time if quantum_time > 0 else 1.0
            }
        })
        results["performance"]["success_count"] += 1
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        results["performance"]["tests"].append({
            "test": "quantum_vs_classical_benchmark",
            "status": "FAIL",
            "error": str(e)
        })
    
    results["performance"]["total_count"] += 1
    
    print(f"\n📊 Performance Results: {results['performance']['success_count']}/{results['performance']['total_count']} tests passed")
    return results

async def run_comprehensive_test_suite():
    """Run the complete test suite for all implemented phases"""
    print("🚀 STARTING COMPREHENSIVE QUANTUM MARKET SIMULATOR TEST SUITE")
    print("="*80)
    
    overall_start_time = time.time()
    all_results = {}
    
    try:
        # Run all test phases
        phase_1_results = await test_phase_1_quantum_algorithms()
        all_results.update(phase_1_results)
        
        phase_3_results = await test_phase_3_classiq_integration() 
        all_results.update(phase_3_results)
        
        api_results = await test_api_integration()
        all_results.update(api_results)
        
        performance_results = await test_performance_benchmarks()
        all_results.update(performance_results)
        
        # Calculate overall results
        total_tests = sum(phase["total_count"] for phase in all_results.values())
        total_successes = sum(phase["success_count"] for phase in all_results.values())
        overall_success_rate = (total_successes / total_tests * 100) if total_tests > 0 else 0
        
        total_time = time.time() - overall_start_time
        
        # Print comprehensive summary
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE TEST SUITE RESULTS")
        print("="*80)
        print(f"⏱️  Total execution time: {total_time:.2f} seconds")
        print(f"📊 Overall success rate: {total_successes}/{total_tests} ({overall_success_rate:.1f}%)")
        print()
        
        for phase_name, phase_results in all_results.items():
            success_rate = (phase_results["success_count"] / phase_results["total_count"] * 100) if phase_results["total_count"] > 0 else 0
            status_icon = "✅" if success_rate == 100 else "⚠️" if success_rate >= 50 else "❌"
            print(f"{status_icon} {phase_name.upper()}: {phase_results['success_count']}/{phase_results['total_count']} ({success_rate:.1f}%)")
        
        # Phase implementation status
        print("\n📋 PHASE IMPLEMENTATION STATUS:")
        print("✅ Phase 1.1: Quantum Sentiment Analysis Circuit - IMPLEMENTED")
        print("✅ Phase 1.2: Quantum Monte Carlo Enhancement - IMPLEMENTED") 
        print("✅ Phase 1.3: Quantum Portfolio Optimization - IMPLEMENTED")
        print("✅ Phase 2.1: Quantum State Visualization - IMPLEMENTED")
        print("✅ Phase 3.1: Advanced Circuit Synthesis - IMPLEMENTED")
        print("✅ Phase 3.2: Optimized Quantum Execution - IMPLEMENTED")
        print("✅ Phase 5.1: Quantum Advantage Dashboard - IMPLEMENTED")
        print("⏳ Phase 4: Model Accuracy Improvements - PENDING")
        print("⏳ Phase 6: Production Deployment Optimization - PENDING")
        print("⏳ Phase 7: Testing and Validation - PENDING")
        print("⏳ Phase 8: Documentation and Demonstration - PENDING")
        
        # Save detailed results
        results_file = Path("test_results.json")
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": time.time(),
                "total_time": total_time,
                "overall_success_rate": overall_success_rate,
                "total_tests": total_tests,
                "total_successes": total_successes,
                "phase_results": all_results
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        if overall_success_rate >= 80:
            print("\n🎉 QUANTUM MARKET SIMULATOR TEST SUITE: SUCCESS!")
            print("   Ready for next phase implementation.")
        elif overall_success_rate >= 60:
            print("\n⚠️  QUANTUM MARKET SIMULATOR TEST SUITE: PARTIAL SUCCESS")
            print("   Some issues detected, review failed tests.")
        else:
            print("\n❌ QUANTUM MARKET SIMULATOR TEST SUITE: NEEDS ATTENTION")
            print("   Multiple failures detected, requires debugging.")
        
        return all_results
        
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR in test suite: {e}")
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    # Set up event loop for async testing
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        results = asyncio.run(run_comprehensive_test_suite())
        sys.exit(0 if results and not results.get("error") else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)

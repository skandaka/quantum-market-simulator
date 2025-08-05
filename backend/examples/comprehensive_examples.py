"""
Comprehensive Quantum Market Simulation Examples
Demonstrates quantum features for market analysis and prediction
"""

import asyncio
import numpy as np
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.models.schemas import SentimentAnalysis, SentimentType
from app.services.unified_market_simulator import UnifiedMarketSimulator


async def basic_simulation_example():
    """Basic market simulation example"""
    print("🚀 Basic Market Simulation Example")
    print("=" * 50)
    
    # Initialize simulator
    simulator = UnifiedMarketSimulator()
    await simulator.initialize()
    
    # Sample sentiment data
    sentiment_results = [
        SentimentAnalysis(
            text="Apple reports strong quarterly earnings with iPhone sales exceeding expectations",
            sentiment=SentimentType.POSITIVE,
            confidence=0.85,
            entities_detected=[{"text": "Apple", "label": "ORG", "confidence": 0.95}],
            key_phrases=["strong quarterly earnings", "iPhone sales", "exceeding expectations"],
            market_impact_keywords=["earnings", "sales"],
            quantum_sentiment_vector=[0.1, 0.2, 0.4, 0.25, 0.05]  # [very_neg, neg, neutral, pos, very_pos]
        )
    ]
    
    # Market data
    market_data = {
        "AAPL": {
            "current_price": 150.0,
            "volume": 50000000,
            "volatility": 0.25
        }
    }
    
    # Simulation parameters
    simulation_params = {
        "target_assets": ["AAPL"],
        "method": "hybrid_qml",
        "time_horizon": 7,
        "num_scenarios": 100
    }
    
    # Run simulation
    predictions = await simulator.simulate(
        sentiment_results=sentiment_results,
        market_data=market_data,
        simulation_params=simulation_params
    )
    
    # Display results
    for prediction in predictions:
        print(f"\n📊 Prediction for {prediction.asset}:")
        print(f"   Current Price: ${prediction.current_price:.2f}")
        print(f"   Expected Return: {prediction.expected_return:.2%}")
        print(f"   Volatility: {prediction.volatility:.2%}")
        print(f"   Quantum Uncertainty: {prediction.quantum_uncertainty:.3f}")
        
        print(f"   Regime Probabilities:")
        for regime, prob in prediction.regime_probabilities.items():
            print(f"     {regime.title()}: {prob:.1%}")
    
    await simulator.cleanup()


async def quantum_vs_classical_comparison():
    """Compare quantum vs classical prediction methods"""
    print("\n🔬 Quantum vs Classical Comparison")
    print("=" * 50)
    
    simulator = UnifiedMarketSimulator()
    await simulator.initialize()
    
    # Sample data
    sentiment_results = [
        SentimentAnalysis(
            text="Market volatility increases amid economic uncertainty",
            sentiment=SentimentType.NEGATIVE,
            confidence=0.75,
            entities_detected=[{"text": "market", "label": "MISC", "confidence": 0.8}],
            key_phrases=["market volatility", "economic uncertainty"],
            market_impact_keywords=["volatility", "uncertainty"],
            quantum_sentiment_vector=[0.05, 0.35, 0.4, 0.15, 0.05]
        )
    ]
    
    market_data = {
        "TSLA": {"current_price": 200.0, "volume": 30000000, "volatility": 0.4}
    }
    
    # Classical simulation
    classical_params = {
        "target_assets": ["TSLA"],
        "method": "classical",
        "time_horizon": 7,
        "num_scenarios": 100
    }
    
    classical_predictions = await simulator.simulate(
        sentiment_results=sentiment_results,
        market_data=market_data,
        simulation_params=classical_params
    )
    
    # Quantum simulation
    quantum_params = classical_params.copy()
    quantum_params["method"] = "quantum"
    
    quantum_predictions = await simulator.simulate(
        sentiment_results=sentiment_results,
        market_data=market_data,
        simulation_params=quantum_params
    )
    
    # Compare results
    comparison = simulator.compare_methods(quantum_predictions, classical_predictions)
    
    print(f"\n📈 Comparison Results:")
    print(f"   Classical Return: {comparison.get('classical_avg_return', 0):.2%}")
    print(f"   Quantum Return: {comparison.get('quantum_avg_return', 0):.2%}")
    print(f"   Return Improvement: {comparison.get('return_improvement', 0):.2%}")
    print(f"   Volatility Reduction: {comparison.get('volatility_reduction', 0):.2%}")
    
    await simulator.cleanup()


async def portfolio_optimization_example():
    """Portfolio optimization example"""
    print("\n💼 Portfolio Optimization Example")
    print("=" * 50)
    
    simulator = UnifiedMarketSimulator()
    await simulator.initialize()
    
    # Multi-asset sentiment
    sentiment_results = [
        SentimentAnalysis(
            text="Tech stocks show strong momentum",
            sentiment=SentimentType.POSITIVE,
            confidence=0.8,
            entities_detected=[{"text": "tech stocks", "label": "MISC", "confidence": 0.9}],
            key_phrases=["tech stocks", "strong momentum"],
            market_impact_keywords=["momentum"],
            quantum_sentiment_vector=[0.05, 0.1, 0.25, 0.5, 0.1]
        )
    ]
    
    # Multi-asset market data
    market_data = {
        "AAPL": {"current_price": 150.0, "volume": 50000000, "volatility": 0.25},
        "GOOGL": {"current_price": 100.0, "volume": 30000000, "volatility": 0.3},
        "MSFT": {"current_price": 300.0, "volume": 40000000, "volatility": 0.22}
    }
    
    simulation_params = {
        "target_assets": ["AAPL", "GOOGL", "MSFT"],
        "method": "hybrid_qml",
        "time_horizon": 30,
        "num_scenarios": 200
    }
    
    # Get predictions
    predictions = await simulator.simulate(
        sentiment_results=sentiment_results,
        market_data=market_data,
        simulation_params=simulation_params
    )
    
    # Optimize portfolio
    optimal_portfolio = await simulator.optimize_portfolio(
        assets=["AAPL", "GOOGL", "MSFT"],
        predictions=predictions,
        risk_tolerance=0.5
    )
    
    print(f"\n🎯 Optimal Portfolio:")
    for asset, weight in optimal_portfolio["optimal_weights"].items():
        print(f"   {asset}: {weight:.1%}")
    
    print(f"\n📊 Portfolio Metrics:")
    print(f"   Expected Return: {optimal_portfolio['expected_return']:.2%}")
    print(f"   Portfolio Volatility: {optimal_portfolio['portfolio_volatility']:.2%}")
    print(f"   Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.2f}")
    print(f"   Method: {optimal_portfolio['optimization_method']}")
    
    await simulator.cleanup()


def quantum_circuit_example():
    """Quantum circuit example for educational purposes"""
    print("\n⚛️  Quantum Circuit Example")
    print("=" * 50)
    
    try:
        from classiq import qfunc, QArray, QBit, Output, allocate, H, CX, create_model, synthesize
        
        @qfunc
        def create_bell_state(q: QArray[QBit]):
            """Create a Bell state for quantum correlation analysis"""
            H(q[0])  # Superposition
            CX(q[0], q[1])  # Entanglement
        
        @qfunc
        def main(q: Output[QArray[QBit]]):
            """Main quantum function"""
            allocate(2, q)
            create_bell_state(q)
        
        # Create and synthesize model
        model = create_model(main)
        qprog = synthesize(model)
        
        print("✅ Quantum Bell state circuit created successfully!")
        print("   - 2 qubits allocated")
        print("   - Hadamard gate applied for superposition")
        print("   - CNOT gate applied for entanglement")
        print("   - Circuit ready for quantum market correlation analysis")
        
    except ImportError:
        print("⚠️  Classiq not available - quantum circuit features disabled")
        print("   Install with: pip install classiq")
    except Exception as e:
        print(f"❌ Quantum circuit creation failed: {e}")


async def main():
    """Run all examples"""
    print("🎯 Quantum Market Simulator - Comprehensive Examples")
    print("=" * 70)
    
    try:
        await basic_simulation_example()
        await quantum_vs_classical_comparison()
        await portfolio_optimization_example()
        quantum_circuit_example()
        
        print("\n" + "=" * 70)
        print("✅ All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

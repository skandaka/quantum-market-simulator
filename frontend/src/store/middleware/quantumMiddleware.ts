import { Middleware, AnyAction } from '@reduxjs/toolkit';
import { RootState } from '../index';

export const quantumMiddleware: Middleware<{}, RootState> = (store) => (next) => (action: AnyAction) => {
    // Log quantum-related actions
    if (action.type.startsWith('quantum/')) {
        console.log('Quantum action:', action);
    }

    // Handle specific quantum actions
    if (action.type === 'simulation/setSimulationResults') {
        const results = action.payload;
        if (results?.quantum_metrics) {
            // Update quantum metrics in the quantum slice
            store.dispatch({
                type: 'quantum/updateQuantumMetrics',
                payload: {
                    circuitDepth: results.quantum_metrics.circuit_depth,
                    quantumVolume: results.quantum_metrics.quantum_volume,
                    executionTime: results.quantum_metrics.execution_time_ms,
                },
            });
        }
    }

    return next(action);
};
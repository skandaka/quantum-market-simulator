import { Middleware } from '@reduxjs/toolkit';

export const quantumMiddleware: Middleware = (store) => (next) => (action) => {
    // Log quantum-related actions
    if (typeof action === 'object' && action !== null && 'type' in action) {
        const typedAction = action as { type: string; payload?: any };

        if (typedAction.type.startsWith('quantum/')) {
            console.log('Quantum action:', typedAction);
        }

        // Handle specific quantum actions
        if (typedAction.type === 'simulation/setSimulationResults') {
            const results = typedAction.payload;
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
    }

    return next(action);
};
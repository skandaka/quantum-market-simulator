import { configureStore } from '@reduxjs/toolkit';
import { useDispatch, useSelector, TypedUseSelectorHook } from 'react-redux';
import simulationReducer from './slices/simulationSlice';
import quantumReducer from './slices/quantumSlice';
import marketReducer from './slices/marketSlice';
import uiReducer from './slices/uiSlice';
import { quantumMiddleware } from './middleware/quantumMiddleware';

export const store = configureStore({
    reducer: {
        simulation: simulationReducer,
        quantum: quantumReducer,
        market: marketReducer,
        ui: uiReducer,
    },
    middleware: (getDefaultMiddleware) =>
        getDefaultMiddleware({
            serializableCheck: {
                // Ignore these action types
                ignoredActions: ['simulation/setSimulationResults'],
                // Ignore these field paths in all actions
                ignoredActionPaths: ['payload.timestamp'],
                // Ignore these paths in the state
                ignoredPaths: ['simulation.results.timestamp'],
            },
        }).concat(quantumMiddleware),
});

// Infer types from store
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Export typed hooks
export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;
import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { SimulationResponse } from '../../types';

interface SimulationState {
    results: SimulationResponse | null;
    status: 'idle' | 'running' | 'completed' | 'error';
    progress: number;
    error: string | null;
    history: SimulationResponse[];
}

const initialState: SimulationState = {
    results: null,
    status: 'idle',
    progress: 0,
    error: null,
    history: [],
};

const simulationSlice = createSlice({
    name: 'simulation',
    initialState,
    reducers: {
        setSimulationResults: (state, action: PayloadAction<SimulationResponse>) => {
            state.results = action.payload;
            state.history.unshift(action.payload);
            if (state.history.length > 10) {
                state.history = state.history.slice(0, 10);
            }
        },
        setSimulationStatus: (state, action: PayloadAction<SimulationState['status']>) => {
            state.status = action.payload;
        },
        setSimulationProgress: (state, action: PayloadAction<any>) => {
            state.progress = action.payload.progress * 100;
        },
        setSimulationError: (state, action: PayloadAction<string>) => {
            state.error = action.payload;
            state.status = 'error';
        },
        clearSimulation: state => {
            state.results = null;
            state.status = 'idle';
            state.progress = 0;
            state.error = null;
        },
    },
});

export const {
    setSimulationResults,
    setSimulationStatus,
    setSimulationProgress,
    setSimulationError,
    clearSimulation,
} = simulationSlice.actions;

export default simulationSlice.reducer;
import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface QuantumState {
  backend: string;
  status: 'operational' | 'degraded' | 'offline';
  availableQubits: number;
  queueLength: number;
  metrics: {
    circuitDepth: number;
    quantumVolume: number;
    executionTime: number;
  } | null;
}

const initialState: QuantumState = {
  backend: 'simulator',
  status: 'operational',
  availableQubits: 10,
  queueLength: 0,
  metrics: null,
};

const quantumSlice = createSlice({
  name: 'quantum',
  initialState,
  reducers: {
    setQuantumBackend: (state, action: PayloadAction<string>) => {
      state.backend = action.payload;
    },
    setQuantumStatus: (state, action: PayloadAction<QuantumState['status']>) => {
      state.status = action.payload;
    },
    updateQuantumMetrics: (state, action: PayloadAction<QuantumState['metrics']>) => {
      state.metrics = action.payload;
    },
    setQueueLength: (state, action: PayloadAction<number>) => {
      state.queueLength = action.payload;
    },
  },
});

export const {
  setQuantumBackend,
  setQuantumStatus,
  updateQuantumMetrics,
  setQueueLength,
} = quantumSlice.actions;

export default quantumSlice.reducer;
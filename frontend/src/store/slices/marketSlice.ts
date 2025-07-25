import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface MarketState {
    selectedAssets: string[];
    marketData: Record<string, any>;
    lastUpdate: string | null;
    isRealtime: boolean;
}

const initialState: MarketState = {
    selectedAssets: ['AAPL'],
    marketData: {},
    lastUpdate: null,
    isRealtime: false,
};

const marketSlice = createSlice({
    name: 'market',
    initialState,
    reducers: {
        setSelectedAssets: (state, action: PayloadAction<string[]>) => {
            state.selectedAssets = action.payload;
        },
        updateMarketData: (state, action: PayloadAction<{ asset: string; data: any }>) => {
            state.marketData[action.payload.asset] = action.payload.data;
            state.lastUpdate = new Date().toISOString();
        },
        toggleRealtime: (state) => {
            state.isRealtime = !state.isRealtime;
        },
        clearMarketData: (state) => {
            state.marketData = {};
            state.lastUpdate = null;
        },
    },
});

export const {
    setSelectedAssets,
    updateMarketData,
    toggleRealtime,
    clearMarketData,
} = marketSlice.actions;

export default marketSlice.reducer;
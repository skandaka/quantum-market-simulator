import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface UIState {
    theme: 'light' | 'dark';
    sidebarOpen: boolean;
    notifications: Array<{
        id: string;
        type: 'info' | 'success' | 'warning' | 'error';
        message: string;
        timestamp: number;
    }>;
}

const initialState: UIState = {
    theme: 'dark',
    sidebarOpen: false,
    notifications: [],
};

const uiSlice = createSlice({
    name: 'ui',
    initialState,
    reducers: {
        toggleTheme: (state) => {
            state.theme = state.theme === 'dark' ? 'light' : 'dark';
        },
        toggleSidebar: (state) => {
            state.sidebarOpen = !state.sidebarOpen;
        },
        addNotification: (state, action: PayloadAction<Omit<UIState['notifications'][0], 'id' | 'timestamp'>>) => {
            state.notifications.push({
                ...action.payload,
                id: Math.random().toString(36).substr(2, 9),
                timestamp: Date.now(),
            });
        },
        removeNotification: (state, action: PayloadAction<string>) => {
            state.notifications = state.notifications.filter(n => n.id !== action.payload);
        },
        clearNotifications: (state) => {
            state.notifications = [];
        },
    },
});

export const {
    toggleTheme,
    toggleSidebar,
    addNotification,
    removeNotification,
    clearNotifications,
} = uiSlice.actions;

export default uiSlice.reducer;
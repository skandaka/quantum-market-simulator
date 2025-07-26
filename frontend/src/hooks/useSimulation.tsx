import { useState, useCallback, useEffect } from 'react';
import { useDispatch } from 'react-redux';
import { io, Socket } from 'socket.io-client';
import { simulationAPI } from '../services/api';
import {
    setSimulationResults,
    setSimulationStatus,
    setSimulationProgress,
    setSimulationError
} from '../store/slices/simulationSlice';
import { SimulationRequest } from '../types';
import toast from 'react-hot-toast';

export const useSimulation = () => {
    const dispatch = useDispatch();
    const [isLoading, setIsLoading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [socket, setSocket] = useState<Socket | null>(null);

    useEffect(() => {
        // Initialize WebSocket connection
        const socketInstance = io(import.meta.env.VITE_WS_URL || 'ws://localhost:8000', {
            path: '/ws',
            transports: ['websocket'],
        });

        socketInstance.on('connect', () => {
            // console.log('WebSocket connected');
        });

        socketInstance.on('progress', (data) => {
            setProgress(data.progress * 100);
            dispatch(setSimulationProgress(data));
        });

        socketInstance.on('error', (error) => {
            console.error('WebSocket error:', error);
            toast.error('Connection error');
        });

        setSocket(socketInstance);

        return () => {
            socketInstance.disconnect();
        };
    }, [dispatch]);

    const runSimulation = useCallback(async (request: SimulationRequest) => {
        setIsLoading(true);
        setProgress(0);
        dispatch(setSimulationStatus('running'));

        try {
            // Run the simulation
            const response = await simulationAPI.runSimulation(request);

            // Subscribe to simulation updates if WebSocket is connected
            if (socket && socket.connected) {
                socket.emit('subscribe', { channel: `simulation:${response.request_id}` });
            }

            dispatch(setSimulationResults(response));
            dispatch(setSimulationStatus('completed'));

            return response;
        } catch (error: any) {
            console.error('Simulation error:', error);
            dispatch(setSimulationError(error.message || 'Simulation failed'));
            dispatch(setSimulationStatus('error'));
            throw error;
        } finally {
            setIsLoading(false);
            setProgress(100);
        }
    }, [dispatch, socket]);

    const runBacktest = useCallback(async (request: any) => {
        setIsLoading(true);

        try {
            const response = await simulationAPI.runBacktest(request);
            return response;
        } catch (error) {
            console.error('Backtest error:', error);
            toast.error('Backtest failed');
            throw error;
        } finally {
            setIsLoading(false);
        }
    }, []);

    const getQuantumStatus = useCallback(async () => {
        try {
            const status = await simulationAPI.getQuantumStatus();
            return status;
        } catch (error) {
            console.error('Failed to get quantum status:', error);
            return null;
        }
    }, []);

    return {
        runSimulation,
        runBacktest,
        getQuantumStatus,
        isLoading,
        progress,
    };
};
import * as d3 from 'd3';

export const formatPrice = (price: number): string => {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(price);
};

export const formatPercent = (value: number): string => {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(value);
};

export const generateGradientId = (baseId: string): string => {
    return `${baseId}-${Math.random().toString(36).substr(2, 9)}`;
};

export const createColorScale = (sentiment: string): string => {
    const colorMap: Record<string, string> = {
        very_positive: '#10B981',
        positive: '#34D399',
        neutral: '#FCD34D',
        negative: '#F87171',
        very_negative: '#DC2626',
    };
    return colorMap[sentiment] || '#6B7280';
};

export const interpolatePath = (path1: number[], path2: number[], t: number): number[] => {
    if (path1.length !== path2.length) {
        throw new Error('Paths must have the same length');
    }

    return path1.map((val, i) => val + (path2[i] - val) * t);
};

export const calculateMovingAverage = (data: number[], window: number): number[] => {
    const result: number[] = [];

    for (let i = 0; i < data.length; i++) {
        if (i < window - 1) {
            result.push(data[i]);
        } else {
            const sum = data.slice(i - window + 1, i + 1).reduce((a, b) => a + b, 0);
            result.push(sum / window);
        }
    }

    return result;
};

export const detectTrend = (prices: number[]): 'up' | 'down' | 'neutral' => {
    if (prices.length < 2) return 'neutral';

    const firstPrice = prices[0];
    const lastPrice = prices[prices.length - 1];
    const change = (lastPrice - firstPrice) / firstPrice;

    if (change > 0.02) return 'up';
    if (change < -0.02) return 'down';
    return 'neutral';
};

export const calculateVolatility = (returns: number[]): number => {
    if (returns.length === 0) return 0;

    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const squaredDiffs = returns.map(r => Math.pow(r - mean, 2));
    const variance = squaredDiffs.reduce((a, b) => a + b, 0) / returns.length;

    return Math.sqrt(variance) * Math.sqrt(252); // Annualized
};

export const generateSparklineData = (values: number[], width: number = 100, height: number = 30): string => {
    const xScale = d3.scaleLinear()
        .domain([0, values.length - 1])
        .range([0, width]);

    const yScale = d3.scaleLinear()
        .domain(d3.extent(values) as [number, number])
        .range([height, 0]);

    const line = d3.line<number>()
        .x((_d, i) => xScale(i))
        .y(d => yScale(d))
        .curve(d3.curveMonotoneX);

    return line(values) || '';
};
import axios, { AxiosError, AxiosInstance, AxiosResponse } from 'axios';
import {
  StockRecommendation,
  PortfolioPosition,
  PerformanceMetrics,
  DailyPerformance,
  MonthlyReturn,
  GpuStatus,
  GpuHistoryPoint,
  SystemStatus,
  BacktestParams,
  BacktestResult,
  MarketOverview,
} from './types';

// Match FastAPI router prefixes defined in backend
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.client.interceptors.response.use(
      (response: AxiosResponse) => response,
      (error: AxiosError) => {
        if (error.response) {
          console.error(`API Error: ${error.response.status} - ${error.response.statusText}`);
        } else if (error.request) {
          console.error('API Error: No response received from server');
        } else {
          console.error(`API Error: ${error.message}`);
        }
        return Promise.reject(error);
      }
    );
  }

  async fetchStockRecommendations(): Promise<StockRecommendation[]> {
    const response = await this.client.get<StockRecommendation[]>('/api/stocks/recommendations');
    return response.data;
  }

  async fetchPortfolio(): Promise<PortfolioPosition[]> {
    const response = await this.client.get<PortfolioPosition[]>('/api/portfolio/positions');
    return response.data;
  }

  async fetchPerformance(range?: string): Promise<{
    metrics: PerformanceMetrics;
    dailyData: DailyPerformance[];
    monthlyData: MonthlyReturn[];
  }> {
    const response = await this.client.get('/api/portfolio/performance', {
      params: range ? { range } : undefined,
    });
    return response.data;
  }

  async fetchGpuStatus(): Promise<GpuStatus> {
    const response = await this.client.get<GpuStatus>('/api/monitor/gpu');
    return response.data;
  }

  async fetchGpuHistory(limit: number = 60): Promise<GpuHistoryPoint[]> {
    const response = await this.client.get<GpuHistoryPoint[]>('/api/monitor/gpu/history', {
      params: { limit },
    });
    return response.data;
  }

  async fetchSystemStatus(): Promise<SystemStatus> {
    const response = await this.client.get<SystemStatus>('/api/monitor/system');
    return response.data;
  }

  async fetchMarketOverview(): Promise<MarketOverview[]> {
    // Use stocks universe endpoint as market overview fallback
    const response = await this.client.get<MarketOverview[]>('/api/stocks/universe');
    return response.data;
  }

  async runBacktest(params: BacktestParams): Promise<BacktestResult> {
    // Backtest served via monitor router for now; adjust if dedicated router added
    const response = await this.client.post<BacktestResult>('/api/monitor/backtest', params);
    return response.data;
  }

  async sellStock(code: string): Promise<{ success: boolean; message: string }> {
    const response = await this.client.post('/api/portfolio/simulate-sell', { code });
    return response.data;
  }
}

export const apiClient = new ApiClient();

export function createWebSocketConnection(endpoint: string): WebSocket {
  const url = `${WS_BASE_URL}/${endpoint}`;
  const ws = new WebSocket(url);

  ws.onopen = () => {
    console.log(`WebSocket connected: ${endpoint}`);
  };

  ws.onerror = (error) => {
    console.error(`WebSocket error on ${endpoint}:`, error);
  };

  ws.onclose = () => {
    console.log(`WebSocket closed: ${endpoint}`);
  };

  return ws;
}

export {
  API_BASE_URL,
  WS_BASE_URL,
};

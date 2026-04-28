export interface StockInfo {
  code: string;
  name: string;
  industry: string;
  marketCap: number;
}

export interface StockRecommendation {
  code: string;
  name: string;
  score: number;
  reasons: string[];
  rank: number;
  expectedReturn?: number;
  industry?: string;
  marketCap?: number;
  factors?: Record<string, number>;
}

export interface PortfolioPosition {
  code: string;
  name: string;
  buyPrice: number;
  buyTime: string;
  currentPrice: number;
  pnl: number;
  pnlPct: number;
  quantity?: number;
  totalValue?: number;
}

export interface PerformanceMetrics {
  totalReturn: number;
  annualizedReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  volatility?: number;
  beta?: number;
  alpha?: number;
}

export interface DailyPerformance {
  date: string;
  nav: number;
  dailyReturn: number;
  cumulativeReturn: number;
}

export interface MonthlyReturn {
  month: string;
  return: number;
}

export interface GpuStatus {
  utilization: number;
  memoryUsed: number;
  memoryTotal: number;
  temperature: number;
  power: number;
  deviceName?: string;
  fanSpeed?: number;
  clockSpeed?: number;
}

export interface GpuHistoryPoint {
  timestamp: string;
  utilization: number;
  memoryUsed: number;
  temperature: number;
  power: number;
}

export interface SystemStatus {
  tradingStatus: string;
  lastUpdate: string;
  nextOperation: string;
  isMarketOpen: boolean;
}

export interface BacktestParams {
  startDate: string;
  endDate: string;
  initialCapital: number;
  strategy: string;
}

export interface BacktestResult {
  params: BacktestParams;
  metrics: PerformanceMetrics;
  dailyData: DailyPerformance[];
  monthlyData: MonthlyReturn[];
  trades: TradeRecord[];
}

export interface TradeRecord {
  date: string;
  code: string;
  name: string;
  action: 'buy' | 'sell';
  price: number;
  quantity: number;
  amount: number;
}

export interface MarketOverview {
  indexName: string;
  indexValue: number;
  change: number;
  changePct: number;
  volume: number;
  turnover: number;
}

export type TimeRange = '1M' | '3M' | '6M' | '1Y' | 'ALL';

export type NavItem = {
  path: string;
  label: string;
  icon: string;
};

export type WebSocketMessage =
  | { type: 'stock_update'; data: StockRecommendation[] }
  | { type: 'portfolio_update'; data: PortfolioPosition[] }
  | { type: 'gpu_update'; data: GpuStatus }
  | { type: 'system_status'; data: SystemStatus }
  | { type: 'heartbeat'; timestamp: string };

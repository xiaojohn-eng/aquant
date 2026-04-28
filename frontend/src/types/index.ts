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
  expected_return?: number;
  risk_level?: string;
  industry?: string;
  marketCap?: number;
  factors?: Record<string, number>;
  model_confidence?: number;
  sector_alignment?: string;
}

export interface PortfolioPosition {
  code: string;
  name: string;
  buyPrice: number;
  buy_price?: number;
  buyTime: string;
  buy_time?: string;
  currentPrice: number;
  current_price?: number;
  pnl: number;
  pnlPct: number;
  pnl_pct?: number;
  quantity?: number;
  totalValue?: number;
  current_value?: number;
  total_value?: number;
}

export interface PerformanceMetrics {
  total_return_pct: number;
  annualized_return_pct?: number;
  sharpe_ratio?: number;
  max_drawdown_pct?: number;
  win_rate_pct?: number;
  profit_factor?: number;
  calmar_ratio?: number;
  avg_win_pct?: number;
  avg_loss_pct?: number;
  total_trades?: number;
  open_positions?: number;
  last_updated?: string;
  volatility?: number;
  beta?: number;
  alpha?: number;
}

export interface DailyPerformance {
  date: string;
  nav: number;
  daily_return?: number;
  cumulative_return?: number;
  dailyReturn?: number;
  cumulativeReturn?: number;
}

export interface MonthlyReturn {
  month: string;
  return: number;
}

export interface GpuStatus {
  index?: number;
  name?: string;
  utilization_gpu_pct?: number;
  memory_used_mb?: number;
  memory_total_mb?: number;
  temperature_c?: number;
  power_draw_w?: number;
  fan_speed_pct?: number;
  clock_sm_mhz?: number;
  utilization?: number;
  memoryUsed?: number;
  memoryTotal?: number;
  temperature?: number;
  power?: number;
  deviceName?: string;
  fanSpeed?: number;
  clockSpeed?: number;
}

export interface GpuHistoryPoint {
  timestamp: string;
  utilization: number;
  memory_used?: number;
  temperature: number;
  power: number;
}

export interface SystemStatus {
  state?: string;
  version?: string;
  tradingStatus?: string;
  lastUpdate?: string;
  nextOperation?: string;
  isMarketOpen?: boolean;
  is_trading_day?: boolean;
  market_open?: boolean;
  last_update_time?: string;
  next_buy_time?: string;
  next_sell_time?: string;
  db_connected?: boolean;
  gpu_available?: boolean;
  gpu_count?: number;
  active_positions?: number;
  today_recommendations?: number;
  uptime_seconds?: number;
  message?: string;
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
  icon: unknown;
};

export type WebSocketMessage =
  | { type: 'stock_update'; data: StockRecommendation[] }
  | { type: 'portfolio_update'; data: PortfolioPosition[] }
  | { type: 'gpu_update'; data: GpuStatus }
  | { type: 'system_status'; data: SystemStatus }
  | { type: 'heartbeat'; timestamp: string };

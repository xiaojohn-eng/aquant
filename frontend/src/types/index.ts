export interface StockRecommendation {
  code: string
  name: string
  score: number
  rank: number
  expected_return?: number
  factors: Record<string, number>
  reasons: string[]
  advanced_report?: AdvancedReport
}

export interface AdvancedReport {
  investment_thesis: string
  technical_signals: TechnicalSignal[]
  risk_assessment: RiskAssessment
  entry_strategy: string
  target_price?: number
  stop_loss?: number
}

export interface TechnicalSignal {
  name: string
  value: string
  direction: 'bullish' | 'bearish' | 'neutral'
}

export interface RiskAssessment {
  level: 'low' | 'medium' | 'high'
  max_drawdown_5d: number
  overnight_volatility: number
}

export interface GPUStatus {
  device_id: number
  name: string
  utilization_pct: number
  memory_used_mb: number
  memory_total_mb: number
  temperature_c: number
}

export interface RiskStatus {
  circuit_breaker_active: boolean
  consecutive_loss_days: number
  cumulative_loss_pct: number
  daily_loss_limit_pct: number
}

export interface MarketSnapshot {
  type: string
  time: string
  state: string
  recommendations?: StockRecommendation[]
}

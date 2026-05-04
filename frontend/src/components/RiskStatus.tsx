import { useEffect, useState } from 'react'
import type { RiskStatus as RiskStatusType } from '../types'

export default function RiskStatus() {
  const [risk, setRisk] = useState<RiskStatusType>({
    circuit_breaker_active: false,
    consecutive_loss_days: 0,
    cumulative_loss_pct: 0,
    daily_loss_limit_pct: -3.0,
  })

  useEffect(() => {
    // Poll risk status from API
    const poll = async () => {
      try {
        const res = await fetch('/api/risk/status')
        if (res.ok) {
          const data = await res.json()
          setRisk(data)
        }
      } catch { /* ignore */ }
    }
    poll()
    const interval = setInterval(poll, 10000)
    return () => clearInterval(interval)
  }, [])

  if (risk.circuit_breaker_active) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
        <span className="text-2xl">🛑</span>
        <div>
          <h3 className="font-bold text-red-700">Circuit Breaker Active</h3>
          <p className="text-sm text-red-600">
            Trading paused. Cumulative loss: {risk.cumulative_loss_pct.toFixed(2)}%
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-4 flex items-center gap-3">
      <span className="text-2xl">✅</span>
      <div className="flex-1">
        <h3 className="font-bold text-emerald-700">Risk Status: Normal</h3>
        <div className="flex gap-4 text-sm text-emerald-600 mt-1">
          <span>Loss Days: {risk.consecutive_loss_days}/5</span>
          <span>Cum. Loss: {risk.cumulative_loss_pct.toFixed(2)}%</span>
          <span>Limit: {risk.daily_loss_limit_pct}%</span>
        </div>
      </div>
    </div>
  )
}

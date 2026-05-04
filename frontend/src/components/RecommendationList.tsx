import { useState } from 'react'
import type { MarketSnapshot, StockRecommendation } from '../types'

interface Props {
  data: MarketSnapshot | null
}

export default function RecommendationList({ data }: Props) {
  const [expanded, setExpanded] = useState<string | null>(null)
  const recs = data?.recommendations ?? []

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-bold">📊 Top 20 Recommendations</h2>
        <span className="text-xs bg-emerald-100 text-emerald-700 px-2 py-1 rounded-full">
          T+1 Adaptive Strategy
        </span>
      </div>

      {recs.length === 0 ? (
        <div className="text-center py-12 text-gray-400">
          <p className="text-4xl mb-2">📡</p>
          <p>Waiting for market data...</p>
        </div>
      ) : (
        <div className="space-y-2 max-h-[600px] overflow-y-auto">
          {recs.map((r) => (
            <RecCard key={r.code} rec={r} isOpen={expanded === r.code} onToggle={() => setExpanded(expanded === r.code ? null : r.code)} />
          ))}
        </div>
      )}
    </div>
  )
}

function RecCard({ rec, isOpen, onToggle }: { rec: StockRecommendation; isOpen: boolean; onToggle: () => void }) {
  const scoreColor = rec.score > 80 ? 'text-emerald-600' : rec.score > 60 ? 'text-amber-600' : 'text-red-600'
  const bgColor = rec.rank <= 3 ? 'bg-gradient-to-r from-amber-50 to-transparent border-l-4 border-amber-400' : 'bg-white'

  return (
    <div className={`rounded-lg border transition ${bgColor} ${isOpen ? 'shadow-md' : ''}`}>
      <button onClick={onToggle} className="w-full px-4 py-3 flex items-center justify-between text-left">
        <div className="flex items-center gap-3">
          <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${rec.rank <= 3 ? 'bg-amber-100 text-amber-700' : 'bg-gray-100 text-gray-600'}`}>
            {rec.rank}
          </span>
          <div>
            <span className="font-semibold text-sm">{rec.name}</span>
            <span className="text-xs text-gray-400 ml-2">{rec.code}</span>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <span className={`font-bold ${scoreColor}`}>{rec.score.toFixed(1)}</span>
          <span className="text-xs text-gray-400">{isOpen ? '▼' : '▶'}</span>
        </div>
      </button>

      {isOpen && (
        <div className="px-4 pb-4 border-t bg-gray-50">
          {/* Factors */}
          <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
            {Object.entries(rec.factors).map(([k, v]) => (
              <div key={k} className="bg-white rounded p-2">
                <span className="text-gray-500">{k}</span>
                <div className={`font-mono font-bold ${typeof v === 'number' && v > 0 ? 'text-emerald-600' : 'text-red-600'}`}>
                  {typeof v === 'number' ? v.toFixed(4) : v}
                </div>
              </div>
            ))}
          </div>

          {/* Reasons */}
          <div className="mt-3">
            <h4 className="text-xs font-semibold text-gray-500 mb-1">Analysis</h4>
            <ul className="text-xs space-y-1">
              {rec.reasons.map((reason, i) => (
                <li key={i} className="text-gray-600">• {reason}</li>
              ))}
            </ul>
          </div>

          {/* Advanced Report */}
          {rec.advanced_report && (
            <div className="mt-3 p-3 bg-white rounded border">
              <h4 className="text-xs font-semibold text-gray-500 mb-2">6-Dimension Analysis</h4>
              <p className="text-xs text-gray-700 mb-2">{rec.advanced_report.investment_thesis}</p>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-gray-500">Risk Level:</span>
                  <span className={`ml-1 font-bold ${rec.advanced_report.risk_assessment.level === 'low' ? 'text-emerald-600' : rec.advanced_report.risk_assessment.level === 'medium' ? 'text-amber-600' : 'text-red-600'}`}>
                    {rec.advanced_report.risk_assessment.level.toUpperCase()}
                  </span>
                </div>
                {rec.advanced_report.target_price && (
                  <div>
                    <span className="text-gray-500">Target:</span>
                    <span className="ml-1 font-bold">{rec.advanced_report.target_price.toFixed(2)}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

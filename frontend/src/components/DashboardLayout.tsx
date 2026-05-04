import type { ReactNode } from 'react'

interface Props {
  children: ReactNode
  darkMode: boolean
  onToggleDark: () => void
  wsConnected: boolean
}

export default function DashboardLayout({ children, darkMode, onToggleDark, wsConnected }: Props) {
  return (
    <div className={`min-h-screen transition-colors ${darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'}`}>
      {/* Header */}
      <header className={`sticky top-0 z-50 border-b ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center text-white font-bold text-sm">
              AQ
            </div>
            <div>
              <h1 className="text-lg font-bold">AQuant <span className="text-xs font-normal opacity-60">v4.4</span></h1>
              <p className="text-xs opacity-60">A-Share T+1 Quantitative Trading</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* WebSocket status */}
            <div className="flex items-center gap-1.5">
              <span className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`} />
              <span className="text-xs opacity-60">{wsConnected ? 'Live' : 'Offline'}</span>
            </div>

            {/* Dark mode toggle */}
            <button
              onClick={onToggleDark}
              className={`p-2 rounded-lg text-sm transition ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-100 hover:bg-gray-200'}`}
            >
              {darkMode ? '☀️' : '🌙'}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {children}
      </main>

      {/* Footer */}
      <footer className={`border-t ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} mt-8`}>
        <div className="max-w-7xl mx-auto px-4 py-3 text-center text-xs opacity-50">
          AQuant v4.4 — Powered by NVIDIA Grace Hopper — Data source: East Money / AkShare
        </div>
      </footer>
    </div>
  )
}

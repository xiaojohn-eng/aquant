import { useState } from 'react'
import DashboardLayout from './components/DashboardLayout'
import RecommendationList from './components/RecommendationList'
import EquityChart from './components/EquityChart'
import GPUMonitor from './components/GPUMonitor'
import RiskStatus from './components/RiskStatus'
import { useWebSocket } from './hooks/useWebSocket'
import './index.css'

function App() {
  const [darkMode, setDarkMode] = useState(false)
  const { marketData, gpuData, connected } = useWebSocket()

  return (
    <div className={`app ${darkMode ? 'dark' : ''}`}>
      <DashboardLayout
        darkMode={darkMode}
        onToggleDark={() => setDarkMode(!darkMode)}
        wsConnected={connected}
      >
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Risk Status */}
          <div className="lg:col-span-3">
            <RiskStatus />
          </div>

          {/* Recommendations */}
          <div className="lg:col-span-2">
            <RecommendationList data={marketData} />
          </div>

          {/* GPU Monitor */}
          <div>
            <GPUMonitor data={gpuData} />
          </div>

          {/* Equity Chart */}
          <div className="lg:col-span-3">
            <EquityChart data={marketData} />
          </div>
        </div>
      </DashboardLayout>
    </div>
  )
}

export default App

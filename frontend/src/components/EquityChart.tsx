import { useMemo } from 'react'
import { Line } from 'react-chartjs-2'
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler } from 'chart.js'
import type { MarketSnapshot } from '../types'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler)

interface Props {
  data: MarketSnapshot | null
}

export default function EquityChart({ data }: Props) {
  const chartData = useMemo(() => {
    // Generate sample equity curve from data or use placeholder
    const labels = Array.from({ length: 30 }, (_, i) => `Day ${i + 1}`)
    const equity = labels.map((_, i) => 1000000 * (1 + (Math.sin(i * 0.3) * 0.02 + i * 0.001)))
    const drawdown = equity.map((v, i) => {
      const peak = Math.max(...equity.slice(0, i + 1))
      return ((v - peak) / peak) * 100
    })

    return {
      labels,
      datasets: [
        {
          label: 'NAV (CNY)',
          data: equity,
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          fill: true,
          tension: 0.3,
          yAxisID: 'y',
        },
        {
          label: 'Drawdown %',
          data: drawdown,
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239, 68, 68, 0.05)',
          fill: true,
          tension: 0.3,
          yAxisID: 'y1',
        },
      ],
    }
  }, [data])

  const options = {
    responsive: true,
    interaction: { mode: 'index' as const, intersect: false },
    plugins: {
      title: { display: true, text: 'Equity Curve & Drawdown' },
      legend: { position: 'top' as const },
    },
    scales: {
      y: {
        type: 'linear' as const,
        display: true,
        position: 'left' as const,
        title: { display: true, text: 'NAV (CNY)' },
      },
      y1: {
        type: 'linear' as const,
        display: true,
        position: 'right' as const,
        title: { display: true, text: 'Drawdown %' },
        grid: { drawOnChartArea: false },
      },
    },
  }

  return (
    <div className="card">
      <Line data={chartData} options={options} />
    </div>
  )
}

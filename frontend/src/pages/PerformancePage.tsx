import React from 'react';
import { BarChart3, RefreshCw } from 'lucide-react';
import PerformanceChart from '../components/PerformanceChart';
import { useAutoRefresh } from '../hooks/useAutoRefresh';
import { apiClient } from '../api';
import { DailyPerformance, MonthlyReturn, PerformanceMetrics } from '../types';

interface PerformanceResponse {
  metrics: PerformanceMetrics;
  dailyData: DailyPerformance[];
  monthlyData: MonthlyReturn[];
}

const PerformancePage: React.FC = () => {
  const {
    data,
    loading,
    refresh,
    lastUpdated,
  } = useAutoRefresh<PerformanceResponse>({
    fetchFn: apiClient.fetchPerformance,
    enabled: true,
    marketCloseInterval: 600000,
  });

  // Fallback mock data for demo
  const mockDailyData: DailyPerformance[] = React.useMemo(() => {
    if (data?.dailyData && data.dailyData.length > 0) return data.dailyData;
    const result: DailyPerformance[] = [];
    let nav = 1.0;
    const now = new Date();
    for (let i = 90; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 86400000);
      const ret = (Math.random() - 0.48) * 0.03;
      nav = nav * (1 + ret);
      result.push({
        date: date.toISOString().split('T')[0],
        nav: Number(nav.toFixed(4)),
        dailyReturn: Number((ret * 100).toFixed(2)),
        cumulativeReturn: Number(((nav - 1) * 100).toFixed(2)),
      });
    }
    return result;
  }, [data?.dailyData]);

  const mockMonthlyData: MonthlyReturn[] = React.useMemo(() => {
    if (data?.monthlyData && data.monthlyData.length > 0) return data.monthlyData;
    const months: MonthlyReturn[] = [];
    const now = new Date();
    for (let i = 11; i >= 0; i--) {
      const d = new Date(now.getFullYear(), now.getMonth() - i, 1);
      months.push({
        month: `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`,
        return: Number(((Math.random() - 0.45) * 8).toFixed(2)),
      });
    }
    return months;
  }, [data?.monthlyData]);

  const mockMetrics: PerformanceMetrics = data?.metrics || {
    totalReturn: 15.82,
    annualizedReturn: 23.45,
    sharpeRatio: 1.68,
    maxDrawdown: -8.32,
    winRate: 0.58,
    profitFactor: 2.14,
    volatility: 12.5,
    beta: 0.85,
    alpha: 8.2,
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-[#d4a574]" />
          <h2 className="text-base font-medium text-[#e6ddd0]">业绩分析</h2>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={refresh}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-[#95a5a6] bg-[#2a3a5c]/50 rounded-lg hover:bg-[#2a3a5c] hover:text-[#e6ddd0] transition-colors"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            刷新
          </button>
          {lastUpdated && (
            <span className="text-xs text-[#95a5a6]">
              更新于 {lastUpdated.toLocaleTimeString('zh-CN')}
            </span>
          )}
        </div>
      </div>

      <PerformanceChart
        dailyData={mockDailyData}
        monthlyData={mockMonthlyData}
        metrics={mockMetrics}
        loading={loading}
      />
    </div>
  );
};

export default PerformancePage;

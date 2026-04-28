import React, { useState, useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from 'recharts';
import { Calendar, TrendingUp, TrendingDown, Activity, Target, Zap, Percent } from 'lucide-react';
import { DailyPerformance, PerformanceMetrics, TimeRange, MonthlyReturn } from '../types';

interface PerformanceChartProps {
  dailyData: DailyPerformance[];
  monthlyData: MonthlyReturn[];
  metrics: PerformanceMetrics;
  loading?: boolean;
}

const TimeRangeOptions: { label: string; value: TimeRange }[] = [
  { label: '1月', value: '1M' },
  { label: '3月', value: '3M' },
  { label: '6月', value: '6M' },
  { label: '1年', value: '1Y' },
  { label: '全部', value: 'ALL' },
];

const PerformanceChart: React.FC<PerformanceChartProps> = ({
  dailyData,
  monthlyData,
  metrics,
  loading,
}) => {
  const [timeRange, setTimeRange] = useState<TimeRange>('3M');

  const filteredData = useMemo(() => {
    if (!dailyData.length) return [];
    const now = new Date();
    let cutoff = new Date();
    switch (timeRange) {
      case '1M':
        cutoff = new Date(now.getFullYear(), now.getMonth() - 1, now.getDate());
        break;
      case '3M':
        cutoff = new Date(now.getFullYear(), now.getMonth() - 3, now.getDate());
        break;
      case '6M':
        cutoff = new Date(now.getFullYear(), now.getMonth() - 6, now.getDate());
        break;
      case '1Y':
        cutoff = new Date(now.getFullYear() - 1, now.getMonth(), now.getDate());
        break;
      case 'ALL':
      default:
        return dailyData;
    }
    return dailyData.filter((d) => new Date(d.date) >= cutoff);
  }, [dailyData, timeRange]);

  const metricCards = [
    {
      label: '总收益',
      value: metrics.totalReturn,
      format: 'pct',
      icon: TrendingUp,
      positive: metrics.totalReturn >= 0,
    },
    {
      label: '年化收益',
      value: metrics.annualizedReturn,
      format: 'pct',
      icon: Zap,
      positive: metrics.annualizedReturn >= 0,
    },
    {
      label: '夏普比率',
      value: metrics.sharpeRatio,
      format: 'num',
      icon: Activity,
      positive: metrics.sharpeRatio >= 1,
    },
    {
      label: '最大回撤',
      value: metrics.maxDrawdown,
      format: 'pct',
      icon: TrendingDown,
      positive: false, // drawdown is always bad
    },
    {
      label: '胜率',
      value: metrics.winRate,
      format: 'pct',
      icon: Target,
      positive: metrics.winRate >= 0.5,
    },
    {
      label: '盈亏比',
      value: metrics.profitFactor,
      format: 'num',
      icon: Percent,
      positive: metrics.profitFactor >= 1.5,
    },
  ];

  const formatValue = (val: number, format: string): string => {
    if (format === 'pct') return `${val >= 0 ? '+' : ''}${val.toFixed(2)}%`;
    return val.toFixed(2);
  };

  const navChartData = filteredData.map((d) => ({
    date: d.date.slice(5),
    nav: d.nav,
    cumulativeReturn: d.cumulativeReturn,
  }));

  const dailyReturnChartData = filteredData.map((d) => ({
    date: d.date.slice(5),
    return: d.dailyReturn,
  }));

  // Heatmap data: last 12 months
  const heatmapData = useMemo(() => {
    const months = monthlyData.slice(-12);
    const maxAbs = Math.max(...months.map((m) => Math.abs(m.return)), 0.01);
    return months.map((m) => ({
      ...m,
      intensity: Math.abs(m.return) / maxAbs,
      positive: m.return >= 0,
    }));
  }, [monthlyData]);

  return (
    <div className="space-y-6">
      {/* Time Range Selector */}
      <div className="flex items-center gap-2">
        <Calendar className="w-4 h-4 text-[#95a5a6]" />
        <div className="flex bg-[#16213e] rounded-lg border border-[#2a3a5c] p-0.5">
          {TimeRangeOptions.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setTimeRange(opt.value)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                timeRange === opt.value
                  ? 'bg-[#d4a574]/15 text-[#d4a574]'
                  : 'text-[#95a5a6] hover:text-[#e6ddd0]'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
        {metricCards.map((m) => {
          const Icon = m.icon;
          return (
            <div
              key={m.label}
              className="bg-[#16213e] rounded-xl border border-[#2a3a5c] p-4"
            >
              <div className="flex items-center gap-2 mb-2">
                <Icon className="w-4 h-4 text-[#95a5a6]" />
                <span className="text-xs text-[#95a5a6]">{m.label}</span>
              </div>
              <div
                className={`text-lg font-bold font-mono ${
                  m.positive ? 'text-[#e74c3c]' : 'text-[#2ecc71]'
                }`}
              >
                {formatValue(m.value, m.format)}
              </div>
            </div>
          );
        })}
      </div>

      {/* Charts */}
      {loading ? (
        <div className="bg-[#16213e] rounded-xl border border-[#2a3a5c] p-12 text-center">
          <div className="flex items-center justify-center gap-2 text-[#95a5a6]">
            <div className="w-5 h-5 border-2 border-[#d4a574] border-t-transparent rounded-full animate-spin" />
            <span>加载数据中...</span>
          </div>
        </div>
      ) : filteredData.length === 0 ? (
        <div className="bg-[#16213e] rounded-xl border border-[#2a3a5c] p-12 text-center text-[#95a5a6]">
          暂无业绩数据
        </div>
      ) : (
        <>
          {/* NAV Curve */}
          <div className="bg-[#16213e] rounded-xl border border-[#2a3a5c] p-5">
            <h3 className="text-sm font-medium text-[#e6ddd0] mb-4">净值曲线</h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={navChartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <defs>
                  <linearGradient id="navGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#d4a574" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#d4a574" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3a5c" />
                <XAxis
                  dataKey="date"
                  stroke="#95a5a6"
                  tick={{ fontSize: 11, fill: '#95a5a6' }}
                  axisLine={{ stroke: '#2a3a5c' }}
                />
                <YAxis
                  stroke="#95a5a6"
                  tick={{ fontSize: 11, fill: '#95a5a6' }}
                  axisLine={{ stroke: '#2a3a5c' }}
                  domain={['auto', 'auto']}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#16213e',
                    border: '1px solid #2a3a5c',
                    borderRadius: '8px',
                    fontSize: '12px',
                    color: '#e6ddd0',
                  }}
                  formatter={(value: number) => [value.toFixed(4), '净值']}
                />
                <Area
                  type="monotone"
                  dataKey="nav"
                  stroke="#d4a574"
                  strokeWidth={2}
                  fill="url(#navGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Daily Return Bars */}
          <div className="bg-[#16213e] rounded-xl border border-[#2a3a5c] p-5">
            <h3 className="text-sm font-medium text-[#e6ddd0] mb-4">每日收益</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={dailyReturnChartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3a5c" vertical={false} />
                <XAxis
                  dataKey="date"
                  stroke="#95a5a6"
                  tick={{ fontSize: 11, fill: '#95a5a6' }}
                  axisLine={{ stroke: '#2a3a5c' }}
                />
                <YAxis
                  stroke="#95a5a6"
                  tick={{ fontSize: 11, fill: '#95a5a6' }}
                  axisLine={{ stroke: '#2a3a5c' }}
                  tickFormatter={(v: number) => `${v.toFixed(1)}%`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#16213e',
                    border: '1px solid #2a3a5c',
                    borderRadius: '8px',
                    fontSize: '12px',
                    color: '#e6ddd0',
                  }}
                  formatter={(value: number) => [`${value.toFixed(2)}%`, '日收益']}
                />
                <Bar dataKey="return" radius={[2, 2, 0, 0]}>
                  {dailyReturnChartData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={(entry.return as number) >= 0 ? '#e74c3c' : '#2ecc71'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Monthly Heatmap */}
          <div className="bg-[#16213e] rounded-xl border border-[#2a3a5c] p-5">
            <h3 className="text-sm font-medium text-[#e6ddd0] mb-4">月度收益热力图</h3>
            <div className="flex flex-wrap gap-2">
              {heatmapData.map((m) => (
                <div
                  key={m.month}
                  className="flex flex-col items-center gap-1"
                  title={`${m.month}: ${m.return >= 0 ? '+' : ''}${m.return.toFixed(2)}%`}
                >
                  <div
                    className="w-12 h-10 rounded-md flex items-center justify-center text-xs font-mono font-medium transition-all"
                    style={{
                      backgroundColor: m.positive
                        ? `rgba(231, 76, 60, ${0.1 + m.intensity * 0.4})`
                        : `rgba(46, 204, 113, ${0.1 + m.intensity * 0.4})`,
                      color: m.positive ? '#e74c3c' : '#2ecc71',
                    }}
                  >
                    {m.return.toFixed(1)}%
                  </div>
                  <span className="text-[10px] text-[#95a5a6]">{m.month}</span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default PerformanceChart;

import React, { useState } from 'react';
import PerformanceChart from './PerformanceChart';
import { apiClient } from '../api';
import { BacktestParams, BacktestResult } from '../types';
import { Play, Settings, RotateCcw, AlertCircle, CheckCircle } from 'lucide-react';

const BacktestView: React.FC = () => {
  const [params, setParams] = useState<BacktestParams>({
    startDate: new Date(Date.now() - 180 * 86400000).toISOString().split('T')[0],
    endDate: new Date().toISOString().split('T')[0],
    initialCapital: 1000000,
    strategy: 'default',
  });

  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    setSuccess(false);
    try {
      const res = await apiClient.runBacktest(params);
      setResult(res);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : '回测执行失败');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
    setSuccess(false);
  };

  return (
    <div className="space-y-6">
      {/* Params Panel */}
      <div className="bg-[#16213e] rounded-xl border border-[#2a3a5c] p-5">
        <div className="flex items-center gap-2 mb-4">
          <Settings className="w-4 h-4 text-[#d4a574]" />
          <h3 className="text-sm font-medium text-[#e6ddd0]">回测参数设置</h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="space-y-1.5">
            <label className="text-xs text-[#95a5a6]">起始日期</label>
            <input
              type="date"
              value={params.startDate}
              onChange={(e) => setParams((p) => ({ ...p, startDate: e.target.value }))}
              className="w-full px-3 py-2 bg-[#1a1a2e] border border-[#2a3a5c] rounded-lg text-sm text-[#e6ddd0] focus:outline-none focus:border-[#d4a574] font-mono"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-xs text-[#95a5a6]">结束日期</label>
            <input
              type="date"
              value={params.endDate}
              onChange={(e) => setParams((p) => ({ ...p, endDate: e.target.value }))}
              className="w-full px-3 py-2 bg-[#1a1a2e] border border-[#2a3a5c] rounded-lg text-sm text-[#e6ddd0] focus:outline-none focus:border-[#d4a574] font-mono"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-xs text-[#95a5a6]">初始资金 (元)</label>
            <input
              type="number"
              value={params.initialCapital}
              onChange={(e) => setParams((p) => ({ ...p, initialCapital: Number(e.target.value) }))}
              className="w-full px-3 py-2 bg-[#1a1a2e] border border-[#2a3a5c] rounded-lg text-sm text-[#e6ddd0] focus:outline-none focus:border-[#d4a574] font-mono"
              step={100000}
              min={10000}
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-xs text-[#95a5a6]">策略</label>
            <select
              value={params.strategy}
              onChange={(e) => setParams((p) => ({ ...p, strategy: e.target.value }))}
              className="w-full px-3 py-2 bg-[#1a1a2e] border border-[#2a3a5c] rounded-lg text-sm text-[#e6ddd0] focus:outline-none focus:border-[#d4a574]"
            >
              <option value="default">默认多因子策略</option>
              <option value="momentum">动量策略</option>
              <option value="value">价值策略</option>
              <option value="quality">质量策略</option>
            </select>
          </div>
        </div>

        <div className="flex items-center gap-3 mt-5">
          <button
            onClick={handleRun}
            disabled={loading}
            className="flex items-center gap-2 px-5 py-2.5 bg-[#d4a574] text-[#1a1a2e] rounded-lg text-sm font-medium hover:bg-[#e8c9a0] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-[#1a1a2e] border-t-transparent rounded-full animate-spin" />
                回测中...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                执行回测
              </>
            )}
          </button>

          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-4 py-2.5 bg-[#2a3a5c]/50 text-[#95a5a6] rounded-lg text-sm hover:bg-[#2a3a5c] hover:text-[#e6ddd0] transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            重置
          </button>

          {success && (
            <div className="flex items-center gap-1.5 text-sm text-[#2ecc71]">
              <CheckCircle className="w-4 h-4" />
              回测完成
            </div>
          )}
        </div>

        {error && (
          <div className="mt-4 flex items-center gap-2 px-4 py-3 bg-[#e74c3c]/10 border border-[#e74c3c]/20 rounded-lg">
            <AlertCircle className="w-4 h-4 text-[#e74c3c] flex-shrink-0" />
            <span className="text-sm text-[#e74c3c]">{error}</span>
          </div>
        )}
      </div>

      {/* Results */}
      {result && (
        <PerformanceChart
          dailyData={result.dailyData}
          monthlyData={result.monthlyData}
          metrics={result.metrics}
          loading={false}
        />
      )}

      {/* Empty State */}
      {!result && !loading && (
        <div className="bg-[#16213e] rounded-xl border border-[#2a3a5c] p-12 text-center">
          <Play className="w-10 h-10 text-[#2a3a5c] mx-auto mb-3" />
          <p className="text-sm text-[#95a5a6]">设置参数后点击「执行回测」开始分析</p>
        </div>
      )}
    </div>
  );
};

export default BacktestView;

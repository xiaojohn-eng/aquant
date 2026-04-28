import React, { useContext } from 'react';
import { TrendingUp, Activity, BarChart3, Globe } from 'lucide-react';
import StockTable from '../components/StockTable';
import { useAutoRefresh } from '../hooks/useAutoRefresh';
import { apiClient } from '../api';
import { AppContext } from '../App';
import { StockRecommendation, MarketOverview } from '../types';

const RecommendationsPage: React.FC = () => {
  const { wsConnected, lastMessage } = useContext(AppContext);

  const {
    data: recs,
    loading,
    lastUpdated,
  } = useAutoRefresh<StockRecommendation[]>({
    fetchFn: apiClient.fetchStockRecommendations,
    enabled: true,
  });

  const { data: marketData } = useAutoRefresh<MarketOverview[]>({
    fetchFn: apiClient.fetchMarketOverview,
    enabled: true,
    marketOpenInterval: 60000,
  });

  // Merge WebSocket data if available
  const displayData = React.useMemo(() => {
    if (lastMessage?.type === 'stock_update') {
      return lastMessage.data;
    }
    return recs || [];
  }, [recs, lastMessage]);

  return (
    <div className="space-y-6">
      {/* Market Overview Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {marketData?.map((mkt) => (
          <div
            key={mkt.indexName}
            className="bg-[#16213e] rounded-xl border border-[#2a3a5c] p-4"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-[#95a5a6] flex items-center gap-1.5">
                <Globe className="w-3.5 h-3.5" />
                {mkt.indexName}
              </span>
              <span className="text-[10px] text-[#95a5a6]/60">
                {lastUpdated?.toLocaleTimeString('zh-CN')}
              </span>
            </div>
            <div className="flex items-end justify-between">
              <span className="text-xl font-bold font-mono text-[#e6ddd0]">
                {mkt.indexValue.toFixed(2)}
              </span>
              <div className="text-right">
                <span
                  className={`text-sm font-mono font-medium ${
                    mkt.change >= 0 ? 'text-[#e74c3c]' : 'text-[#2ecc71]'
                  }`}
                >
                  {mkt.change >= 0 ? '+' : ''}{mkt.change.toFixed(2)}
                </span>
                <span
                  className={`text-xs font-mono ml-1 ${
                    mkt.changePct >= 0 ? 'text-[#e74c3c]' : 'text-[#2ecc71]'
                  }`}
                >
                  ({mkt.changePct >= 0 ? '+' : ''}{mkt.changePct.toFixed(2)}%)
                </span>
              </div>
            </div>
            <div className="mt-2 text-[10px] text-[#95a5a6]/60">
              成交量: {(mkt.volume / 1e8).toFixed(2)}亿
            </div>
          </div>
        ))}

        {/* Fallback mock cards if no market data */}
        {(!marketData || marketData.length === 0) &&
          ['上证指数', '深证成指', '创业板指', '科创50'].map((name, i) => {
            const mockValue = [3050.23, 9800.56, 1950.78, 850.32][i];
            const mockChange = [12.5, -8.3, 5.2, -3.1][i];
            return (
              <div
                key={name}
                className="bg-[#16213e] rounded-xl border border-[#2a3a5c] p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-[#95a5a6] flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5" />
                    {name}
                  </span>
                </div>
                <div className="flex items-end justify-between">
                  <span className="text-xl font-bold font-mono text-[#e6ddd0]">
                    {mockValue.toFixed(2)}
                  </span>
                  <span
                    className={`text-sm font-mono font-medium ${
                      mockChange >= 0 ? 'text-[#e74c3c]' : 'text-[#2ecc71]'
                    }`}
                  >
                    {mockChange >= 0 ? '+' : ''}{mockChange.toFixed(2)}
                  </span>
                </div>
              </div>
            );
          })}
      </div>

      {/* Stock Recommendations */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-[#d4a574]" />
            <h2 className="text-base font-medium text-[#e6ddd0]">今日推荐</h2>
            <span className="text-xs text-[#95a5a6]">Top 20</span>
          </div>
          <div className="flex items-center gap-3">
            {wsConnected && (
              <span className="flex items-center gap-1 text-xs text-[#2ecc71]">
                <div className="w-1.5 h-1.5 rounded-full bg-[#2ecc71] animate-pulse" />
                实时推送
              </span>
            )}
            {lastUpdated && (
              <span className="text-xs text-[#95a5a6]">
                更新于 {lastUpdated.toLocaleTimeString('zh-CN')}
              </span>
            )}
          </div>
        </div>

        <StockTable data={displayData} loading={loading} />
      </div>
    </div>
  );
};

export default RecommendationsPage;

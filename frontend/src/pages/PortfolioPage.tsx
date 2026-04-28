import React, { useContext, useState } from 'react';
import { Wallet, RefreshCw } from 'lucide-react';
import Portfolio from '../components/Portfolio';
import { useAutoRefresh } from '../hooks/useAutoRefresh';
import { apiClient } from '../api';
import { AppContext } from '../App';
import { PortfolioPosition } from '../types';

const PortfolioPage: React.FC = () => {
  const { lastMessage } = useContext(AppContext);
  const [sellMessage, setSellMessage] = useState<string | null>(null);

  const {
    data: positions,
    loading,
    refresh,
    lastUpdated,
  } = useAutoRefresh<PortfolioPosition[]>({
    fetchFn: apiClient.fetchPortfolio,
    enabled: true,
  });

  const displayData = React.useMemo(() => {
    if (lastMessage?.type === 'portfolio_update') {
      return lastMessage.data;
    }
    return positions || [];
  }, [positions, lastMessage]);

  const handleSell = async (code: string) => {
    try {
      const res = await apiClient.sellStock(code);
      setSellMessage(res.message);
      refresh();
      setTimeout(() => setSellMessage(null), 3000);
    } catch {
      setSellMessage('卖出请求失败');
      setTimeout(() => setSellMessage(null), 3000);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Wallet className="w-5 h-5 text-[#d4a574]" />
          <h2 className="text-base font-medium text-[#e6ddd0]">持仓监控</h2>
        </div>
        <div className="flex items-center gap-3">
          {sellMessage && (
            <span className="text-xs text-[#d4a574] bg-[#d4a574]/10 px-3 py-1 rounded-full">
              {sellMessage}
            </span>
          )}
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

      <Portfolio
        data={displayData}
        loading={loading}
        availableFunds={500000}
        onSell={handleSell}
      />
    </div>
  );
};

export default PortfolioPage;

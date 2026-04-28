import React, { useState } from 'react';
import { DollarSign, TrendingUp, Package, CreditCard, AlertCircle } from 'lucide-react';
import { PortfolioPosition } from '../types';

interface PortfolioProps {
  data: PortfolioPosition[];
  loading?: boolean;
  availableFunds?: number;
  onSell?: (code: string) => void;
}

const Portfolio: React.FC<PortfolioProps> = ({
  data,
  loading,
  availableFunds = 0,
  onSell,
}) => {
  const [confirmSell, setConfirmSell] = useState<string | null>(null);

  const totalAsset = data.reduce((sum, p) => sum + (p.totalValue ?? p.currentPrice * (p.quantity ?? 1)), 0) + availableFunds;
  const totalPnl = data.reduce((sum, p) => sum + p.pnl * (p.quantity ?? 1), 0);
  const todayPnl = data.reduce((sum, p) => sum + (p.currentPrice - (p.currentPrice * 0.99)) * (p.quantity ?? 1), 0); // placeholder

  const summaryCards = [
    {
      title: '总资产',
      value: totalAsset,
      prefix: '¥',
      icon: DollarSign,
      color: 'text-[#e6ddd0]',
    },
    {
      title: '今日盈亏',
      value: todayPnl,
      prefix: '¥',
      icon: TrendingUp,
      color: todayPnl >= 0 ? 'text-[#e74c3c]' : 'text-[#2ecc71]',
    },
    {
      title: '持仓数量',
      value: data.length,
      prefix: '',
      suffix: '只',
      icon: Package,
      color: 'text-[#d4a574]',
    },
    {
      title: '可用资金',
      value: availableFunds,
      prefix: '¥',
      icon: CreditCard,
      color: 'text-[#e6ddd0]',
    },
  ];

  const handleSellClick = (code: string) => {
    if (confirmSell === code) {
      onSell?.(code);
      setConfirmSell(null);
    } else {
      setConfirmSell(code);
      setTimeout(() => setConfirmSell(null), 3000);
    }
  };

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {summaryCards.map((card) => {
          const Icon = card.icon;
          return (
            <div
              key={card.title}
              className="bg-[#16213e] rounded-xl border border-[#2a3a5c] p-4 flex items-center gap-4"
            >
              <div className="w-10 h-10 rounded-lg bg-[#2a3a5c]/50 flex items-center justify-center flex-shrink-0">
                <Icon className="w-5 h-5 text-[#d4a574]" />
              </div>
              <div>
                <p className="text-xs text-[#95a5a6] mb-0.5">{card.title}</p>
                <p className={`text-lg font-bold font-mono ${card.color}`}>
                  {typeof card.value === 'number' && card.prefix === '¥'
                    ? `${card.prefix}${card.value.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                    : `${card.prefix}${card.value}${card.suffix || ''}`}
                </p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Holdings Table */}
      <div className="bg-[#16213e] rounded-xl border border-[#2a3a5c] overflow-hidden">
        <div className="px-4 py-3 border-b border-[#2a3a5c] flex items-center justify-between">
          <h3 className="text-sm font-medium text-[#e6ddd0]">当前持仓</h3>
          <span className="text-xs text-[#95a5a6]">
            总盈亏:{' '}
            <span className={`font-mono font-medium ${totalPnl >= 0 ? 'text-[#e74c3c]' : 'text-[#2ecc71]'}`}>
              {totalPnl >= 0 ? '+' : ''}
              {totalPnl.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-[#2a3a5c]">
                <th className="px-4 py-3 text-left text-xs font-medium text-[#95a5a6] uppercase">代码</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-[#95a5a6] uppercase">名称</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-[#95a5a6] uppercase">买入价</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-[#95a5a6] uppercase">现价</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-[#95a5a6] uppercase">数量</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-[#95a5a6] uppercase">盈亏额</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-[#95a5a6] uppercase">盈亏率</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-[#95a5a6] uppercase w-24">操作</th>
              </tr>
            </thead>
            <tbody>
              {loading && data.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-4 py-12 text-center text-[#95a5a6]">
                    <div className="flex items-center justify-center gap-2">
                      <div className="w-5 h-5 border-2 border-[#d4a574] border-t-transparent rounded-full animate-spin" />
                      <span>加载中...</span>
                    </div>
                  </td>
                </tr>
              ) : data.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-4 py-12 text-center text-[#95a5a6]">
                    <div className="flex flex-col items-center gap-2">
                      <AlertCircle className="w-8 h-8 text-[#95a5a6]/50" />
                      <span>暂无持仓</span>
                    </div>
                  </td>
                </tr>
              ) : (
                data.map((pos) => {
                  const isProfit = pos.pnl >= 0;
                  const qty = pos.quantity ?? 1;
                  return (
                    <tr
                      key={pos.code}
                      className="border-b border-[#2a3a5c]/50 hover:bg-[#1e2a4a]/60 transition-colors"
                    >
                      <td className="px-4 py-3">
                        <span className="font-mono text-sm text-[#e6ddd0]">{pos.code}</span>
                      </td>
                      <td className="px-4 py-3">
                        <span className="text-sm text-[#e6ddd0]">{pos.name}</span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="text-sm font-mono text-[#95a5a6]">
                          {pos.buyPrice.toFixed(2)}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="text-sm font-mono text-[#e6ddd0]">
                          {pos.currentPrice.toFixed(2)}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="text-sm font-mono text-[#e6ddd0]">{qty}</span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className={`text-sm font-mono font-medium ${isProfit ? 'text-[#e74c3c]' : 'text-[#2ecc71]'}`}>
                          {isProfit ? '+' : ''}
                          {(pos.pnl * qty).toLocaleString('zh-CN', { minimumFractionDigits: 2 })}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span
                          className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-medium ${
                            isProfit ? 'bg-[#e74c3c]/10 text-[#e74c3c]' : 'bg-[#2ecc71]/10 text-[#2ecc71]'
                          }`}
                        >
                          {isProfit ? '+' : ''}
                          {pos.pnlPct.toFixed(2)}%
                        </span>
                      </td>
                      <td className="px-4 py-3 text-center">
                        <button
                          onClick={() => handleSellClick(pos.code)}
                          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
                            confirmSell === pos.code
                              ? 'bg-[#e74c3c]/20 text-[#e74c3c] border border-[#e74c3c]/40'
                              : 'bg-[#2a3a5c]/50 text-[#95a5a6] hover:bg-[#e74c3c]/10 hover:text-[#e74c3c] border border-transparent'
                          }`}
                        >
                          {confirmSell === pos.code ? '确认卖出' : '卖出'}
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Portfolio;

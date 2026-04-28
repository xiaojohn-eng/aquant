import React, { useState, useMemo } from 'react';
import {
  ChevronDown,
  ChevronUp,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Search,
  Filter,
} from 'lucide-react';
import { StockRecommendation } from '../types';

type SortField = 'rank' | 'score' | 'expectedReturn' | 'code';
type SortDirection = 'asc' | 'desc';

interface StockTableProps {
  data: StockRecommendation[];
  loading?: boolean;
}

const StockTable: React.FC<StockTableProps> = ({ data, loading }) => {
  const [sortField, setSortField] = useState<SortField>('rank');
  const [sortDir, setSortDir] = useState<SortDirection>('asc');
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [industryFilter, setIndustryFilter] = useState<string>('all');

  const industries = useMemo(() => {
    const set = new Set<string>();
    data.forEach((d) => d.industry && set.add(d.industry));
    return Array.from(set);
  }, [data]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDir(field === 'rank' || field === 'score' ? 'desc' : 'asc');
    }
  };

  const sortedData = useMemo(() => {
    let filtered = [...data];

    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(
        (s) =>
          s.code.toLowerCase().includes(term) ||
          s.name.toLowerCase().includes(term)
      );
    }

    if (industryFilter !== 'all') {
      filtered = filtered.filter((s) => s.industry === industryFilter);
    }

    filtered.sort((a, b) => {
      let aVal: number | string;
      let bVal: number | string;

      switch (sortField) {
        case 'rank':
          aVal = a.rank;
          bVal = b.rank;
          break;
        case 'score':
          aVal = a.score;
          bVal = b.score;
          break;
        case 'expectedReturn':
          aVal = a.expectedReturn ?? 0;
          bVal = b.expectedReturn ?? 0;
          break;
        case 'code':
          aVal = a.code;
          bVal = b.code;
          break;
        default:
          aVal = a.rank;
          bVal = b.rank;
      }

      if (typeof aVal === 'string') {
        return sortDir === 'asc'
          ? aVal.localeCompare(bVal as string)
          : (bVal as string).localeCompare(aVal);
      }
      return sortDir === 'asc' ? aVal - (bVal as number) : (bVal as number) - aVal;
    });

    return filtered.slice(0, 20);
  }, [data, sortField, sortDir, searchTerm, industryFilter]);

  const getScoreColor = (score: number): string => {
    if (score >= 90) return 'text-[#e74c3c] bg-[#e74c3c]/10';
    if (score >= 75) return 'text-[#d4a574] bg-[#d4a574]/10';
    if (score >= 60) return 'text-[#e6ddd0] bg-[#e6ddd0]/10';
    return 'text-[#95a5a6] bg-[#95a5a6]/10';
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return <ArrowUpDown className="w-3.5 h-3.5 text-[#95a5a6]" />;
    return sortDir === 'asc' ? (
      <ArrowUp className="w-3.5 h-3.5 text-[#d4a574]" />
    ) : (
      <ArrowDown className="w-3.5 h-3.5 text-[#d4a574]" />
    );
  };

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#95a5a6]" />
          <input
            type="text"
            placeholder="搜索代码/名称"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-9 pr-4 py-2 bg-[#16213e] border border-[#2a3a5c] rounded-lg text-sm text-[#e6ddd0] placeholder-[#95a5a6] focus:outline-none focus:border-[#d4a574] w-48"
          />
        </div>

        <div className="relative flex items-center gap-2">
          <Filter className="w-4 h-4 text-[#95a5a6]" />
          <select
            value={industryFilter}
            onChange={(e) => setIndustryFilter(e.target.value)}
            className="py-2 px-3 bg-[#16213e] border border-[#2a3a5c] rounded-lg text-sm text-[#e6ddd0] focus:outline-none focus:border-[#d4a574]"
          >
            <option value="all">全部行业</option>
            {industries.map((ind) => (
              <option key={ind} value={ind}>
                {ind}
              </option>
            ))}
          </select>
        </div>

        <div className="ml-auto text-sm text-[#95a5a6]">
          共 <span className="text-[#d4a574] font-mono">{sortedData.length}</span> 条
        </div>
      </div>

      {/* Table */}
      <div className="bg-[#16213e] rounded-xl border border-[#2a3a5c] overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-[#2a3a5c]">
                <th
                  className="px-4 py-3 text-left text-xs font-medium text-[#95a5a6] uppercase tracking-wider cursor-pointer hover:text-[#e6ddd0] transition-colors"
                  onClick={() => handleSort('rank')}
                >
                  <div className="flex items-center gap-1">
                    排名 <SortIcon field="rank" />
                  </div>
                </th>
                <th
                  className="px-4 py-3 text-left text-xs font-medium text-[#95a5a6] uppercase tracking-wider cursor-pointer hover:text-[#e6ddd0] transition-colors"
                  onClick={() => handleSort('code')}
                >
                  <div className="flex items-center gap-1">
                    代码 <SortIcon field="code" />
                  </div>
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-[#95a5a6] uppercase tracking-wider">
                  名称
                </th>
                <th
                  className="px-4 py-3 text-left text-xs font-medium text-[#95a5a6] uppercase tracking-wider cursor-pointer hover:text-[#e6ddd0] transition-colors"
                  onClick={() => handleSort('score')}
                >
                  <div className="flex items-center gap-1">
                    综合评分 <SortIcon field="score" />
                  </div>
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-[#95a5a6] uppercase tracking-wider">
                  行业
                </th>
                <th
                  className="px-4 py-3 text-right text-xs font-medium text-[#95a5a6] uppercase tracking-wider cursor-pointer hover:text-[#e6ddd0] transition-colors"
                  onClick={() => handleSort('expectedReturn')}
                >
                  <div className="flex items-center justify-end gap-1">
                    预期收益 <SortIcon field="expectedReturn" />
                  </div>
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-[#95a5a6] uppercase tracking-wider w-10">
                  {/* expand */}
                </th>
              </tr>
            </thead>
            <tbody>
              {loading && data.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-12 text-center text-[#95a5a6]">
                    <div className="flex items-center justify-center gap-2">
                      <div className="w-5 h-5 border-2 border-[#d4a574] border-t-transparent rounded-full animate-spin" />
                      <span>加载中...</span>
                    </div>
                  </td>
                </tr>
              ) : sortedData.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-12 text-center text-[#95a5a6]">
                    暂无数据
                  </td>
                </tr>
              ) : (
                sortedData.map((stock) => {
                  const isExpanded = expandedRow === stock.code;
                  return (
                    <React.Fragment key={stock.code}>
                      <tr
                        className={`border-b border-[#2a3a5c]/50 hover:bg-[#1e2a4a]/60 transition-colors cursor-pointer ${
                          stock.rank <= 3 ? 'bg-[#d4a574]/5' : ''
                        }`}
                        onClick={() => setExpandedRow(isExpanded ? null : stock.code)}
                      >
                        <td className="px-4 py-3">
                          <span
                            className={`inline-flex items-center justify-center w-7 h-7 rounded-full text-xs font-bold font-mono ${
                              stock.rank === 1
                                ? 'bg-[#d4a574]/20 text-[#d4a574]'
                                : stock.rank === 2
                                ? 'bg-[#95a5a6]/20 text-[#95a5a6]'
                                : stock.rank === 3
                                ? 'bg-[#b8894a]/20 text-[#b8894a]'
                                : 'text-[#95a5a6]'
                            }`}
                          >
                            {stock.rank}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="font-mono text-sm text-[#e6ddd0]">{stock.code}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm text-[#e6ddd0] font-medium">{stock.name}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span
                            className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold font-mono ${getScoreColor(
                              stock.score
                            )}`}
                          >
                            {stock.score.toFixed(1)}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-xs text-[#95a5a6]">{stock.industry || '-'}</span>
                        </td>
                        <td className="px-4 py-3 text-right">
                          <span
                            className={`text-sm font-mono font-medium ${
                              (stock.expectedReturn ?? 0) > 0
                                ? 'text-[#e74c3c]'
                                : (stock.expectedReturn ?? 0) < 0
                                ? 'text-[#2ecc71]'
                                : 'text-[#95a5a6]'
                            }`}
                          >
                            {(stock.expectedReturn ?? 0) > 0 ? '+' : ''}
                            {(stock.expectedReturn ?? 0).toFixed(2)}%
                          </span>
                        </td>
                        <td className="px-4 py-3 text-center">
                          {isExpanded ? (
                            <ChevronUp className="w-4 h-4 text-[#d4a574] mx-auto" />
                          ) : (
                            <ChevronDown className="w-4 h-4 text-[#95a5a6] mx-auto" />
                          )}
                        </td>
                      </tr>

                      {/* Expanded Detail */}
                      {isExpanded && (
                        <tr>
                          <td colSpan={7} className="px-4 py-4 bg-[#1e2a4a]/40">
                            <div className="space-y-3">
                              <div>
                                <h4 className="text-xs font-medium text-[#95a5a6] mb-2 uppercase tracking-wider">
                                  推荐理由
                                </h4>
                                <ul className="space-y-1.5">
                                  {stock.reasons.map((reason, idx) => (
                                    <li
                                      key={idx}
                                      className="text-sm text-[#e6ddd0] flex items-start gap-2"
                                    >
                                      <span className="text-[#d4a574] mt-1">•</span>
                                      {reason}
                                    </li>
                                  ))}
                                </ul>
                              </div>

                              {stock.factors && Object.keys(stock.factors).length > 0 && (
                                <div>
                                  <h4 className="text-xs font-medium text-[#95a5a6] mb-2 uppercase tracking-wider">
                                    因子分析
                                  </h4>
                                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                                    {Object.entries(stock.factors).map(([key, value]) => (
                                      <div
                                        key={key}
                                        className="bg-[#16213e] rounded-lg px-3 py-2 border border-[#2a3a5c]/50"
                                      >
                                        <div className="text-xs text-[#95a5a6] mb-1">{key}</div>
                                        <div className="text-sm font-mono font-medium text-[#e6ddd0]">
                                          {typeof value === 'number' ? value.toFixed(3) : value}
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {stock.marketCap && (
                                <div className="flex items-center gap-4">
                                  <span className="text-xs text-[#95a5a6]">
                                    市值:{' '}
                                    <span className="text-[#e6ddd0] font-mono">
                                      {(stock.marketCap / 1e8).toFixed(2)}亿
                                    </span>
                                  </span>
                                </div>
                              )}
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
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

export default StockTable;

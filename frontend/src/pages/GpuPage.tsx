import React, { useContext } from 'react';
import { Cpu, RefreshCw } from 'lucide-react';
import GpuMonitor from '../components/GpuMonitor';
import { useAutoRefresh } from '../hooks/useAutoRefresh';
import { apiClient } from '../api';
import { AppContext } from '../App';
import { GpuStatus, GpuHistoryPoint } from '../types';

const GpuPage: React.FC = () => {
  const { lastMessage } = useContext(AppContext);

  const {
    data: gpuStatus,
    loading: statusLoading,
    refresh: refreshStatus,
    lastUpdated,
  } = useAutoRefresh<GpuStatus>({
    fetchFn: apiClient.fetchGpuStatus,
    enabled: true,
    marketOpenInterval: 5000,
    marketCloseInterval: 30000,
  });

  const {
    data: gpuHistory,
    loading: historyLoading,
    refresh: refreshHistory,
  } = useAutoRefresh<GpuHistoryPoint[]>({
    fetchFn: () => apiClient.fetchGpuHistory(60),
    enabled: true,
    marketOpenInterval: 10000,
    marketCloseInterval: 60000,
  });

  const displayStatus = React.useMemo(() => {
    if (lastMessage?.type === 'gpu_update') {
      return lastMessage.data;
    }
    return gpuStatus || {
      utilization: 45,
      memoryUsed: 32,
      memoryTotal: 80,
      temperature: 62,
      power: 280,
      deviceName: 'NVIDIA H100 80GB',
      fanSpeed: 65,
      clockSpeed: 1410,
    };
  }, [gpuStatus, lastMessage]);

  const displayHistory = React.useMemo(() => {
    return gpuHistory || [];
  }, [gpuHistory]);

  const refreshAll = () => {
    refreshStatus();
    refreshHistory();
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Cpu className="w-5 h-5 text-[#d4a574]" />
          <h2 className="text-base font-medium text-[#e6ddd0]">GPU监控</h2>
          <span className="text-xs text-[#95a5a6]">H100 80GB</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={refreshAll}
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

      <GpuMonitor
        status={displayStatus}
        history={displayHistory}
        loading={statusLoading || historyLoading}
      />
    </div>
  );
};

export default GpuPage;

import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import { Cpu, Thermometer, Zap, HardDrive } from 'lucide-react';
import { GpuStatus, GpuHistoryPoint } from '../types';

interface GpuMonitorProps {
  status: GpuStatus;
  history: GpuHistoryPoint[];
  loading?: boolean;
}

const CircularGauge: React.FC<{ value: number; label: string; color: string; size?: number }> = ({
  value,
  label,
  color,
  size = 120,
}) => {
  const radius = (size - 10) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="transform -rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="#2a3a5c"
            strokeWidth={8}
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={8}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{ transition: 'stroke-dashoffset 0.5s ease-out' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-xl font-bold font-mono" style={{ color }}>
            {Math.round(value)}%
          </span>
        </div>
      </div>
      <span className="text-xs text-[#95a5a6]">{label}</span>
    </div>
  );
};

const LinearGauge: React.FC<{ value: number; max: number; label: string; color: string; unit: string }> = ({
  value,
  max,
  label,
  color,
  unit,
}) => {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-xs text-[#95a5a6]">{label}</span>
        <span className="text-xs font-mono" style={{ color }}>
          {value.toFixed(1)}{unit} / {max.toFixed(1)}{unit}
        </span>
      </div>
      <div className="h-2 bg-[#2a3a5c] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
};

const GpuMonitor: React.FC<GpuMonitorProps> = ({ status, history, loading }) => {
  const memUsed = (status.memory_used_mb ?? status.memoryUsed ?? 0) / 1024;
  const memTotal = (status.memory_total_mb ?? status.memoryTotal ?? 81920) / 1024;
  const gpuUtil = status.utilization_gpu_pct ?? status.utilization ?? 0;
  const gpuTemp = status.temperature_c ?? status.temperature ?? 0;
  const gpuPower = status.power_draw_w ?? status.power ?? 0;
  const deviceName = status.name ?? status.deviceName ?? 'NVIDIA H100 80GB';

  const memPct = useMemo(
    () => (memTotal > 0 ? (memUsed / memTotal) * 100 : 0),
    [memUsed, memTotal]
  );

  const tempColor = gpuTemp >= 80 ? '#e74c3c' : gpuTemp >= 60 ? '#d4a574' : '#2ecc71';
  const utilColor = gpuUtil >= 90 ? '#e74c3c' : gpuUtil >= 50 ? '#d4a574' : '#2ecc71';

  const chartData = useMemo(() => {
    return history.map((h) => ({
      time: h.timestamp.split(' ')[1] || h.timestamp.slice(-8),
      utilization: h.utilization,
      memory: (h.memory_used ?? h.memoryUsed ?? 0) / (memTotal || 1) * 100,
      temperature: h.temperature,
    }));
  }, [history, memTotal]);

  return (
    <div className="space-y-6">
      {/* Device Info */}
      <div className="bg-[#16213e] rounded-xl border border-[#2a3a5c] p-5">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-lg bg-[#d4a574]/10 flex items-center justify-center">
            <Cpu className="w-5 h-5 text-[#d4a574]" />
          </div>
          <div>
            <h3 className="text-sm font-medium text-[#e6ddd0]">
              {deviceName}
            </h3>
            <p className="text-xs text-[#95a5a6]">GPU计算设备状态监控</p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                gpuUtil > 0 ? 'bg-[#2ecc71] animate-pulse' : 'bg-[#95a5a6]'
              }`}
            />
            <span className="text-xs text-[#95a5a6]">
              {gpuUtil > 0 ? '运行中' : '空闲'}
            </span>
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-8">
            <div className="w-5 h-5 border-2 border-[#d4a574] border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Utilization Gauge */}
            <div className="bg-[#1e2a4a]/50 rounded-xl p-4 flex flex-col items-center">
              <CircularGauge
                value={gpuUtil}
                label="GPU利用率"
                color={utilColor}
              />
            </div>

            {/* Memory Gauge */}
            <div className="bg-[#1e2a4a]/50 rounded-xl p-4 flex flex-col items-center">
              <CircularGauge
                value={memPct}
                label="显存使用"
                color={memPct >= 80 ? '#e74c3c' : memPct >= 50 ? '#d4a574' : '#2ecc71'}
              />
            </div>

            {/* Temperature */}
            <div className="bg-[#1e2a4a]/50 rounded-xl p-4 flex flex-col items-center">
              <div className="flex items-center gap-2 mb-3">
                <Thermometer className="w-4 h-4 text-[#95a5a6]" />
                <span className="text-xs text-[#95a5a6]">温度</span>
              </div>
              <span
                className="text-3xl font-bold font-mono"
                style={{ color: tempColor }}
              >
                {gpuTemp}°C
              </span>
              <div className="mt-3 w-full">
                <div className="h-1.5 bg-[#2a3a5c] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${Math.min((gpuTemp / 100) * 100, 100)}%`,
                      backgroundColor: tempColor,
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Power */}
            <div className="bg-[#1e2a4a]/50 rounded-xl p-4 flex flex-col items-center">
              <div className="flex items-center gap-2 mb-3">
                <Zap className="w-4 h-4 text-[#95a5a6]" />
                <span className="text-xs text-[#95a5a6]">功耗</span>
              </div>
              <span className="text-3xl font-bold font-mono text-[#e6ddd0]">
                {gpuPower}W
              </span>
              <div className="mt-3 w-full">
                <div className="h-1.5 bg-[#2a3a5c] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full bg-[#d4a574] transition-all duration-500"
                    style={{
                      width: `${Math.min((gpuPower / 700) * 100, 100)}%`,
                    }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Linear Gauges */}
        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
          <LinearGauge
            value={memUsed}
            max={memTotal || 80}
            label="显存占用"
            color={memPct >= 80 ? '#e74c3c' : '#d4a574'}
            unit="GB"
          />
          {(status.fan_speed_pct ?? status.fanSpeed) !== undefined && (
            <LinearGauge
              value={status.fan_speed_pct ?? status.fanSpeed ?? 0}
              max={100}
              label="风扇转速"
              color="#2ecc71"
              unit="%"
            />
          )}
        </div>
      </div>

      {/* History Charts */}
      {chartData.length > 0 && (
        <div className="bg-[#16213e] rounded-xl border border-[#2a3a5c] p-5">
          <h3 className="text-sm font-medium text-[#e6ddd0] mb-4">GPU利用率历史</h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <defs>
                <linearGradient id="gpuUtilGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#d4a574" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#d4a574" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3a5c" />
              <XAxis
                dataKey="time"
                stroke="#95a5a6"
                tick={{ fontSize: 10, fill: '#95a5a6' }}
                axisLine={{ stroke: '#2a3a5c' }}
              />
              <YAxis
                stroke="#95a5a6"
                tick={{ fontSize: 11, fill: '#95a5a6' }}
                axisLine={{ stroke: '#2a3a5c' }}
                domain={[0, 100]}
                tickFormatter={(v: number) => `${v}%`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#16213e',
                  border: '1px solid #2a3a5c',
                  borderRadius: '8px',
                  fontSize: '12px',
                  color: '#e6ddd0',
                }}
              />
              <Area
                type="monotone"
                dataKey="utilization"
                stroke="#d4a574"
                strokeWidth={2}
                fill="url(#gpuUtilGradient)"
                name="利用率"
              />
            </AreaChart>
          </ResponsiveContainer>

          <h3 className="text-sm font-medium text-[#e6ddd0] mb-4 mt-6">温度历史</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3a5c" />
              <XAxis
                dataKey="time"
                stroke="#95a5a6"
                tick={{ fontSize: 10, fill: '#95a5a6' }}
                axisLine={{ stroke: '#2a3a5c' }}
              />
              <YAxis
                stroke="#95a5a6"
                tick={{ fontSize: 11, fill: '#95a5a6' }}
                axisLine={{ stroke: '#2a3a5c' }}
                domain={['auto', 'auto']}
                tickFormatter={(v: number) => `${v}°C`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#16213e',
                  border: '1px solid #2a3a5c',
                  borderRadius: '8px',
                  fontSize: '12px',
                  color: '#e6ddd0',
                }}
              />
              <Line
                type="monotone"
                dataKey="temperature"
                stroke="#e74c3c"
                strokeWidth={2}
                dot={false}
                name="温度"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

export default GpuMonitor;

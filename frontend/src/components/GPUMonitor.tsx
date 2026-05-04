import type { GPUStatus } from '../types'

interface Props {
  data: GPUStatus[]
}

export default function GPUMonitor({ data }: Props) {
  const gpus = data.length > 0 ? data : [{
    device_id: 0,
    name: 'NVIDIA Grace Hopper',
    utilization_pct: 0,
    memory_used_mb: 0,
    memory_total_mb: 624000,
    temperature_c: 45,
  }]

  return (
    <div className="card">
      <h2 className="text-lg font-bold mb-4">🖥️ GPU Status</h2>
      <div className="space-y-3">
        {gpus.map((gpu) => (
          <div key={gpu.device_id} className="bg-gray-50 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold">{gpu.name}</span>
              <span className={`text-xs px-2 py-0.5 rounded-full ${gpu.temperature_c > 80 ? 'bg-red-100 text-red-700' : gpu.temperature_c > 60 ? 'bg-amber-100 text-amber-700' : 'bg-emerald-100 text-emerald-700'}`}>
                {gpu.temperature_c}°C
              </span>
            </div>

            {/* Utilization */}
            <div className="mb-2">
              <div className="flex justify-between text-xs mb-1">
                <span className="text-gray-500">Utilization</span>
                <span className="font-mono">{gpu.utilization_pct.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-emerald-500 h-2 rounded-full transition-all" style={{ width: `${gpu.utilization_pct}%` }} />
              </div>
            </div>

            {/* Memory */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-gray-500">Memory</span>
                <span className="font-mono">{(gpu.memory_used_mb / 1024).toFixed(1)} / {(gpu.memory_total_mb / 1024).toFixed(0)} GB</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all"
                  style={{ width: `${(gpu.memory_used_mb / gpu.memory_total_mb) * 100}%` }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

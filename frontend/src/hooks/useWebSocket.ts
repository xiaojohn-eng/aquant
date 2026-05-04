import { useEffect, useRef, useState, useCallback } from 'react'
import type { MarketSnapshot, GPUStatus } from '../types'

export function useWebSocket() {
  const [marketData, setMarketData] = useState<MarketSnapshot | null>(null)
  const [gpuData, setGpuData] = useState<GPUStatus[]>([])
  const [connected, setConnected] = useState(false)
  const marketWs = useRef<WebSocket | null>(null)
  const gpuWs = useRef<WebSocket | null>(null)

  const connect = useCallback(() => {
    const wsUrl = (import.meta.env.VITE_WS_URL as string) || 'ws://localhost:8000/ws'

    // Market WebSocket
    const mws = new WebSocket(`${wsUrl}/market`)
    mws.onopen = () => setConnected(true)
    mws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data) as MarketSnapshot
        setMarketData(data)
      } catch { /* ignore non-JSON */ }
    }
    mws.onclose = () => setConnected(false)
    marketWs.current = mws

    // GPU WebSocket
    const gws = new WebSocket(`${wsUrl}/gpu`)
    gws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data) as GPUStatus[]
        setGpuData(data)
      } catch { /* ignore */ }
    }
    gpuWs.current = gws
  }, [])

  useEffect(() => {
    connect()
    return () => {
      marketWs.current?.close()
      gpuWs.current?.close()
    }
  }, [connect])

  return { marketData, gpuData, connected }
}

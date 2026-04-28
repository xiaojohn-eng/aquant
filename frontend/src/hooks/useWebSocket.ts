import { useEffect, useRef, useState, useCallback } from 'react';
import { WebSocketMessage } from '../types';

interface UseWebSocketOptions {
  endpoint: string;
  onMessage?: (msg: WebSocketMessage) => void;
  reconnectInterval?: number;
  heartbeatInterval?: number;
  autoConnect?: boolean;
}

interface WebSocketState {
  connected: boolean;
  lastMessage: WebSocketMessage | null;
  error: string | null;
  reconnectCount: number;
}

export function useWebSocket(options: UseWebSocketOptions): {
  state: WebSocketState;
  send: (data: unknown) => void;
  connect: () => void;
  disconnect: () => void;
} {
  const {
    endpoint,
    onMessage,
    reconnectInterval = 5000,
    heartbeatInterval = 30000,
    autoConnect = true,
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const heartbeatTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectCountRef = useRef(0);
  const shouldReconnectRef = useRef(true);

  const [state, setState] = useState<WebSocketState>({
    connected: false,
    lastMessage: null,
    error: null,
    reconnectCount: 0,
  });

  const clearTimers = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (heartbeatTimerRef.current) {
      clearInterval(heartbeatTimerRef.current);
      heartbeatTimerRef.current = null;
    }
  }, []);

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;
    clearTimers();
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setState((prev) => ({ ...prev, connected: false }));
  }, [clearTimers]);

  const connect = useCallback(() => {
    clearTimers();
    shouldReconnectRef.current = true;

    const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000/ws';
    const url = `${WS_BASE_URL}/${endpoint}`;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        reconnectCountRef.current = 0;
        setState({
          connected: true,
          lastMessage: null,
          error: null,
          reconnectCount: 0,
        });

        heartbeatTimerRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
          }
        }, heartbeatInterval);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data) as WebSocketMessage;
          setState((prev) => ({ ...prev, lastMessage: msg }));
          onMessage?.(msg);
        } catch {
          console.warn('WebSocket received non-JSON message:', event.data);
        }
      };

      ws.onerror = () => {
        setState((prev) => ({
          ...prev,
          error: 'WebSocket连接错误',
        }));
      };

      ws.onclose = () => {
        setState((prev) => ({ ...prev, connected: false }));
        clearTimers();

        if (shouldReconnectRef.current) {
          reconnectCountRef.current += 1;
          reconnectTimerRef.current = setTimeout(() => {
            setState((prev) => ({
              ...prev,
              reconnectCount: reconnectCountRef.current,
            }));
            connect();
          }, reconnectInterval);
        }
      };
    } catch (err) {
      setState((prev) => ({
        ...prev,
        error: err instanceof Error ? err.message : '连接失败',
      }));
    }
  }, [endpoint, onMessage, reconnectInterval, heartbeatInterval, clearTimers]);

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  useEffect(() => {
    if (autoConnect) {
      connect();
    }
    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return { state, send, connect, disconnect };
}

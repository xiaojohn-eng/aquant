import { useEffect, useRef, useState, useCallback } from 'react';

function isMarketOpen(): boolean {
  const now = new Date();
  const hours = now.getHours();
  const minutes = now.getMinutes();
  const day = now.getDay();

  // 周一至周五
  if (day === 0 || day === 6) return false;

  // 上午 9:30-11:30, 下午 13:00-15:00
  const time = hours * 60 + minutes;
  const morningOpen = 9 * 60 + 30;
  const morningClose = 11 * 60 + 30;
  const afternoonOpen = 13 * 60;
  const afternoonClose = 15 * 60;

  return (
    (time >= morningOpen && time <= morningClose) ||
    (time >= afternoonOpen && time <= afternoonClose)
  );
}

interface UseAutoRefreshOptions<T> {
  fetchFn: () => Promise<T>;
  marketOpenInterval?: number;
  marketCloseInterval?: number;
  enabled?: boolean;
  onError?: (error: Error) => void;
}

export function useAutoRefresh<T>(options: UseAutoRefreshOptions<T>): {
  data: T | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
  lastUpdated: Date | null;
} {
  const {
    fetchFn,
    marketOpenInterval = 30000,
    marketCloseInterval = 300000,
    enabled = true,
    onError,
  } = options;

  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const refresh = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
    }
    abortRef.current = new AbortController();

    setLoading(true);
    setError(null);

    fetchFn()
      .then((result) => {
        setData(result);
        setLastUpdated(new Date());
        setError(null);
      })
      .catch((err) => {
        if (err.name === 'AbortError') return;
        const msg = err instanceof Error ? err.message : '刷新失败';
        setError(msg);
        onError?.(err);
      })
      .finally(() => {
        setLoading(false);
      });
  }, [fetchFn, onError]);

  useEffect(() => {
    if (!enabled) return;

    refresh();

    const scheduleNext = () => {
      const interval = isMarketOpen() ? marketOpenInterval : marketCloseInterval;
      timerRef.current = setTimeout(() => {
        refresh();
        scheduleNext();
      }, interval);
    };

    scheduleNext();

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
      if (abortRef.current) {
        abortRef.current.abort();
      }
    };
  }, [enabled, refresh, marketOpenInterval, marketCloseInterval]);

  return { data, loading, error, refresh, lastUpdated };
}

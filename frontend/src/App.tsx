import React, { useState, useCallback } from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import RecommendationsPage from './pages/RecommendationsPage';
import PortfolioPage from './pages/PortfolioPage';
import PerformancePage from './pages/PerformancePage';
import GpuPage from './pages/GpuPage';
import BacktestView from './components/BacktestView';
import { useWebSocket } from './hooks/useWebSocket';
import { WebSocketMessage } from './types';

export interface AppContextType {
  wsConnected: boolean;
  lastMessage: WebSocketMessage | null;
  isDark: boolean;
  setIsDark: (v: boolean) => void;
}

export const AppContext = React.createContext<AppContextType>({
  wsConnected: false,
  lastMessage: null,
  isDark: true,
  setIsDark: () => {},
});

const App: React.FC = () => {
  const [isDark, setIsDark] = useState(true);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

  const handleWsMessage = useCallback((msg: WebSocketMessage) => {
    setLastMessage(msg);
  }, []);

  const { state: wsState } = useWebSocket({
    endpoint: 'main',
    onMessage: handleWsMessage,
    autoConnect: true,
  });

  return (
    <AppContext.Provider
      value={{
        wsConnected: wsState.connected,
        lastMessage,
        isDark,
        setIsDark,
      }}
    >
      <Layout>
        <Routes>
          <Route path="/" element={<RecommendationsPage />} />
          <Route path="/portfolio" element={<PortfolioPage />} />
          <Route path="/performance" element={<PerformancePage />} />
          <Route path="/gpu" element={<GpuPage />} />
          <Route path="/backtest" element={<BacktestView />} />
        </Routes>
      </Layout>
    </AppContext.Provider>
  );
};

export default App;

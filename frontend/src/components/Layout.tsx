import React, { useContext, useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  TrendingUp,
  Wallet,
  BarChart3,
  Cpu,
  History,
  Circle,
  Clock,
  Wifi,
  WifiOff,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { AppContext } from '../App';

interface LayoutProps {
  children: React.ReactNode;
}

const navItems = [
  { path: '/', label: '今日推荐', icon: TrendingUp },
  { path: '/portfolio', label: '持仓监控', icon: Wallet },
  { path: '/performance', label: '业绩分析', icon: BarChart3 },
  { path: '/gpu', label: 'GPU监控', icon: Cpu },
  { path: '/backtest', label: '回测结果', icon: History },
];

function formatTime(date: Date): string {
  return date.toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function isMarketOpenNow(): boolean {
  const now = new Date();
  const hours = now.getHours();
  const minutes = now.getMinutes();
  const day = now.getDay();
  if (day === 0 || day === 6) return false;
  const time = hours * 60 + minutes;
  return (
    (time >= 570 && time <= 690) || // 9:30-11:30
    (time >= 780 && time <= 900) // 13:00-15:00
  );
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const { wsConnected } = useContext(AppContext);
  const location = useLocation();
  const [currentTime, setCurrentTime] = useState(new Date());
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const marketOpen = isMarketOpenNow();
  const currentPage = navItems.find((item) => item.path === location.pathname)?.label || 'AQuant';

  return (
    <div className="flex h-screen w-screen bg-[#1a1a2e] overflow-hidden">
      {/* Sidebar */}
      <aside
        className={`flex flex-col bg-[#16213e] border-r border-[#2a3a5c] transition-all duration-300 ${
          sidebarCollapsed ? 'w-16' : 'w-60'
        }`}
      >
        {/* Logo */}
        <div className="flex items-center h-16 px-4 border-b border-[#2a3a5c]">
          <div className="w-8 h-8 rounded-lg bg-[#d4a574] flex items-center justify-center flex-shrink-0">
            <TrendingUp className="w-5 h-5 text-[#1a1a2e]" />
          </div>
          {!sidebarCollapsed && (
            <span className="ml-3 text-lg font-semibold text-[#d4a574] tracking-wide">
              AQuant
            </span>
          )}
        </div>

        {/* Nav Links */}
        <nav className="flex-1 py-4 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center px-4 py-3 mx-2 rounded-lg transition-all duration-200 group ${
                    isActive
                      ? 'bg-[#d4a574]/15 text-[#d4a574]'
                      : 'text-[#95a5a6] hover:bg-[#2a3a5c]/50 hover:text-[#e6ddd0]'
                  }`
                }
              >
                <Icon className={`w-5 h-5 flex-shrink-0 ${isActive ? 'text-[#d4a574]' : ''}`} />
                {!sidebarCollapsed && (
                  <span className="ml-3 text-sm font-medium">{item.label}</span>
                )}
                {isActive && !sidebarCollapsed && (
                  <div className="ml-auto w-1.5 h-1.5 rounded-full bg-[#d4a574]" />
                )}
              </NavLink>
            );
          })}
        </nav>

        {/* Collapse Toggle */}
        <div className="p-3 border-t border-[#2a3a5c]">
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="flex items-center justify-center w-full py-2 rounded-lg text-[#95a5a6] hover:bg-[#2a3a5c]/50 hover:text-[#e6ddd0] transition-colors"
          >
            {sidebarCollapsed ? (
              <ChevronRight className="w-5 h-5" />
            ) : (
              <>
                <ChevronLeft className="w-4 h-4 mr-2" />
                <span className="text-xs">收起侧边栏</span>
              </>
            )}
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Bar */}
        <header className="h-16 bg-[#16213e]/80 backdrop-blur-sm border-b border-[#2a3a5c] flex items-center justify-between px-6">
          <div className="flex items-center">
            <h1 className="text-lg font-medium text-[#e6ddd0]">{currentPage}</h1>
          </div>

          <div className="flex items-center gap-6">
            {/* Trading Status */}
            <div className="flex items-center gap-2">
              <Circle
                className={`w-3 h-3 ${
                  marketOpen ? 'text-[#e74c3c] fill-[#e74c3c]' : 'text-[#95a5a6] fill-[#95a5a6]'
                }`}
              />
              <span className="text-sm text-[#95a5a6]">
                {marketOpen ? '交易中' : '休市中'}
              </span>
            </div>

            {/* WS Status */}
            <div className="flex items-center gap-2" title={wsConnected ? '实时连接正常' : '实时连接断开'}>
              {wsConnected ? (
                <Wifi className="w-4 h-4 text-[#2ecc71]" />
              ) : (
                <WifiOff className="w-4 h-4 text-[#e74c3c]" />
              )}
            </div>

            {/* Clock */}
            <div className="flex items-center gap-2 text-[#95a5a6]">
              <Clock className="w-4 h-4" />
              <span className="text-sm font-mono">{formatTime(currentTime)}</span>
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-auto p-6">
          <div className="animate-fade-in">{children}</div>
        </main>
      </div>
    </div>
  );
};

export default Layout;

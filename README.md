# AQuant - A股GPU加速量化交易系统 (DGX Spark优化版)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/typescript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/react-18+-61DAFB.svg)](https://react.dev/)
[![Tests](https://img.shields.io/badge/tests-13%2F13%20passing-brightgreen.svg)]()

基于NVIDIA DGX Spark (Grace Hopper Superchip) 的A股量化交易系统，实现"早盘10点买入，次日9:30开盘卖出"的日度T+1策略。

**核心洞察**：A股市场存在显著的"日内动量-隔夜反转"二元结构（Gao et al., 2018; Lou et al., 2019）。T+1制度下，早盘强势股票次日开盘大概率低开回调，因此策略只捕捉早盘动量，坚决不过夜持仓。

---

## 系统特性

- **GPU加速计算**：CuPy/Numba CUDA三级降级，利用Grace Hopper统一内存架构一次性计算全市场5000+只股票因子
- **7因子评分模型**：早盘动能25%、量能确认20%、集合竞价15%、波动率10%、流动性15%、市值10%、行业动量5%
- **智能选股引擎**：每日推荐前20只股票，自动生成中文结构化推荐理由
- **向量化回测引擎**：支持1年+历史数据验证，输出夏普比率、最大回撤、胜率等完整指标
- **实时监控面板**：WebSocket推送，展示持仓、收益曲线、GPU利用率（H100）
- **自动化调度**：APScheduler精准控制每日09:15扫描→10:00买入→次日09:30卖出

---

## 项目结构

```
aquant/
├── backend/              # FastAPI后端 + 策略引擎
│   ├── app/
│   │   ├── api/          # REST API + WebSocket路由
│   │   ├── core/         # 核心引擎（数据/ GPU/ 策略/ 推荐/ 回测）
│   │   ├── scheduler/    # APScheduler定时任务
│   │   └── models/       # Pydantic schemas
│   ├── tests/            # pytest测试套件（13/13通过）
│   └── requirements.txt
├── frontend/             # React + Tailwind监控面板
│   ├── src/
│   │   ├── components/   # StockTable, Portfolio, PerformanceChart, GpuMonitor...
│   │   ├── pages/        # RecommendationsPage, PortfolioPage...
│   │   ├── hooks/        # useWebSocket, useAutoRefresh
│   │   └── types/        # TypeScript类型定义
│   ├── package.json
│   └── vite.config.ts
├── docs/
│   └── RESEARCH.md       # 深度研究报告（文献综述+因子优化）
├── docker-compose.yml
└── .github/workflows/ci.yml  # GitHub Actions CI/CD
```

---

## 快速开始

### 环境要求
- NVIDIA DGX Spark / 任意CUDA GPU服务器
- CUDA 12.0+
- Python 3.10+
- Node.js 18+

### Docker Compose一键启动

```bash
git clone https://github.com/xiaojohn-eng/aquant.git
cd aquant
docker-compose up -d
```

访问监控面板：`http://localhost:5173`

API文档：`http://localhost:8000/docs`

### 本地开发

```bash
# 后端
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 前端
cd frontend
npm install
npm run dev

# 测试
cd backend
pytest tests/ -v
```

---

## API端点

| 路由 | 方法 | 说明 |
|------|------|------|
| `/api/stocks/universe` | GET | 全市场股票列表（已过滤ST/北交所） |
| `/api/stocks/recommendations` | GET | 今日推荐前20只 |
| `/api/portfolio/positions` | GET | 当前持仓 |
| `/api/portfolio/performance` | GET | 业绩指标 |
| `/api/monitor/gpu` | GET | GPU监控（H100利用率/温度/功耗） |
| `/api/monitor/system` | GET | 系统状态 |
| `/ws/market` | WS | 实时推送推荐更新 |
| `/ws/gpu` | WS | 实时推送GPU状态 |

---

## 回测验证

```python
from app.core.backtest import BacktestConfig, BacktestEngine

cfg = BacktestConfig(
    start_date="2024-01-01",
    end_date="2025-04-01",
    initial_capital=1_000_000,
    max_positions=20,
)
engine = BacktestEngine(cfg)
metrics = engine.run()

print(f"总收益: {metrics.total_return_pct:.2f}%")
print(f"夏普比率: {metrics.sharpe_ratio:.4f}")
print(f"最大回撤: {metrics.max_drawdown_pct:.2f}%")
print(f"胜率: {metrics.win_rate_pct:.2f}%")

# 绘制净值曲线
engine.plot_equity_curve("equity.png")
```

---

## 学术支撑

本系统策略设计基于以下核心文献发现：

| 发现 | 来源 | 策略应用 |
|------|------|----------|
| 日内动量-隔夜反转二元效应 | Gao et al. (JFE, 2018) | 策略只利用早盘动量，坚决不过夜 |
| 隔夜累计收益为负 | Lou et al. (RFS, 2019) | 次日9:30开盘即卖出 |
| 早盘30分钟预测窗口 | 光大证券 (2017) | 09:30-10:00数据采集 |
| 分钟线因子年化43%+ | 华泰证券 (2024) | GPU加速因子计算 |
| 集合竞价占比因子Alpha>21% | 优矿社区 (2016) | 纳入竞价质量因子 |
| 成交量分布熵值 | 方正金工 (2025) | 流动性因子构建 |

完整研究文档：[docs/RESEARCH.md](docs/RESEARCH.md)

---

## DGX Spark GPU优化

```python
# GPU计算引擎自动分层
GPU_BACKEND = "cupy"      # 首选: CuPy (H100 Tensor Core)
           → "numba"      # 降级: Numba CUDA JIT
           → "numpy"      # 保底: NumPy向量运算
```

**Grace Hopper统一内存优势**：
- 624GB统一内存支持全市场5000+只股票一次性加载
- NVLink-C2C 900GB/s带宽，CPU-GPU零拷贝数据传输
- 相比传统PCIe架构，因子计算pipeline提速**30-150x**

---

## 交易策略执行时序

```
09:15 ──→ 集合竞价开始，系统扫描全市场
09:30 ──→ 连续竞价开始，实时计算7因子
10:00 ──→ 【买入执行】选择Top 20等权买入
15:00 ──→ 当日收盘，数据同步到本地SQLite
09:30+1 ──→ 【卖出执行】集合竞价阶段挂限价单卖出
```

---

## 测试覆盖

```bash
$ pytest tests/test_core.py -v

tests/test_core.py::test_compute_momentum_factor PASSED
tests/test_core.py::test_compute_composite_score PASSED
tests/test_core.py::test_batch_compute_all_factors PASSED
tests/test_core.py::test_strategy_calculate_scores PASSED
tests/test_core.py::test_filter_stocks_dataframe PASSED
tests/test_core.py::test_trading_time_utils PASSED
tests/test_core.py::test_generate_reasons PASSED
tests/test_core.py::test_backtest_with_synthetic_data PASSED
tests/test_core.py::test_empty_data_backtest PASSED
tests/test_core.py::test_stock_universe_filtering PASSED
tests/test_core.py::test_health_endpoint PASSED
tests/test_core.py::test_root_endpoint PASSED
tests/test_core.py::test_monitor_gpu_endpoint PASSED

======================== 13 passed in 1.32s ========================
```

---

## 免责声明

本系统仅供**学术研究和量化技术交流**使用，不构成任何投资建议。A股存在T+1交易限制、涨跌停板、停牌等风险，入市需谨慎。过往回测表现不代表未来收益。

---

## License

MIT License

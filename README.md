# AQuant - A股GPU加速量化交易系统 (DGX Spark优化版)

基于NVIDIA DGX Spark (Grace Hopper) 的A股量化交易系统，实现"早盘10点买入，次日9:30开盘卖出"的日度T+1策略。

## 系统特性

- **GPU加速计算**：利用NVIDIA Grace Hopper的统一内存架构，通过CuPy/Numba CUDA加速全市场5000+只股票的因子计算
- **智能选股引擎**：多因子评分 + 机器学习排序，每日推荐前20只股票
- **A股微观结构适配**：针对T+1交易制度和"日内动量-隔夜反转"效应专门优化
- **实时监控面板**：WebSocket推送，实时展示持仓、收益、GPU利用率
- **自动化交易调度**：APScheduler精准控制每日10:00买入、次日9:30卖出

## 项目结构

```
aquant/
├── backend/          # FastAPI后端 + 策略引擎
│   ├── app/
│   │   ├── api/      # REST API路由
│   │   ├── core/     # 核心引擎（数据/GPU/策略/推荐）
│   │   ├── scheduler/# 定时任务
│   │   └── models/   # Pydantic模型
│   └── requirements.txt
├── frontend/         # React + Tailwind监控面板
│   └── src/
├── tests/            # 测试用例
└── docker-compose.yml
```

## 核心策略逻辑

### A股微观结构洞察

基于学术研究（Gao et al., 2018; Lou et al., 2019）发现：
1. **日内动量效应**：开盘前半小时收益可预测收盘前半小时收益
2. **隔夜反转诅咒**：T+1制度下，日内强势股次日开盘大概率低开/回调
3. **10点入场窗口**：9:30-10:00的集合竞价到连续竞价过渡区，信息充分且趋势初显

### 策略执行流程

```
09:15-09:25  集合竞价数据采集
09:30-10:00  开盘后30分钟实时计算全市场因子
10:00:00     执行买入信号——选择排名前20只股票等权买入
09:30+1d    次日开盘执行卖出（集合竞价阶段挂限价单）
```

### 多因子评分模型

| 因子类别 | 具体因子 | 权重 | 说明 |
|---------|---------|------|------|
| 早盘动能 | 开盘后30分钟涨跌幅 | 25% | 核心动量信号 |
| 量能确认 | 开盘后30分钟成交量/20日均量 | 20% | 放量上涨更可靠 |
| 集合竞价 | 开盘价/集合竞价加权价 | 15% | 竞价阶段资金意图 |
| 波动率 | ATR(14) / 价格 | 10% | 控制隔夜风险 |
| 流动性 | 30分钟成交额排名 | 15% | 确保可成交 |
| 市值偏置 | -log(流通市值) | 10% | 小市值弹性 |
| 行业动量 | 所属行业早盘涨幅排名 | 5% | 板块效应 |

### 推荐理由生成

每只推荐股票自动生成结构化理由：
```
- 早盘动能：开盘后涨幅X%，位列市场前Y%
- 量能配合：成交量为20日均量的X倍
- 竞价质量：开盘价高于竞价加权价X%
- 风险指标：ATR波动率X%，流动性良好
- 综合评分：X分（击败Z%的股票）
```

## 技术栈

| 层级 | 技术 |
|-----|------|
| 数据获取 | AkShare（免费A股分钟级数据） |
| GPU计算 | CuPy + Numba CUDA |
| 后端 | FastAPI + APScheduler + WebSocket |
| 前端 | React + Tailwind CSS + shadcn/ui |
| 监控 | nvidia-ml-py + Grafana风格面板 |
| 部署 | Docker + Docker Compose |

## 快速开始

### 环境要求
- NVIDIA DGX Spark / 任意CUDA GPU服务器
- CUDA 12.0+
- Python 3.10+
- Node.js 18+

### 安装
```bash
# 后端
cd backend
pip install -r requirements.txt

# 前端
cd frontend
npm install
```

### 运行
```bash
# 启动后端
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

# 启动前端（开发模式）
cd frontend && npm run dev

# 或使用Docker Compose一键启动
docker-compose up -d
```

### 访问监控面板
浏览器打开 `http://localhost:5173`

## API文档

启动后访问 `http://localhost:8000/docs` 查看Swagger文档。

## 回测验证

系统内置回测引擎，支持至少1年历史数据验证：
```bash
cd backend
python -m app.core.backtest --start 2024-01-01 --end 2025-04-01
```

## 免责声明

本系统仅供学术研究和量化技术交流使用，不构成任何投资建议。股市有风险，投资需谨慎。

## License

MIT License

# Zhipu MaaS API Test Framework

智谱 MaaS 平台 API 自动化测试框架，覆盖 Chat Completion、Streaming（SSE）、Embedding、鉴权等核心场景。

## 项目结构

```
zhipu-api-test/
├── config/                    # 多环境配置
│   ├── base.yaml              # 基础配置（URL、超时、阈值）
│   └── staging.yaml           # 预发环境覆盖配置
├── core/                      # 核心能力封装层
│   ├── client.py              # API Client（鉴权、重试、日志、计时）
│   ├── assertions.py          # 自定义断言（响应校验、性能校验、SSE校验）
│   ├── data_factory.py        # 测试数据工厂（Prompt生成、参数组合）
│   └── sse_parser.py          # SSE流式响应解析器（TTFT、完整性校验）
├── tests/                     # 测试用例
│   ├── test_chat.py           # Chat Completion 测试（基础/多轮/参数/边界/性能）
│   ├── test_stream.py         # 流式输出测试（完整性/TTFT/内容一致性）
│   ├── test_auth.py           # 鉴权测试（有效/无效/过期/安全边界）
│   └── test_models.py         # 模型相关测试（列表/校验/Embedding）
├── utils/                     # 工具类
│   ├── config_loader.py       # 配置加载器（多环境、环境变量）
│   └── report_helper.py       # 指标收集与报告生成
├── conftest.py                # Pytest Fixtures & Hooks
├── pytest.ini                 # Pytest 配置（Markers、默认参数）
├── requirements.txt           # Python 依赖
└── README.md
```

## 设计理念

### 分层架构

```
┌─────────────────────────────────────┐
│      CI/CD 触发层                     │  Jenkins / GitLab CI / GitHub Actions
├─────────────────────────────────────┤
│      测试管理层                       │  pytest markers, Allure报告, 通知
├─────────────────────────────────────┤
│      测试用例层                       │  tests/ — 只关注业务逻辑
├─────────────────────────────────────┤
│      能力封装层（核心）                 │  core/ — Client、断言、数据工厂
├─────────────────────────────────────┤
│      基础设施层                       │  config/ + utils/ — 配置、日志、指标
└─────────────────────────────────────┘
```

### 核心设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 语言 | Python | AI 生态主流语言，团队协作成本低 |
| 框架 | Pytest | 简洁、生态丰富、fixture 机制强大 |
| HTTP | Requests | 稳定可靠，支持 Session/重试/流式 |
| 报告 | Allure | 可视化好，支持截图/日志/步骤 |
| 配置 | YAML + 环境变量 | 多环境支持，密钥不入代码 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# Linux / macOS
export ZHIPU_API_KEY="your-api-key-here"

# Windows
set ZHIPU_API_KEY=your-api-key-here
```

> API Key 从 [智谱开放平台](https://open.bigmodel.cn/) 获取

### 3. 运行测试

```bash
# 运行冒烟测试（快速验证）
pytest -m smoke

# 运行全量测试
pytest

# 运行性能测试
pytest -m performance

# 指定环境
pytest --env=staging

# 生成 Allure 报告
pytest --alluredir=./allure-results
allure serve ./allure-results

# 并行执行（加速）
pytest -n 4

# 失败自动重试
pytest --reruns 2 --reruns-delay 3
```

## 测试覆盖

| 模块 | 场景 | 用例数 |
|------|------|--------|
| **Chat Completion** | 基础对话、多轮对话、参数校验、边界测试、性能 | ~20 |
| **Streaming (SSE)** | 完整性、TTFT、内容一致性、参数组合 | ~12 |
| **Authentication** | 有效/无效/过期密钥、安全注入、流式鉴权 | ~10 |
| **Models & Embedding** | 模型列表、无效模型、Embedding 基础 | ~10 |

### Marker 说明

| Marker | 用途 | 运行方式 |
|--------|------|----------|
| `@pytest.mark.smoke` | 冒烟测试，每次提交必跑 | `pytest -m smoke` |
| `@pytest.mark.performance` | 性能测试，关注延迟 | `pytest -m performance` |
| `@pytest.mark.regression` | 全量回归 | `pytest -m regression` |

## CI/CD 集成示例

### GitHub Actions

```yaml
name: API Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest -m smoke
        env:
          ZHIPU_API_KEY: ${{ secrets.ZHIPU_API_KEY }}
```

## 扩展方向

- [ ] 全链路压测场景（Locust 集成）
- [ ] 故障注入测试（超时、断连模拟）
- [ ] Token 配额和限流测试
- [ ] 多模型对比评测
- [ ] 性能基线自动对比（本次 vs 历史）

## License

MIT

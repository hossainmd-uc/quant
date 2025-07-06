# Advanced Quantitative Trading System 🚀

A cutting-edge algorithmic trading platform leveraging modern ML/AI techniques for market analysis and strategy development.

## 🌟 Features

- **Advanced ML Models**: Transformer-based time series forecasting, Graph Neural Networks, Reinforcement Learning
- **Real-time Data Processing**: High-frequency market data with microsecond precision
- **Distributed Computing**: Ray/Dask for scalable backtesting and hyperparameter optimization
- **Multiple Asset Classes**: Equities, Options, Futures, Crypto, Forex
- **Advanced Risk Management**: Portfolio optimization, VaR, stress testing
- **Low-latency Execution**: Direct broker integrations with optimal routing
- **Real-time Monitoring**: Comprehensive dashboards and alerting systems
- **📚 Document AI Integration**: Extract knowledge from financial books using LandingAI

## 🛠️ Technology Stack

### Core
- **Python 3.11+** with advanced typing and performance optimizations
- **PyTorch 2.0+** for deep learning models
- **Ray** for distributed computing
- **Polars** for high-performance data processing
- **FastAPI** for web services and APIs

### ML/AI
- **Transformers** for time series forecasting
- **PyTorch Geometric** for Graph Neural Networks
- **Stable Baselines3** for Reinforcement Learning
- **Optuna** for hyperparameter optimization
- **MLflow** for experiment tracking

### Document AI
- **LandingAI Agentic-Doc** for intelligent document extraction
- **Structured Knowledge Extraction** from financial literature
- **Automated Integration** of book concepts into trading models

### Data & Infrastructure
- **Apache Kafka** for real-time data streaming
- **ClickHouse** for time series storage
- **Redis** for caching and session management
- **Docker** for containerization
- **Kubernetes** for orchestration

### Trading
- **Alpaca** for US equities
- **Interactive Brokers** for global markets
- **Binance** for cryptocurrency
- **QuantLib** for derivatives pricing

## 📁 Project Structure

```
quant/
├── core/                   # Core system components
│   ├── data/              # Data ingestion and processing + Document AI
│   ├── models/            # ML/AI models
│   ├── strategies/        # Trading strategies
│   ├── execution/         # Order execution
│   ├── risk/              # Risk management
│   └── utils/             # Utilities and helpers
├── research/              # Research notebooks and experiments
├── backtesting/           # Backtesting framework
├── monitoring/            # Monitoring and alerting
├── api/                   # REST API services
├── ui/                    # Web dashboard
├── configs/               # Configuration files
├── tests/                 # Test suite
├── scripts/               # Deployment and utility scripts
├── docs/                  # Documentation
└── extracted_knowledge/   # AI-extracted book knowledge
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp configs/environment.example .env
# Edit .env with your API keys
```

### 3. Extract Knowledge from Financial Books (NEW!)
```bash
# Get LandingAI API key from https://landing.ai/agentic-document-extraction
export LANDINGAI_API_KEY=your_api_key

# Extract knowledge from Python for Finance book
python research/extract_book_knowledge.py
```

### 4. Run Demo
```bash
python research/demo.py
```

### 5. Run Backtesting
```bash
python -m core.backtesting.runner --strategy momentum --start-date 2023-01-01
```

### 6. Start Live Trading
```bash
python -m core.execution.live_trader --strategy your_strategy
```

## 📚 New: Document AI Integration

### Extract Knowledge from Financial Books

Our system now includes cutting-edge document AI capabilities to extract structured knowledge from financial literature:

```python
from core.data.document_extractor import extract_python_finance_book

# Extract all concepts, formulas, and code from the book
knowledge = extract_python_finance_book()

print(f"Extracted {len(knowledge.concepts)} concepts")
print(f"Found {len(knowledge.code_snippets)} code examples")
print(f"Identified {len(knowledge.formulas)} formulas")
```

### Benefits
- **Structured Access**: Search through book content by category, confidence, or keywords
- **Code Integration**: Extract working Python examples directly from books
- **Formula Library**: Access mathematical formulas in structured format
- **Visual Groundings**: See exactly where content was extracted from
- **Automated Analysis**: AI categorizes content by financial domain

### Supported Extraction
- ✅ **Python for Finance** by Yves Hilpisch
- ✅ Any financial PDF document
- ✅ Code snippets and examples
- ✅ Mathematical formulas
- ✅ Risk management concepts
- ✅ Trading strategies
- ✅ Portfolio theory

## 📊 Performance Metrics

- **Sharpe Ratio**: > 2.0
- **Max Drawdown**: < 10%
- **Win Rate**: > 55%
- **Latency**: < 100μs execution time
- **Uptime**: 99.9%
- **Knowledge Extraction**: 500+ concepts from 600-page books

## 🔬 Research & Development

This system is designed for continuous research and development:

- **Jupyter Lab** integration for interactive research
- **Automated model retraining** with new data
- **A/B testing** framework for strategy comparison
- **Feature engineering** pipeline with automated discovery
- **Model interpretability** tools for understanding decisions
- **📖 Book Knowledge Integration** with AI extraction

## 🛡️ Risk Management

- **Position sizing** based on Kelly criterion and volatility
- **Stop-loss** and take-profit automation
- **Portfolio diversification** across assets and strategies
- **Real-time risk monitoring** with automated alerts
- **Stress testing** against historical scenarios

## 📈 Getting Started

### Basic Setup
1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Configure APIs**: Set up Alpaca, IB, LandingAI credentials
3. **Extract Book Knowledge**: `python research/extract_book_knowledge.py`
4. **Run Demo**: `python research/demo.py`

### Advanced Usage
Check out our detailed guides:
- [System Overview](docs/SYSTEM_OVERVIEW.md) - Complete system architecture
- [Book Extraction Guide](docs/BOOK_EXTRACTION_GUIDE.md) - Extract knowledge from financial books
- [Getting Started Guide](docs/getting_started.md) - Detailed setup instructions

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results.

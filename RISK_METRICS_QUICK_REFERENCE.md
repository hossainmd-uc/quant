# 📋 Risk Metrics Quick Reference

*Handy cheat sheet for understanding and interpreting financial risk metrics*

---

## 🏆 **PERFORMANCE METRICS**

| Metric | Formula | Good Value | Interpretation |
|--------|---------|------------|----------------|
| **Sharpe Ratio** | (Return - Risk-Free) / Volatility | > 1.0 | Risk-adjusted performance |
| **Sortino Ratio** | (Return - Risk-Free) / Downside Vol | > 1.0 | Downside risk-adjusted performance |
| **Information Ratio** | (Return - Benchmark) / Tracking Error | > 0.5 | Consistency of outperformance |
| **Calmar Ratio** | Annual Return / Max Drawdown | > 0.5 | Return per unit of worst risk |

---

## 📉 **RISK METRICS**

| Metric | What It Measures | Critical Levels | Action Required |
|--------|------------------|-----------------|-----------------|
| **Maximum Drawdown** | Worst peak-to-trough loss | > 30% | 🚨 High stress risk |
| **VaR 95%** | Max loss (95% confidence) | > 5% daily | 🚨 Reduce position size |
| **CVaR 95%** | Average loss in worst 5% | > 1.5x VaR | 🚨 Tail risk present |
| **Volatility** | Price fluctuation | > 40% annual | 🚨 High uncertainty |

---

## 🎯 **VaR METHODS COMPARISON**

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Historical** | Stable markets | Real market behavior | Assumes future = past |
| **Parametric** | Quick estimates | Fast calculation | Assumes normal distribution |
| **Monte Carlo** | Complex portfolios | Flexible modeling | Computationally intensive |
| **Cornish-Fisher** | Skewed returns | Accounts for fat tails | Still parametric |

---

## 🔧 **PRACTICAL DECISION RULES**

### **Position Sizing**
```
If Daily VaR = $1,000
And you can risk $2,000/day
→ Double your position size
```

### **Strategy Selection**
```
Strategy A: Sharpe 1.2, Max DD 15%
Strategy B: Sharpe 0.8, Max DD 8%
→ Choose A if risk-tolerant
→ Choose B if risk-averse
```

### **Stop-Loss Setting**
```
Set stops at 1.5-2x Daily VaR
If VaR = $1,000 → Stop at $1,500-$2,000 loss
```

### **Portfolio Allocation**
```
Max Sharpe → Best risk-adjusted returns
Min Volatility → Capital preservation
Target Return → Specific return goal
```

---

## 📊 **INTERPRETATION GUIDE**

### **Sharpe Ratio Scale**
- **> 2.0**: 🔥 Outstanding (rare)
- **1.0-2.0**: 🌟 Excellent
- **0.5-1.0**: ✅ Good
- **0-0.5**: ⚠️ Poor
- **< 0**: ❌ Losing money

### **Maximum Drawdown Scale**
- **< 10%**: 🟢 Low risk
- **10-20%**: 🟡 Moderate risk
- **20-40%**: 🟠 High risk
- **> 40%**: 🔴 Extreme risk

### **VaR Confidence Levels**
- **95%**: Standard risk management
- **99%**: Conservative/regulatory
- **90%**: Aggressive trading

---

## 🎲 **MONTE CARLO INSIGHTS**

### **Simulation Parameters**
- **1,000 runs**: Quick analysis
- **10,000 runs**: Standard analysis
- **100,000 runs**: High precision

### **Key Outputs**
- **Mean**: Expected outcome
- **VaR**: Worst-case scenario
- **CVaR**: Tail risk measure
- **Probabilities**: Outcome likelihoods

---

## ⚠️ **STRESS TESTING SCENARIOS**

### **Common Stress Tests**
- **Market Crash**: -30% to -50%
- **Volatility Spike**: 2x to 3x normal
- **Correlation Breakdown**: All correlations → 1
- **Liquidity Crisis**: Increased bid-ask spreads

### **Interpretation**
- **< 20% increase in VaR**: 🟢 Resilient
- **20-50% increase**: 🟡 Moderate stress
- **> 50% increase**: 🔴 High stress vulnerability

---

## 🚀 **QUICK WORKFLOW**

1. **Calculate Sharpe ratios** → Select best strategies
2. **Check max drawdowns** → Assess risk tolerance
3. **Optimize portfolio** → Allocate capital
4. **Calculate VaR** → Size positions
5. **Run stress tests** → Set risk limits
6. **Monitor & rebalance** → Maintain performance

---

## 💡 **COMMON PITFALLS**

### ❌ **What NOT to Do**
- Don't use only one metric
- Don't ignore correlations in crises
- Don't assume normal distributions
- Don't neglect transaction costs
- Don't over-optimize on historical data

### ✅ **Best Practices**
- Use multiple complementary metrics
- Regularly update risk models
- Account for regime changes
- Include transaction costs
- Validate out-of-sample

---

## 🎯 **METRIC COMBINATIONS**

### **For Conservative Investors**
- Focus on: Sortino Ratio, Max Drawdown, Min Vol optimization
- Monitor: VaR 99%, stress test results
- Targets: Sharpe > 0.7, Max DD < 15%

### **For Aggressive Investors**
- Focus on: Sharpe Ratio, Information Ratio, Max Return optimization
- Monitor: VaR 95%, Monte Carlo scenarios
- Targets: Sharpe > 1.0, Max DD < 30%

### **For Institutional Managers**
- Focus on: Information Ratio, Tracking Error, Risk-adjusted returns
- Monitor: Component VaR, stress tests, regulatory metrics
- Targets: Consistent outperformance, controlled tracking error

---

*Keep this reference handy when analyzing your trading strategies and making risk management decisions!* 
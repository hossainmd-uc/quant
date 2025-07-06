# 📚 Python for Finance - Complete Knowledge Catalog
================================================================================

**Total Chunks**: 7650
**Categories Found**: 9

## 📂 General Concepts
**Concepts**: 4196

**Key Topics:**
  • {'title': 'Figure 16-2. Average capital over time for different fractions', 'x_axis': {'label': 'Tim... [picture]
  • In [64]: np.arange('2020-01-01', '2020-01-04', dtype='datetime64[D]')
Out[64]: array(['2020-01-01', ... [code]
  • {'type': 'line_chart', 'x_axis': {'label': 'time', 'values': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ... [picture]
  • In [78]: dti = pd.date_range('2020/01/01', freq='M', periods=12)
    ...: dti
Out[78]: DatetimeIndex... [code]
  • {'type': 'line_chart', 'x_axis': {'label': 'time', 'values': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ... [picture]
  • except:
            pass
        paths = self.underlying.get_instrument_values(fixed_seed=fixed_seed... [code]
  • In [86]: pd.date_range('2020/01/01', freq='M', periods=12, 
    ...: tz=pytz.timezone('CET')) 
Out[8... [code]
  • {'type': 'line_chart', 'x_axis': {'label': 'time steps', 'values': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10... [picture]
  • [30]: news = ek.get_news_headlines('R:AAPL.O Language:LEN',
date_from='2018-05-01',
date_to='2018-06... [code]
  • {'type': 'line_chart', 'x_axis': {'label': 'time', 'values': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ... [picture]

**Content Types:**
  • text: 2758
  • section_header: 505
  • code: 373
  • list_item: 330
  • caption: 95
  • picture: 54
  • formula: 36
  • table: 28
  • footnote: 9
  • paragraph: 5
  • page_footer: 2
  • page_header: 1

**What You Can Ask:**
  ❓ What is general concepts?

------------------------------------------------------------

## 📂 Data Analysis
**Concepts**: 1136

**Key Topics:**
  • {
  "type": "line_chart",
  "title": null,
  "x_label": "x",
  "y_label": "f(x)",
  "legend": [
    ... [picture]
  • {'type': 'histogram', 'title': 'SPY', 'x_axis': {'label': 'Returns', 'range': [-0.06, 0.04]}, 'y_axi... [picture]
  • {'data_head': [{'Date': '2010-01-01', 'AAPL.O': None, 'MSFT.O': None, 'INTC.O': None, 'AMZN.O': None... [table]
  • {'type': 'histogram', 'x_axis': {'label': 'value', 'range': [0.005, 0.045]}, 'y_axis': {'label': 'fr... [picture]
  • {'type': 'histogram', 'x_axis': {'label': 'index level', 'range': [0, 450]}, 'y_axis': {'label': 'fr... [picture]
  • {'code_snippets': [{'line_number': 32, 'code': "candles = api.get_candles('USD/JPY', period='D1', nu... [picture]
  • In [36]: import fxcmpy

In [37]: fxcmpy.__version__
Out[37]: '1.1.33'

In [38]: api = fxcmpy.fxcmpy(... [code]
  • {'type': 'histogram', 'x_axis': {'label': 'index level', 'range': [0, 400]}, 'y_axis': {'label': 'fr... [picture]
  • {'type': 'line_chart', 'x_axis': {'label': 'Date', 'range': ['2010', '2018']}, 'y_axis': {'label': '... [picture]
  • {'title': 'A Simple Plot', 'x_label': 'index', 'y_label': 'value', 'lines': [{'label': '1st', 'color... [picture]

**Content Types:**
  • text: 669
  • code: 144
  • picture: 120
  • section_header: 78
  • list_item: 59
  • caption: 40
  • table: 19
  • footnote: 4
  • page_header: 1
  • page_footer: 1
  • formula: 1

**What You Can Ask:**
  ❓ How do I use pandas for finance?
  ❓ What are time series techniques?
  ❓ How do I visualize financial data?
  ❓ How do I clean market data?

------------------------------------------------------------

## 📂 Options Derivatives
**Concepts**: 877

**Key Topics:**
  • {
  "code": [
    {
      "line": 32,
      "operation": "comparison",
      "expression": "pred == ... [picture]
  • In [53]: val_env = dx.market_environment('val_env', pricing_date)
    ...: val_env.add_constant('sta... [code]
  • def report_positions(pos):
    ''' Prints, logs and sends position data.
    '''
    out = '\n\n' + ... [code]
  • {
  "code": [
    {
      "line": 26,
      "content": "symbol = '.SPX'"
    },
    {
      "line": ... [picture]
  • In [1]: import eikon as ek
   ...: import pandas as pd
   ...: import datetime as dt
   ...: import ... [code]
  • 'optimizer': None, 'activation_fn': <function relu at 0x1a3aa75b70>, 'dropout': None, 'gradient_clip... [code]
  • In [12]: limit = 500

In [13]: option_selection = calls[abs(calls['STRIKE_PRC'] - initial_value) < l... [code]
  • {'code': [{'line': 17, 'operation': 'data.pct_change().round(3).head()', 'output': {'AAPL.O': {'2010... [picture]
  • {'figure': {'type': 'code_output', 'content': [{'step': 1, 'description': "Weekly resampling of data... [picture]
  • {'type': 'code_output', 'description': 'The image shows Python code output for calculating and displ... [picture]

**Content Types:**
  • text: 617
  • code: 89
  • list_item: 67
  • picture: 36
  • section_header: 31
  • caption: 23
  • table: 10
  • paragraph: 2
  • footnote: 2

**What You Can Ask:**
  ❓ How does the Black-Scholes model work?
  ❓ What are the Greeks in options trading?
  ❓ How do I implement option pricing?
  ❓ What is implied volatility?

------------------------------------------------------------

## 📂 Risk Management
**Concepts**: 366

**Key Topics:**
  • {'data': [{'Function': 'exponential', 'Parameters': '[scale, size]', 'Returns/result': 'Samples from... [table]
  • ```python
def update(self, initial_value=None, volatility=None, lamb=None,
           mu=None, delta... [code]
  • def update(self, initial_value=None, volatility=None,
strike=None, maturity=None):
if initial_value ... [code]
  • ```python
try:
    # if there are special dates, then add these
    self.special_dates = mar_env.get... [code]
  • {'ranges': {'volatility': '(0.10, 0.201, 0.025)', 'jump_intensity': '(0.10, 0.80, 0.10)', 'average_j... [picture]
  • {'type': 'line_chart', 'x_axis': {'label': 'time', 'values': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ... [picture]
  • self.instrument_values = None

def generate_paths(self, fixed_seed=False, day_count=365):
    if sel... [code]
  • ```
#
# DX Package
#
# Simulation Class -- Base Class
#
# simulation_class.py
#
# Python for Finance... [code]
  • {'data': [{'Element': 'initial_value', 'Type': 'Constant', 'Mandatory': 'Yes', 'Description': 'Initi... [table]
  • def update(self, initial_value=None, volatility=None, kappa=None,
theta=None, final_date=None):
if i... [code]

**Content Types:**
  • text: 247
  • code: 40
  • picture: 28
  • list_item: 19
  • section_header: 15
  • caption: 7
  • table: 5
  • paragraph: 3
  • footnote: 2

**What You Can Ask:**
  ❓ How do I calculate Value-at-Risk?
  ❓ What is the Sharpe ratio?
  ❓ How do I measure portfolio risk?
  ❓ What are risk-adjusted returns?

------------------------------------------------------------

## 📂 Machine Learning
**Concepts**: 231

**Key Topics:**
  • In [55]: %time model.fit(X, y)
        CPU times: user 537 ms, sys: 14.2 ms, total: 551 ms
        W... [code]
  • self.time_grid = np.array(time_grid)
self.val_env.add_list('time_grid', self.time_grid)

if correlat... [code]
  • In [108]: from sklearn.neural_network import MLPClassifier

In [109]: model = MLPClassifier(solver='... [code]
  • In [69]: data = pd.DataFrame(raw[symbol])

In [70]: data['returns'] = np.log(data / data.shift(1))

... [code]
  • # 
# DX Package 
# 
# Simulation Class -- Square-Root Diffusion 
# 
# square_root_diffusion.py 
# 
#... [code]
  • Methods
=======
add_constant:
    adds a constant (e.g. model parameter)
get_constant:
    gets a co... [code]
  • In [32]: story_html = ek.get_news_story(news.iloc[1, 2]) 

In [33]: from bs4 import BeautifulSoup 

... [code]
  • # 
# DX Package
# 
# Simulation Class -- Jump Diffusion
# 
# jump_diffusion.py
# 
# Python for Finan... [code]
  • In [18]: from sklearn.linear_model import LinearRegression  ❶

In [19]: model = LinearRegression()  ... [code]
  • In [28]: from pymc3.distributions.timeseries import GaussianRandomWalk

In [29]: subsample_alpha = 5... [code]

**Content Types:**
  • text: 153
  • code: 37
  • list_item: 24
  • section_header: 15
  • caption: 1
  • picture: 1

**What You Can Ask:**
  ❓ How can I use ML for trading?
  ❓ What are the best algorithms for finance?
  ❓ How do I build predictive models?
  ❓ What is feature engineering?

------------------------------------------------------------

## 📂 Trading Strategies
**Concepts**: 177

**Key Topics:**
  • In [108]: def automated_strategy(data, dataframe):
         global min_bars, position, df
         l... [code]
  • # 
# Automated ML-Based Trading Strategy for FXCM 
# Online Algorithm, Logging, Monitoring 
# 
# Pyt... [code]
  • In [21]: from itertools import product

In [22]: sma1 = range(20, 61, 4) ①
         sma2 = range(180... [code]
  • {'type': 'line_chart', 'x_axis': {'label': 'Date', 'range': ['2010', '2018']}, 'y_axis': {'label': '... [picture]
  • In [24]: results.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 121 entries, 0 to 120
Data... [code]
  • equs.append(equ)
cap = 'capital_{:.2f}'.format(f)
data[equ] = 1
data[cap] = data[equ] * f
for i, t i... [code]
  • {'type': 'line_chart', 'title': None, 'x_axis': {'label': 'Date', 'range': ['2011', '2018']}, 'y_axi... [picture]
  • In [14]: data['Returns'] = np.log(data[symbol] / data[symbol].shift(1)) ①
In [15]: data['Strategy'] ... [code]
  • This second edition of Python for Finance is more of an upgrade than an
update. For example, it adds... [text]
  • All in all, the brief analysis in this section reveals some support for both the RWH and the EMH. Fo... [text]

**Content Types:**
  • text: 119
  • caption: 15
  • section_header: 15
  • list_item: 13
  • code: 10
  • picture: 5

**What You Can Ask:**
  ❓ How do I backtest strategies?
  ❓ What are momentum strategies?
  ❓ How do I implement mean reversion?
  ❓ What are performance metrics?

------------------------------------------------------------

## 📂 Portfolio Theory
**Concepts**: 113

**Key Topics:**
  • Methods
=======
get_positions:
prints information about the single portfolio positions
get_statistic... [code]
  • {'code': {'path_no': 888, 'path_gbm': "portfolio.underlying_objects['gbm'].get_instrument_values()[:... [picture]
  • {'type': 'histogram', 'title': 'Portfolio frequency distribution of present values', 'x_axis': {'lab... [picture]
  • In [69]: optv = sco.minimize(port_vol, eweights,
                            method='SLSQP', bounds=... [code]
  • In [71]: class PortfolioPosition(object):
    def __init__(self, financial_instrument, position_size... [code]
  • As verified here, the weights indeed add up to 1; i.e., ∑w, = 1 , where I is
the number of financial... [text]
  • In [39]: %%time
    i = 0
    opt_local = spo.fmin(mean_squared_error, opt_global,
                 ... [code]
  • So far, all the optimization efforts have focused on the sequential code execution. In particular wi... [text]
  • In [131]: path_gbm = port_corr.underlying_objects['gbm'].\ 
                     get_instrument_valu... [code]
  • In [76]: result
Out[76]:      fun: -9.700883611487832
     jac: array([-0.48508096, -0.48489535])
 m... [code]

**Content Types:**
  • text: 78
  • section_header: 19
  • code: 7
  • list_item: 7
  • picture: 2

**What You Can Ask:**
  ❓ How does Modern Portfolio Theory work?
  ❓ What is the efficient frontier?
  ❓ How do I optimize portfolios?
  ❓ What is asset allocation?

------------------------------------------------------------

## 📂 Mathematical Finance
**Concepts**: 79

**Key Topics:**
  • In [38]: from geometric_brownian_motion import geometric_brownian_motion

In [39]: gbm = geometric_b... [code]
  • based algorithm to derive digits for the number pi (π).2 The basic idea relies
on the fact that the ... [text]
  • Although rather specific in nature, these results are in contrast to what the
random walk hypothesis... [text]
  • Monte Carlo simulation is a task that lends itself well to parallelization. One
approach would be to... [text]
  • In finance, there are many algorithms that are useful for parallelization. Some of these even allow ... [text]
  • A similar picture emerges for the dynamic simulation and valuation approach, whose results are repor... [text]
  • Consider now the following parameterization for the geometric Brownian motion and the valuation func... [text]
  • A major element of Bayesian regression is Markov chain Monte Carlo (MCMC) sampling.7 In principle, t... [text]
  • The simulation is based on the parameterization for the Monte Carlo simulation as shown here, genera... [text]
  • mcs_pi_py() is a Python function using a for loop and implementing the Monte Carlo simulation in a m... [text]

**Content Types:**
  • text: 49
  • section_header: 12
  • list_item: 9
  • caption: 8
  • code: 1

**What You Can Ask:**
  ❓ What is geometric Brownian motion?
  ❓ How do Monte Carlo simulations work?
  ❓ What are stochastic processes?
  ❓ How do I model random variables?

------------------------------------------------------------

## 📂 Fixed Income
**Concepts**: 13

**Key Topics:**
  • In [107]: class Vector(object):
     def __init__(self, x=0, y=0, z=0):
         self.x = x
        ... [code]
  • In [103]: class Vector(Vector):
     ...:     def __iter__(self):
     ...:         for i in range(l... [code]
  • In [23]: import keyword

In [24]: keyword.kwlist
Out[24]: ['False',
 'None',
 'True',
 'and',
 'as',... [code]
  • The discount factors can also be interpreted as the value of a unit zero-
coupon bond (ZCB) as of to... [text]
  • The three estimates shown are rather close to the original values (4, 2, 2). However, the whole proc... [text]
  • Composition is similar to aggregation, but here the single objects cannot exist independently of eac... [text]
  • Cox, John, Jonathan Ingersoll, and Stephen Ross (1985). "A Theory of the Term Structure of Interest ... [list_item]
  • Cox, John, Jonathan Ingersoll, and Stephen Ross (1985). "A Theory of the Term Structure of Interest ... [list_item]
  • A unit zero-coupon bond pays exactly one currency unit at its maturity and no coupons between today ... [list_item]
  • Often, logical operators are applied on bool objects, which in turn yields another bool object: [text]

**Content Types:**
  • text: 7
  • code: 3
  • list_item: 3

**What You Can Ask:**
  ❓ How do bond pricing models work?
  ❓ What is duration and convexity?
  ❓ How do I model yield curves?
  ❓ What are credit risk models?

------------------------------------------------------------

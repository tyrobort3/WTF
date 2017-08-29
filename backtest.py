import urllib2 #only available for python 2.7, not for python 3.6
#from urllib.request import urlopen
import json
import matplotlib.pyplot as plt


#get data 
market="BTC-ETH"
#this command has error in python3.6, it is why we use python 2.7
data=json.loads(urllib2.urlopen("https://bittrex.com/Api/v2.0/pub/market/GetTicks?marketName="+market+"&tickInterval=fiveMin").read())['result']

print data[0]

from pandas import pandas as pd

df = pd.DataFrame(data).set_index("T")

df.index = pd.DatetimeIndex(df.index)

import numpy as np

#define returns 
df['returns'] = np.log(df['C'] / df['C'].shift(1))

cols = []

# position
for momentum in [15]: 
    col = 'position_%s' % momentum
    #df[col] = np.sign(df['returns'].rolling(momentum,min_periods=momentum).mean())
    df[col] = np.sign(df['returns'].rolling(momentum,min_periods=momentum).mean()) #here 
    cols.append(col)
    
%matplotlib inline
import matplotlib

import seaborn as sns

sns.set()

#strategy
strats = ['returns']

for col in cols:
    strat = 'strategy_%s' % col.split('_')[1]
    df[strat] = df[col].shift(1) * df['returns']
    strats.append(strat)

df[strats].dropna().cumsum().apply(np.exp).plot()
plt.show()


#next, how to calculate return based on buy, sell signal?(am) then, how to implement strategy. (pm)


######################
#bt backtest example, code from http://pmorissette.github.io/bt/examples.html
#try bt
import bt
data2 = bt.get('msft', start='2010-01-01')

data=df[['C']]

sma = data.rolling(24).mean()
plot = bt.merge(data, sma).plot(figsize=(15, 5))
plt.show()
signal = data > sma
# first we create the Strategy
s = bt.Strategy('above2hoursma', [bt.algos.SelectWhere(data > sma),
                               bt.algos.WeighEqually(),
                               bt.algos.Rebalance()])

# now we create the Backtest
t = bt.Backtest(s, data)


s_long = bt.Strategy('long', [bt.algos.RunOnce(),
                           bt.algos.SelectAll(),
                           bt.algos.WeighEqually(),
                           bt.algos.Rebalance()])
t_long=bt.Backtest(s_long,data)
                         
# and let's run it!
res = bt.run(t,t_long)
# what does the equity curve look like?
res.plot()  #here, add price to this? later 
res.plot_security_weights()
# and some performance stats
res.display()


############################
#try portfolio code from https://s3.amazonaws.com/quantstart/media/powerpoint/an-introduction-to-backtesting.pdf

df['signal']=(data>sma) 

df2=df[['C','signal']]

df2['position']=df2['signal']-df2['signal'].shift(1)

df2['portfolio']=df2['position']*df2['C']

df2['pos_diff']=df2['position']-df2['position'].shift(1)

df2['holdings']=(df2['position']*df2['C']).sum(axis=1)

#from internet 
portfolio['holdings'] = (self.positions*self.bars['Adj Close']).sum(axis=1)
 portfolio['cash'] = self.initial_capital - (pos_diff*self.bars['Adj Close']).sum(axis=1).cumsum()
 # Sum up the cash and holdings to create full account ‘equity’, then create the percentage returns
 portfolio['total'] = portfolio['cash'] + portfolio['holdings']
 portfolio['returns'] = portfolio['total'].pct_change()
 
















#old code; example to use bt 
def above_sma(tickers, sma_per=50, start='2010-01-01', name='above_sma'):
    """
    Long securities that are above their n period
    Simple Moving Averages with equal weights.
    """
    # download data
    data = bt.get(tickers, start=start)
    # calc sma
    sma = data.rolling(sma_per).mean()

    # create strategy
    s = bt.Strategy(name, [SelectWhere(data > sma),
                           bt.algos.WeighEqually(),
                           bt.algos.Rebalance()])

    # now we create the backtest
    return bt.Backtest(s, data)

# simple backtest to test long-only allocation
def long_only_ew(tickers, start='2010-01-01', name='long_only_ew'):
    s = bt.Strategy(name, [bt.algos.RunOnce(),
                           bt.algos.SelectAll(),
                           bt.algos.WeighEqually(),
                           bt.algos.Rebalance()])
    data = bt.get(tickers, start=start)
    return bt.Backtest(s, data)

# create the backtests
#tickers = 'msft'
#sma10 = above_sma(tickers, sma_per=10, name='sma10')
#sma20 = above_sma(tickers, sma_per=20, name='sma20')
#sma40 = above_sma(tickers, sma_per=40, name='sma40')
benchmark = long_only_ew('spy', name='spy')

# run all the backtests!
res2 = bt.run(sma10, benchmark)
res2.plot(freq='m')

res2.display()












#install any new package 
#install pandas in terminal: application 
#pip install wheel
pip install pandas


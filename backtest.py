import urllib2 #only available for python 2.7, not for python 3.6
#from urllib.request import urlopen
import json
import matplotlib.pyplot as plt
import matplotlib
from pandas import pandas as pd
import numpy as np

#get data 
market="BTC-ETH"
#market="USDT-BTC"
#this command has error in python3.6, it is why we use python 2.7
data=json.loads(urllib2.urlopen("https://bittrex.com/Api/v2.0/pub/market/GetTicks?marketName="+market+"&tickInterval=fiveMin").read())['result']

#1499127220008

print data[0]



df = pd.DataFrame(data).set_index("T")

df.index = pd.DatetimeIndex(df.index)





#example price>sma20
#define signal, return, strategy
sma20 =df['C'].rolling(20).mean()

a=pd.concat([df['C'],sma20],axis=1)
a.plot()

signal_cols=[]

for i in [20]:
	col='signal_%s' %i
	df[col]= (df['C'] > sma20)*1
	signal_cols.append(col)  

#define returns 
df['returns'] = np.log(df['C'] / df['C'].shift(1))

for col in signal_cols:
	strat='mystrategy_%s' % col.split('_')[1]
	df[strat]=df[col].shift(1) * df['returns']


df[strat].dropna().cumsum().apply(np.exp).plot()

plt.title('my code')

plt.show()

col_name=['returns', 'mystrategy_20']

df[col_name].dropna().cumsum().apply(np.exp).plot()

plt.title('market return vs. sma20')

return_vec=df[col_name].dropna().cumsum().apply(np.exp)


#buy #define rolling volume, diff , if diff> threshold, buy
num_windows=[12] #one hour, 60/5
thre_s=[1,0.1]
refresh_s=[1]
v_cols=list()
j=0.01
for i in num_windows: #here, check code 
	col='rollingV_%s' %i
	col_2='rollingV_diff_%s' %i 
	col_3='rollingV_ind_%s' %i 
	df[col]=df['V'].rolling(i,min_periods=3).mean()
	#v_cols=v_cols.append(col) #append has error , debug here 
	df[col_2]=(df[col]-df[col].shift(1))/df[col].shift(1)
	df[col_3]=np.where(df[col_2]>j, 1, 0)

i2=12
j2=0.001
col_4='price_ind_%s' %i2
df[col_4]=np.where((df['C']-df['C'].shift(12))/df['C'].shift(12)>j2,1,0)

df['ind_both']=df['rollingV_ind_12']* df['price_ind_12']

#sell: find the buy signal, loop after the buy signal to get sell position, get signal 
df['buy_ind']=np.where((df['ind_both']-df['ind_both'].shift(1)==1),1,0)

price_buy=np.nan
price_peak=np.nan
df['sell_ind']=0
df['buy_new']=0
df['holding_status']=0


for i,data in df.iterrows():
	row_i=df.index.get_loc(i)
	if(row_i==0):
		df.loc[i,'sell_ind']=0
	elif(row_i>0):
		index_previous=df.index[row_i-1]
		holding_status= (df.loc[:index_previous,'buy_new'] - df.loc[:index_previous,'sell_ind']).sum()
		df.loc[i,'holding_status']=holding_status
		if holding_status==0 and data['buy_ind']==1:
			df.loc[i,'buy_new']=1 
			price_buy=data['C']
			price_peak = data['C']
		if holding_status==0 and data['C']> price_peak:
			price_peak = data['C']
		elif holding_status == 1 and (data['C']-price_buy)/price_buy<-0.2 and (data['C']-price_peak)/price_peak<-0.1: 
			df.loc[i,'sell_ind']=1

#next, get one trading pair and test 


		





##########################
#test stance idea , not the same as using 'signal'

x=0

df['Stance']=np.where( (df['C'] - sma20) > x, 1, 0)

df['Stance']=np.where( (df['C'] - sma20) < x, -1, df['Stance'])

df['Stance'].value_counts()

a=df['signal_20']-df['signal_20'].shift(1)

a.fillna().value_counts() 

df['strategy_stance']=df['returns']*df['Stance'].shift(1)

col_name=['returns', 'mystrategy_20', 'strategy_stance']

df[col_name].dropna().cumsum().apply(np.exp).plot()




#################
#code code , have bugs in it. 
cols = []

# position
for momentum in [15]: 
	col = 'position_%s' % momentum
    #df[col] = np.sign(df['returns'].rolling(momentum,min_periods=momentum).mean())
    df[col] = np.sign(df['returns'].rolling(momentum,min_periods=momentum).mean()) #here 
    cols.append(col)
    
%matplotlib inline


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


######################
#bt backtest example, code from http://pmorissette.github.io/bt/examples.html
#We can use bt for single and portfolio trading pairs 
import bt

data=df[['C']]
sma = data.rolling(20).mean()
plot = bt.merge(data, sma).plot(figsize=(15, 5))
plt.show()
signal = data > sma
# first we create the Strategy
s = bt.Strategy('above2hoursma', [bt.algos.SelectWhere(data > sma), #do not replace with df['C'] or sma20
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
#code for portfolio; buy 100 shares each time, not applicable; 
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


import pandas as pd
import numpy as np
import math
import os.path
import time
def StochRSI_EMA(series, period=14, smoothK=3, smoothD=3):
    # Calculate RSI 
    delta = series.diff().dropna()
    ups = delta * 0
    downs = ups.copy()
    ups[delta > 0] = delta[delta > 0]
    downs[delta < 0] = -delta[delta < 0]
    ups[ups.index[period-1]] = np.mean( ups[:period] ) #first value is sum of avg gains
    ups = ups.drop(ups.index[:(period-1)])
    downs[downs.index[period-1]] = np.mean( downs[:period] ) #first value is sum of avg losses
    downs = downs.drop(downs.index[:(period-1)])
    rs = ups.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() / \
         downs.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() 
    rsi = 100 - 100 / (1 + rs)

    # Calculate StochRSI 
    stochrsi  = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    #stochrsi_K = stochrsi.ewm(span=smoothK).mean()
    #stochrsi_D = stochrsi_K.ewm(span=smoothD).mean()

    return stochrsi
data = pd.read_csv("BTCUSDT-2h-data.csv", delimiter = ',')

counter = 0

data["bsk"] = 0 #BUY = 1 SELL = 2 KEEP = 0
data["bsk"][0] = 1
y = True
vtb1 = data.iloc[0]["close"]
test2 = data.iloc[4]["close"]
tests = pd.Series([vtb1,test2])
test = tests.pct_change()
#print(test)
data = data.drop(columns=["high","low","close_time","quote_av","tb_base_av","tb_quote_av","ignore"])

counter = 1
doit = True
pos = 0
if doit:
    while counter < (data.shape[0] ):
        d1 = data.iloc[counter]["close"]
        s = pd.Series([vtb1,d1])
        overall = s.pct_change()
        #true is find sell position
        
        if y:
            if  d1> vtb1:
                vtb1 = d1
                pos = counter
            elif (overall[1]) <- 0.05:
                data["bsk"][pos] = 2
                y = False
                vtb1 = d1
        else:
            if d1 < vtb1:
                vtb1 = d1
                pos = counter
            elif (overall[1]) > 0.05:
                y = True
                data["bsk"][pos] = 1
                vtb1 = d1

        counter += 1
    #ema calcs
    data['ewm8'] = data['close'].ewm(span=8,min_periods=0,adjust=False,ignore_na=False).mean()
    data['ewm13'] = data['close'].ewm(span=13,min_periods=0,adjust=False,ignore_na=False).mean()
    data['ewm21'] = data['close'].ewm(span=21,min_periods=0,adjust=False,ignore_na=False).mean()
    data['ewm55'] = data['close'].ewm(span=55,min_periods=0,adjust=False,ignore_na=False).mean()
    #stocRSI
    data['StotchRSI'] = StochRSI_EMA(data["close"])


data.to_csv(r"C:\Users\ZeePu\Documents\python workspace\bitcoin model\data\2hdata.csv")

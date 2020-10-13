import datetime
import backtrader as bt
#from backtesting import Backtest, Strategy
#from Strategy import MyStrategy


class MyStrategy(bt.Strategy):
    def next(self):
        print(self.datas[0].close[0]) #Print close prices


#Instantiate Cerebro engine
cerebro = bt.Cerebro()

#Set data parameters and add to Cerebro
data = bt.feeds.YahooFinanceCSVData(
    dataname='MSFT.csv',
    fromdate=datetime.datetime(2019, 10, 1),
    todate=datetime.datetime(2020, 9, 30))

cerebro.adddata(data)

#Add strategy to Cerebro
cerebro.addstrategy(MyStrategy)

#Run Cerebro Engine
cerebro.run()
cerebro.plot()

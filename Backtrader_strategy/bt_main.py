import datetime
import backtrader as bt
from backtrader import Cerebro
#from mycomm import comm_amzn

from strategies import *

#Instantiate Cerebro engine
cerebro: Cerebro = bt.Cerebro()

#Set data parameters and add to Cerebro
data = bt.feeds.YahooFinanceCSVData(
    dataname='GOOG.csv',
    fromdate=datetime.datetime(2015, 10, 1),
    todate=datetime.datetime(2018, 10, 1),
    reverse=False)


cerebro.adddata(data)

#Add strategy to Cerebro
cerebro.addstrategy(MAcrossover)
# Add commission rate of 0.1% per trade
cerebro.broker.setcommission(commission=0.0025, name='GOOG')
#Default position size
cerebro.addsizer(bt.sizers.SizerFix, stake=1000)

if __name__ == '__main__':
    #Run Cerebro Engine
    cerebro.broker.set_cash(1000000)
    start_portfolio_value = cerebro.broker.getvalue()


    cerebro.run()

    end_portfolio_value = cerebro.broker.getvalue()
    pnl = end_portfolio_value - start_portfolio_value
    print('Starting Portfolio Value: %.2f' % start_portfolio_value)
    print('Final Portfolio Value: %.2f' % end_portfolio_value)
    print('Pnl: %.2f' % pnl)

cerebro.plot()
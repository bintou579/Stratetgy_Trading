import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.models import load_model
from pylab import rcParams
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from tensorflow.keras.models import load_model
import tensorflow as tf





#get data
@st.cache
def load_data():
    return pd.read_csv("CSV/data_stockprice.csv")

df = load_data()

#get data API
@st.cache
def get_data(symbol):
    tickerData = yf.Ticker(symbol)
    data = tickerData.history(period='1d', start='2010-10-1', end="2020-10-2")
    data['companies'] = symbol
    data.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
    return data


# split data train /test
@st.cache
def split_data(df, symbol):
    data = df.filter(['Close']).loc[df['companies'] == symbol]
    dataset = data.values.reshape(-1, 1)
    training_data_len = int(np.ceil(len(dataset) * .8))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test, scaler, data, dataset, training_data_len

# strategy
def moyenne_mobile(symbol, df_price):
    # 30 jours 
    SMA30 = pd.DataFrame()
    SMA30["Close Price"] = df_price["Close"].rolling(30).mean()

    # 100 jours 

    SMA100 = pd.DataFrame()
    SMA100["Close Price"] = df_price["Close"].rolling(100).mean()
    
    #Create a new dataframe to store all the datasets
    data =pd.DataFrame()
    data[symbol] =  df_price["Close"]
    data['SMA30'] = SMA30["Close Price"]
    data['SMA100'] = SMA100["Close Price"]
   
    return data, df_price
    


def signal(data, symbol, df_price):
    data, df_price = moyenne_mobile(symbol, df_price)
    sigPriceBuy = []
    sigPriceSell= []
    flag = -1
    
    for i in range(len(data)):
        if data['SMA30'][i] > data['SMA100'][i]:
            if flag != 1:
                sigPriceBuy.append(data[symbol][i])
                sigPriceSell.append(np.nan)
                flag = 1
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
        elif data['SMA30'][i] < data['SMA100'][i]:
            if flag != 0:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(data[symbol][i])
                flag = 0
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
        else:
            sigPriceBuy.append(np.nan)
            sigPriceSell.append(np.nan)
            
    return(sigPriceBuy,sigPriceSell)

pf = 1000   
@st.cache
def porto(data):
    buy_sell = data[(data['Buy_Signal_Price'] > 0) | (data['Sell_Signal_Price'] > 0)]
    sell =buy_sell['Sell_Signal_Price'].sum()
    buy = buy_sell['Buy_Signal_Price'][:-1].sum()
    gain_perte = (sell*1.01) - (buy*1.01)
    pf_final = pf + gain_perte
    return pf_final

def main():
    df = load_data()
    
    page = st.sidebar.selectbox("Choose a page", ['Homepage', 'Exploration', 'Prediction', 'Strategy'])

    if page == 'Homepage':
        st.title("Stock Market Predictions with LSTM")
        img1 = Image.open("Images/accueille.PNG")
        st.image(img1, use_column_width=True)
        st.subheader("Stock Market for Microsoft, Amazon and Google within the last 10 years 01/10/2010 au 01/10/2020")
        st.markdown("The goal of this project is to predict with the model LSTM the closing stock price and backtest stock trading strategies.")


    elif page == 'Exploration':
        
        pages = st.sidebar.selectbox("Choose a company", ['Microsoft', 'Amazon', 'Google'])
        st.title("Explore Data")
        if pages =='Microsoft':
            st.subheader("DataFrame Microsoft")
            st.dataframe(df)
            st.subheader("History Closing Prices")
            df = df.loc[df['companies']== "MSFT"]           
            rcParams['figure.figsize'] = 10,10 
            plt.title("History Closing Prices Microsoft")
            plt.plot(range(df.shape[0]), df['Close'], label = "Microsoft")
            plt.xticks(range(0,df.shape[0],95),df['Date'].loc[::95],rotation=40)
            plt.xlabel('Date',fontsize=18)
            plt.ylabel('Closing Prices',fontsize=18)
            plt.legend()
            st.pyplot() 
            st.subheader("statistical analysis")
            st.dataframe(df.describe())
        elif pages == "Amazon":
            st.title("DataFrame Amazon")
            df = df.loc[df['companies']== "AMZN"]
            st.dataframe(df)
            
            st.subheader("History Closing Prices")
            rcParams['figure.figsize'] = 10, 10
            plt.title("History Closing Prices Amazon")
            plt.plot(range(df.shape[0]), df['Close'], label = "Amazon")
            plt.xticks(range(0,df.shape[0],95),df['Date'].loc[::95],rotation=40)
            plt.xlabel('Date',fontsize=18)
            plt.ylabel('Closing Prices',fontsize=18)
            plt.legend()
            st.pyplot()
            st.subheader("Statistical analysis")
            st.dataframe(df.describe())
        else :
            st.title("DataFrame Google")
            df = df.loc[df['companies']== "GOOG"]
            st.dataframe(df)
            st.subheader("History Closing Prices")
            rcParams['figure.figsize'] = 10, 10
            plt.title("History Closing Prices")
            plt.plot(range(df.shape[0]), df['Close'], label = "Google")
            plt.xticks(range(0,df.shape[0],95),df['Date'].loc[::95],rotation=40)
            plt.xlabel('Date',fontsize=18)
            plt.ylabel('Closing Prices',fontsize=18)
            plt.legend()
            st.pyplot()
            st.subheader("statistical analysis")
            st.dataframe(df.describe())
  
            
    elif page=='Prediction':
        pag = st.sidebar.selectbox("Choose a company", ['Microsoft', 'Amazon', 'Google'])
        if pag =='Microsoft':
            st.title('Modelling')
            st.subheader("Microsoft")
            model = load_model("Modeles/SaveModel/MSFT_model.h5")
            x_train, y_train, x_test, y_test, scaler, data, dataset, training_data_len = split_data(df, "MSFT")
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)
            rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
            st.subheader('We are going to predict a closing price with LSTM network')
            st.write('RMSE :  ' + str(rmse))
            
            st.subheader('Visualize the predicted stock price with original stock price')
            st.markdown("The exact price points from our predicted price is close to the actual price")
            train = data[:training_data_len]
            valid = data[training_data_len:]
            valid['Predictions'] = predictions
            predict = pd.read_csv("CSV/prediction_MSFT.csv")
            plt.figure(figsize=(16, 8))
            plt.title('Prédictions vs. Actual Prices')
            plt.xlabel('Date', fontsize=18)
            plt.ylabel('Close USD ($)', fontsize=18)
            plt.plot(train['Close'])
            plt.plot(valid[['Close', 'Predictions']])
            plt.legend(['Train', 'Valeurs actuelles', 'Prédictions'], loc='lower right')
            st.pyplot()
            st.subheader("Dataframe Prédictions vs. Actual Prices")
            st.dataframe(predict)
        elif pag =='Amazon':
            st.title('Modelling')
            st.subheader("Amazon")
            model_a = load_model('Modeles/SaveModel/AMZN_model.h5', compile = False)
            
            x_train, y_train, x_test, y_test, scaler, data, dataset, training_data_len = split_data(df, "AMZN")
            predictions = model_a.predict(x_test)
            predictions = scaler.inverse_transform(predictions)
            rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
            st.subheader('We are going to predict a closing price with LSTM network')
            st.write('RMSE: ' + str(rmse))
            st.subheader('Visualize the predicted stock price with original stock price')
            st.markdown("The exact price points from our predicted price is close to the actual price")
            train = data[:training_data_len]
            valid = data[training_data_len:]
            valid['Predictions'] = predictions
            plt.figure(figsize=(16, 8))
            plt.title('Prédictions vs. valeurs réeles')
            plt.xlabel('Date', fontsize=18)
            plt.ylabel('Close USD ($)', fontsize=18)
            plt.plot(train['Close'])
            plt.plot(valid[['Close', 'Predictions']])
            plt.legend(['Train', 'Valeurs actuelles', 'Prédictions'], loc='lower right')
            st.pyplot()
            predict2 = pd.read_csv("CSV/predict_amzn.csv")
            st.subheader("Dataframe Prédictions vs. Actual Prices")
            st.dataframe(predict2)
        else:
            st.title('Modelling')
            st.subheader("Google")
            model = load_model('Modeles/SaveModel/GOOG_model.h5', compile = False)
            x_train, y_train, x_test, y_test, scaler, data, dataset, training_data_len = split_data(df, "GOOG")
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)
            rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
            st.subheader('We are going to predict a closing price with LSTM network')
            st.write('RMSE: ' + str(rmse))
            st.subheader('Visualize the predicted stock price with original stock price')
            st.markdown("The exact price points from our predicted price is close to the actual price")
            train = data[:training_data_len]
            valid = data[training_data_len:]u
            valid['Predictions'] = predictions
            plt.figure(figsize=(16, 8))
            plt.title('Prédictions vs. valeurs réeles')
            plt.xlabel('Date', fontsize=18)
            plt.ylabel('Close USD ($)', fontsize=18)
            plt.plot(train['Close'])
            plt.plot(valid[['Close', 'Predictions']])
            plt.legend(['Train', 'Valeurs actuelles', 'Prédictions'], loc='lower right')
            st.pyplot()
            predict3 = pd.read_csv("CSV/predict-goog.csv")
            st.subheader("Dataframe Prédictions vs. Actual Prices")
            st.dataframe(predict3)
           
            
    else:
        
        st.title("Strategy Trading")
        pg = st.sidebar.selectbox("Choose a company", ['Microsoft', 'Amazon', 'Google'])
        if pg =='Microsoft':
            msft = get_data("MSFT")
            data_f, msft = moyenne_mobile("MSFT", msft)
            signal_f = signal(data_f, "MSFT", msft)
            data_f['Buy_Signal_Price'] = signal_f[0]
            data_f['Sell_Signal_Price']= signal_f[1]
            plt.figure(figsize=(15,5))
            plt.plot(data_f['MSFT'],label = 'MSFT', alpha = 0.30)
            plt.plot(data_f['SMA30'], label = 'SMA30', alpha = 0.30)
            plt.plot(data_f['SMA100'], label = 'SMA100', alpha = 0.30)
            plt.scatter(data_f.index,data_f['Buy_Signal_Price'], label = 'Buy', marker = '^', color = 'green')
            plt.scatter(data_f.index,data_f['Sell_Signal_Price'], label = 'Sell', marker = 'v', color = 'red')
            plt.title('Microsoft Close Price Data, Buy & Sell Signal')
            plt.xlabel('Octobre 01, 2010 - Octobre 02, 2020')
            plt.ylabel('Close Price of USD ($)')
            plt.legend(loc='upper left', fontsize=10)
            st.pyplot() 
            pf_final = porto(data_f)
            st.write(f"Valeur intiale du portefeuille est {pf}")
            st.write("Valeur finale du portefeuille =  " +  str(pf_final))
        elif pg == "Amazon":
            amzn = get_data("AMZN")
            data_z, amzn = moyenne_mobile("AMZN", amzn)
            signal_z = signal(data_z, "AMZN", amzn)
            data_z['Buy_Signal_Price'] = signal_z[0]
            data_z['Sell_Signal_Price']= signal_z[1]
            plt.figure(figsize=(15,5))
            plt.plot(data_z['AMZN'],label = 'AMZN', alpha = 0.30)
            plt.plot(data_z['SMA30'], label = 'SMA30', alpha = 0.30)
            plt.plot(data_z['SMA100'], label = 'SMA100', alpha = 0.30)
            plt.scatter(data_z.index,data_z['Buy_Signal_Price'], label = 'Buy', marker = '^', color = 'green')
            plt.scatter(data_z.index,data_z['Sell_Signal_Price'], label = 'Sell', marker = 'v', color = 'red')
            plt.title('AMAZON Close Price Data, Buy & Sell Signal')
            plt.xlabel('Jan 02, 2015 - Jan 31, 2020')
            plt.ylabel('Close Price of USD ($)')
            plt.legend(loc='upper left', fontsize=10)
            st.pyplot()

            pf_final = porto(data_z)
            st.write(f"Valeur intiale du portefeuille est {pf}")
            st.write("Valeur finale du portefeuille =  " +  str(pf_final))
            
        else :
            goog = get_data("GOOG")
            data_goog, goog = moyenne_mobile("GOOG", goog)
            signal_g = signal(data_goog, "GOOG", goog)
            data_goog['Buy_Signal_Price'] = signal_g[0]
            data_goog['Sell_Signal_Price']= signal_g[1]
            plt.figure(figsize=(15,5))
            plt.plot(data_goog['GOOG'],label = 'GOOG', alpha = 0.30)
            plt.plot(data_goog['SMA30'], label = 'SMA30', alpha = 0.30)
            plt.plot(data_goog['SMA100'], label = 'SMA100', alpha = 0.30)
            plt.scatter(data_goog.index,data_goog['Buy_Signal_Price'], label = 'Buy', marker = '^', color = 'green')
            plt.scatter(data_goog.index,data_goog['Sell_Signal_Price'], label = 'Sell', marker = 'v', color = 'red')
            plt.title('Google Close Price Data, Buy & Sell Signal')
            plt.xlabel('Jan 02, 2015 - Jan 31, 2020')
            plt.ylabel('Adj Close Price of USD ($)')
            plt.legend(loc='upper left', fontsize=10)
            st.pyplot()

            pf_final = porto(data_goog)
            st.write(f"Valeur intiale du portefeuille est {pf}")
            st.write("Valeur finale du portefeuille =  " +  str(pf_final))




# def predict_price(symbol):
#     tickerData = yf.Ticker(symbol)
#     df = tickerData.history(period='1d', start='2010-10-1', end="2020-9-30")
#     new_df = df.filter(['Close'])
#     last_60_days = new_df[-60:].values
#     scaler =  MinMaxScaler()

#     last_60_days_scaled = scaler.fit_transform(last_60_days)
#     X_test = []
#     X_test.append(last_60_days_scaled)
#     X_test = np.array(X_test)
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#     pred_price = model.predict(X_test)
#     pred_price = scaler.inverse_transform(pred_price)
#     st.write('Prix prédit:' + str(pred_price))


#     actual_price = tickerData.history(period='1d', start="2020-10-03", end="2020-10-03")
#     actual_price = actual_price.Close.values
#     actual_price = np.array(actual_price)
#     st.write('Prix réel:' + str(actual_price))
#     return pred_price, actual_price





if __name__ == '__main__':
    main()


import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import math
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.models import load_model

@st.cache
def load_data():
    return pd.read_csv("CSV/data_stockprice.csv")

@st.cache
def load_msft():
    return pd.read_csv("CSV/microsoft.csv")
@st.cache
def load_data():
    return pd.read_csv("CSV/'google.csv'")
@st.cache
def load_data():
    return pd.read_csv("CSV/data_stockprice.csv")

def main():
    df = load_data()

    page = st.sidebar.selectbox("Choose a page", ['Homepage', 'Exploration', 'Prediction'])

    if page == 'Homepage':
        st.title("Stock Market Predictions with LSTM")
        st.subheader("Stok Market for Microsoft, Amazon and Google  within the last 10 years")
        st.markdown("The goal of this project is to predict with the model LSTM the closing stock price of a corporation SANOFI using the the past 10 years.")
        if 
        image = Image.open("images/image_acceuil.PNG")
        st.image(image, use_column_width=True)
        st.title("Data from Yahoo's Finance API")
        st.dataframe(df)
        
    elif page == 'Exploration':
        st.title('Data Visualization')
        st.subheader('Mid Price history of SANOFI')
        plt.figure(figsize = (20,15))
        plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
        plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Mid Price',fontsize=18)
        st.pyplot()  
            
    else:
        st.title('Modelling')
        st.subheader('We are going to predict a closing price with LSTM network')
        st.write('RMSE' )
        img = Image.open("images/RMSE.PNG")
        st.subheader('Visualize the predicted stock price with original stock price')
        st.markdown("The exact price points from our predicted price is close to the actual price")
        img = Image.open("images/Result_final.PNG")
        st.image(img, use_column_width=True)
        st.subheader('Original stock price & Stock Price Prediction')
        img2 = Image.open("images/Data_predict.PNG")
        st.image(img2, use_column_width=True)


@st.cache
def train_model(df):
    df = load_data()
    data = df.filter(['Close'])
    dataset = data.values

    training_data_len = math.ceil( len(dataset) *.8)
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(dataset)
    
    train_data = scaled_data[0:training_data_len  , : ]
    x_train=[]
    y_train = []

    
    for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(x_train, y_train, batch_size=1, epochs=10)

    #Test data set
    test_data = scaled_data[training_data_len - 60: , : ]
    #Create the x_test and y_test data sets
    x_test = []
    y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0])
    #Convert x_test to a numpy array 
    x_test = np.array(x_test)

    #Reshape the data into the shape accepted by the LSTM
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
    #Getting the models predicted price values
    predictions = model.predict(x_test) 
    predictions = scaler.inverse_transform(predictions)   
    rmse=np.sqrt(np.mean(((predictions- y_test)**2)))

    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    tickerSymbol = "SNY"
    tickerData = yf.Ticker(tickerSymbol)
    predict = tickerData.history(period='1d', start='2010-10-1', end="2020-9-6")
    new_df = predict.filter(['Close'])
    last_60_days = new_df[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    actual_price = tickerData.history(period='1d', start="2020-9-4", end="2020-9-5")
    actual_price = actual_price.Close.values
    actual_price = np.array(actual_price )

    return rmse, valid, pred_price, actual_price



if __name__ == '__main__':
    main()
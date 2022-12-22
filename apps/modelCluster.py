import streamlit as st
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.preprocessing import Normalizer
import plotly.graph_objects as go
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

def app():
    st.title('Model - Clustering K means')

    companies = [
    "Oro",
    "Apple",
    "Walgreen",
    "Northrop Grumman",
    "Boeing",
    "Lockheed Martin",
    "McDonalds",
    "Navistar",
    "IBM",
    "Texas Instruments",
    "MasterCard",
    "Microsoft",
    "General Electrics",
    "Symantec",
    "American Express"]

    companies_dict = {
    "Oro":"GC=F",
    "Apple":"BZ=F",
    "Walgreen":"CL=F",
    "Northrop Grumman":"NG=F",
    "Boeing":"SI=F",
    "Lockheed Martin":"RB=F",
    "McDonalds":"HO=F",
    "Navistar":"PL=F",
    "IBM":"HG=F",
    "Texas Instruments":"PA=F",
    "MasterCard":"ZC=F",
    "Microsoft":"ZO=F",
    "General Electrics":"KE=F",
    "Symantec":"ZR=F",
    "American Express":"ZS=F"}
    
    yf.pdr_override()

    start = st.date_input('Start' , value=pd.to_datetime('2004-08-18'))
    end = st.date_input('End' , value=pd.to_datetime('today'))

    st.title('Predicción de tendencia de acciones')

    user_input = st.text_input('Introducir cotización bursátil' , 'GC=F')

    lista = [user_input]

    y_symbols = ['SCHAND.NS', 'TATAPOWER.NS', 'ITC.NS']

    df2 = pdr.get_data_yahoo([user_input], start,end)


    # Describiendo los datos

    st.subheader('Datos del 2004 al 2022') 
    st.write(df2.describe())
    
    # Candlestick chart
    st.subheader('Gráfico Financiero') 
    candlestick = go.Candlestick(
                            x=df2.index,
                            open=df2['Open'],
                            high=df2['High'],
                            low=df2['Low'],
                            close=df2['Close']
                            )

    fig = go.Figure(data=[candlestick])

    fig.update_layout(
        width=800, height=600,
        title=user_input,
        yaxis_title='Precio'
    )
    
    st.plotly_chart(fig)
    
    
    df = pdr.get_data_yahoo(list(companies_dict.values()),start,end)


    def clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)


    dflimpio = clean_dataset(df)

    stock_open = np.array(dflimpio["Open"]).T 
    stock_close = np.array(dflimpio["Close"]).T 

    movements = stock_close - stock_open

    sum_of_movement = np.sum(movements)

    normalizer = Normalizer() # Define a Normalizer
    norm_movements = normalizer.fit_transform(movements)

    # Define a normalizer
    normalizer = Normalizer()
    # Create Kmeans model
    kmeans = KMeans(n_clusters = 10,max_iter = 1000)
    # Make a pipeline chaining normalizer and kmeans
    pipeline = make_pipeline(normalizer,kmeans)
    # Fit pipeline to daily stock movements
    pipeline.fit(movements)
    labels = pipeline.predict(movements)


    st.title('Clustering K means Labels')

    df1 = pd.DataFrame({"labels":labels,"companies":list(companies)}).sort_values(by=["labels"],axis = 0)


    st.write(df1)


import streamlit as st
import pandas as pd
import datetime
import time
import pickle
from PIL import Image
from sklearn.preprocessing import (LabelEncoder,StandardScaler,PolynomialFeatures)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='Industrial Copper Modelling',layout='wide',initial_sidebar_state='auto')
st.title(':red[Machine Learning - Training & Testing]')
st.write('This app intends to use a trained Machine Learning model of an Industrial Copper Modelling dataset and predict the selling price based on the User input.')
st.write('________')
st.markdown(' ')
st.markdown(' ')
st.markdown(' ')
st.markdown('Step 1 : ----------------------------------------> <b>:violet[Input Collection]</b>',unsafe_allow_html=True)
st.markdown(' ')
selmodel = st.selectbox('Select Model',options=['Regression (predicts selling price)', 'Classification (classifies WON or LOST)'],index=None)
st.markdown(' ')

if(selmodel == 'Regression (predicts selling price)'):
    col1,col2,col3 = st.columns([0.67,0.67,0.67])
    with col1:
        getitemdate = st.date_input('Input Date',value='default_value_today',min_value=datetime.date(2020,7,2),max_value=datetime.date(2024,8,13),format='YYYY-MM-DD',label_visibility='visible')
    with col2:
        getthick = st.number_input('Thickness',min_value=0,max_value=5)
    with col3:
        getdeliverydate = st.date_input('Delivery Date',value='default_value_today',min_value=datetime.date(2020,1,1),max_value=datetime.date(2024,8,13),format='YYYY-MM-DD',label_visibility='visible')
    st.markdown(' ')
    st.markdown(' ')
    st.markdown('Step 2 : ----------------------------------------> <b>:violet[Predict Selling Price]</b>',unsafe_allow_html=True)
    st.markdown(' ')
    predregr = st.button('Predict Price')
    if(predregr):
     with open('trained_model_regress','rb') as f:
        getmodel = pickle.load(f)
        with st.spinner('Loading ML Model'):
            time.sleep(5)
        st.write(':green[ML Model Loaded successfully..!!]')       
        st.markdown(' ')
        st.markdown(' ')
        st.markdown('Step 3 : ----------------------------------------> <b>:violet[Results]</b>',unsafe_allow_html=True)
        st.markdown(' ')
        repitemdate = str(getitemdate).replace('-','')
        repdelidate = str(getdeliverydate).replace('-','')
        inputpd = pd.DataFrame({'item_date':[int(repitemdate)],'thickness':[getthick],'delivery date':[int(repdelidate)]})
        getout = int(getmodel.predict(inputpd))
        with st.spinner('Predicting Price'): 
            time.sleep(5)
        concatstr = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b><font size="10">:green[' + str(getout) + ']</b>'
        st.write('Predicted Selling Price : ',concatstr,unsafe_allow_html=True)
elif(selmodel == 'Classification (classifies WON or LOST)'):
    col4,col5,col6,col7,col8 = st.columns([0.4,0.4,0.4,0.4,0.4])
    with col4:
        getcountry = st.number_input('Country Code',min_value=0)
    with col5:
        getapplication = st.number_input('Application',min_value=0)
    with col6:
        getthick = st.number_input('Thickness',min_value=0)
    with col7:
        getwidth = st.number_input('Width',min_value=0)
    with col8:
        getdeliverydate = st.date_input('Delivery Date',value='default_value_today',min_value=datetime.date(2020,1,1),max_value=datetime.date(2024,8,13),format='YYYY-MM-DD',label_visibility='visible')
    st.markdown(' ')
    st.markdown(' ')
    st.markdown('Step 2 : ----------------------------------------> <b>:violet[Predict Product Classification]</b>',unsafe_allow_html=True)
    st.markdown(' ')
    predclass = st.button('Predict Classification')
    if(predclass):
     with open('trained_model_class','rb') as f:
        getmodel = pickle.load(f)
        with st.spinner('Loading ML Model'):
            time.sleep(5)
        st.write(':green[ML Model Loaded successfully..!!]')       
        st.markdown(' ')
        st.markdown(' ')
        st.markdown('Step 3 : ----------------------------------------> <b>:violet[Results]</b>',unsafe_allow_html=True)
        st.markdown(' ')
        repdelidate = str(getdeliverydate).replace('-','')
        inputpd = pd.DataFrame({'country':[getcountry],'application':[getapplication],'thickness':[getthick],'delivery date':[int(repdelidate)],'width':[getwidth]})
        getout = int(getmodel.predict(inputpd))
        with st.spinner('Predicting Classification'): 
            time.sleep(5)
        if(getout == 0):
            concatstr = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b><font size="10">:green[Lost]</b>'
            st.write('Predicted Product Classification : ',concatstr,unsafe_allow_html=True)
        elif(getout == 1):
            concatstr = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b><font size="10">:green[Won]</b>'
            st.write('Predicted Product Classification : ',concatstr,unsafe_allow_html=True)

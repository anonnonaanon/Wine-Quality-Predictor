import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from PIL import Image

st.write("""
# Wine Quality Prediction App

This app predicts the **Quality of red wine**! 
         The kernel of the model is Random Forest Classifier algorithm that predict quality of wine based on your input parameters. 
         Enjoy and do not drink much!! 
""")
st.write('---')

st.markdown('![cartoon](https://img.freepik.com/premium-vector/bottle-wine-cartoon-style_348404-56.jpg)')


# Loads the Wine dataset
DATA_PATH = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"
X = pd.read_csv(DATA_PATH + "winequality-white.csv", sep=";")
d = {3:0, 4:1, 5:2, 6:3, 7:4, 8:5, 9:6}
X['quality'] = X['quality'].map(d)
y = X['quality']
X.drop('quality', axis=1, inplace=True)



# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    fixed_acidity = st.sidebar.slider('fixed acidity', X['fixed acidity'].min(), X['fixed acidity'].max(), X['fixed acidity'].mean())
    volatile_acidity = st.sidebar.slider('volatile acidity', X['volatile acidity'].min(), X['volatile acidity'].max(), X['volatile acidity'].mean())
    citric_acid = st.sidebar.slider('citric acid', X['citric acid'].min(), X['citric acid'].max(), X['citric acid'].mean())
    residual_sugar = st.sidebar.slider('residual sugar', X['residual sugar'].min(), X['residual sugar'].max(), X['residual sugar'].mean())
    chlorides = st.sidebar.slider('chlorides', X['chlorides'].min(), X['chlorides'].max(), X['chlorides'].mean())
    free_sulfur_dioxide = st.sidebar.slider('free sulfur dioxide', X['free sulfur dioxide'].min(), X['free sulfur dioxide'].max(), X['free sulfur dioxide'].mean())
    total_sulfur_dioxide = st.sidebar.slider('total sulfur dioxide', X['total sulfur dioxide'].min(), X['total sulfur dioxide'].max(), X['total sulfur dioxide'].mean())
    density = st.sidebar.slider('density', X['density'].min(), X['density'].max(), X['density'].mean())
    pH = st.sidebar.slider('pH', X['pH'].min(), X['pH'].max(), X['pH'].mean())
    sulphates = st.sidebar.slider('sulphates', X['sulphates'].min(), X['sulphates'].max(), X['sulphates'].mean())
    alcohol = st.sidebar.slider('alcohol', X['alcohol'].min(), X['alcohol'].max(), X['alcohol'].mean())

    data = {'Fixed Acidity': fixed_acidity,
            'Volatile Acidity': volatile_acidity,
            'Citric Acid': citric_acid,
            'Residual Sugar': residual_sugar,
            'Chlorides': chlorides,
            'Free Sulfur Dioxide': free_sulfur_dioxide,
            'Total Sulfur Dioxide': total_sulfur_dioxide,
            'Density': density,
            'pH': pH,
            'Sulphates': sulphates,
            'Alcohol': alcohol}
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build RF Model
model = RandomForestClassifier()
model.fit(X, y)

# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Wine Quality prediction:')

quality = {0:'Bad, do not play with it', 1:'Bad, do not play with it',
           2:'Pretty bad, but drinkable', 3:'Pretty bad, but drinkable',
           4:'Good wine, enjoy your life!', 5:'Good wine, enjoy your life!',
           6:'Best wine, you smell expensive!'}


st.info(f'**{quality[int(prediction)]}: {int(prediction)} out of 6**')

st.write('You also need to keep in mind that in such classification task there are 7 classes, graded from 0 to 6.')

st.write('---')

st.header('Feature importance based on SHAP values')

image = Image.open('shap.png')
st.image(image)



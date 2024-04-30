import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Assuming an exchange rate; update this as needed
#exchange_rate = 82  # 1 USD = 82 INR

# Load and preprocess the dataset
@st.cache_data  # Cache the data to load only once
def load_data():
    data = pd.read_csv('Housing.csv')
    data = pd.get_dummies(data, drop_first=True)
    return data

housing_data = load_data()
y = housing_data['price']
X = housing_data.drop('price', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train base models and the meta-model
def train_models(X_train, y_train):
    linear_model = LinearRegression()
    tree_model = DecisionTreeRegressor(random_state=42)
    forest_model = RandomForestRegressor(random_state=42)
    gbm_model = GradientBoostingRegressor(random_state=42)

    models = [linear_model, tree_model, forest_model, gbm_model]
    for model in models:
        model.fit(X_train, y_train)
    return models

models = train_models(X_train, y_train)

def generate_meta_features(models, X):
    meta_features = pd.DataFrame({
        "linear_pred": models[0].predict(X),
        "tree_pred": models[1].predict(X),
        "forest_pred": models[2].predict(X),
        "gbm_pred": models[3].predict(X),
    })
    return meta_features

meta_features_train = generate_meta_features(models, X_train)
ridge_meta_model = Ridge(alpha=1.0)
ridge_meta_model.fit(meta_features_train, y_train)

# Streamlit user interface for predictions
st.title('House Price Prediction App Rupee')

area = st.number_input('Area (in square feet)', min_value=500, max_value=10000, value=1500)
bedrooms = st.selectbox('Number of Bedrooms', [1, 2, 3, 4, 5])
bathrooms = st.selectbox('Number of Bathrooms', [1, 2, 3, 4])
stories = st.selectbox('Number of Stories', [1, 2, 3, 4])
parking = st.selectbox('Number of Parking Spaces', [0, 1, 2, 3])
mainroad = st.selectbox('Is the house on the main road?', ['yes', 'no'])
guestroom = st.selectbox('Is there a guest room?', ['yes', 'no'])
basement = st.selectbox('Is there a basement?', ['yes', 'no'])
hotwaterheating = st.selectbox('Is there hot water heating?', ['yes', 'no'])
airconditioning = st.selectbox('Is there air conditioning?', ['yes', 'no'])
prefarea = st.selectbox('Is the house in a preferred area?', ['yes', 'no'])
furnishingstatus = st.selectbox('Furnishing Status', ['furnished', 'semi-furnished', 'unfurnished'])

input_data = pd.DataFrame({
    'area': [area],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'stories': [stories],
    'parking': [parking],
    'mainroad_yes': [1 if mainroad == 'yes' else 0],
    'guestroom_yes': [1 if guestroom == 'yes' else 0],
    'basement_yes': [1 if basement == 'yes' else 0],
    'hotwaterheating_yes': [1 if hotwaterheating == 'yes' else 0],
    'airconditioning_yes': [1 if airconditioning == 'yes' else 0],
    'prefarea_yes': [1 if prefarea == 'yes' else 0],
    'furnishingstatus_semi-furnished': [1 if furnishingstatus == 'semi-furnished' else 0],
    'furnishingstatus_unfurnished': [1 if furnishingstatus == 'unfurnished' else 0],
})

if st.button('Predict Price'):
    input_meta_features = generate_meta_features(models, input_data)
    predicted_price_inr = ridge_meta_model.predict(input_meta_features)[0]
    predicted_price_usd = predicted_price_inr #/ exchange_rate
    st.write(f"The estimated price of the house is: ${int(predicted_price_usd):,}")
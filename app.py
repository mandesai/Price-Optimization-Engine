import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics import mean_absolute_error

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("ratings_and_prices.csv")
    return df

# Preprocess data
@st.cache_data
def preprocess_data(df):
    # Data preprocessing
    df['Selling Price'] = df['Selling Price'].str.replace('Rs', '').astype(float)
    df['MRP'] = df['MRP'].str.replace('Rs', '').astype(float)
    df['Price'] = df['Price'].str.replace('₹', '').astype(float)
    df = df.drop(columns=['City', 'SKU ID', 'Image', 'SKU Name', 'Item Link'])
    df.dropna(inplace=True)
    sample = df['Category'] + " " + df['Sub-Category'] + " " + df['Product'] + " " + df['Brand'] + " " + df['SKU Size']
    df['description'] = sample
    df = df[['description', 'MRP', 'Selling Price', 'In Stock', 'Out of Stock', 'Rating', 'Price']]

    # Apply TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])
    df['vectorized'] = list(tfidf_matrix.toarray())

    return df

# Train model
@st.cache_data
def train_model(df):
    # Splitting data and training Random Forest model
    X_vectorized = np.array(df['vectorized'].tolist())
    X_vectorized_flat = np.vstack(X_vectorized)
    X_numeric = df[['MRP', 'Rating']]
    X = np.hstack((X_numeric, X_vectorized_flat))
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)

    return rf_model

# Predict price
def predict_price(df, rf_model, category, sub_category, product, brand, sku_size):
    product_name = f"{category} {sub_category} {product} {brand} {sku_size}"
    product_row = df[df['description'].str.contains(product_name)].iloc[0]
    features_numeric = product_row[['MRP', 'Rating']].values.reshape(1, -1)
    vectorized_data = np.array(product_row['vectorized'])
    vectorized_data = np.array([np.array(v) for v in vectorized_data])
    features_vectorized_flat = np.hstack(vectorized_data)
    features = np.hstack((features_numeric, features_vectorized_flat.reshape(1, -1)))
    predicted_price = rf_model.predict(features)[0]
    return predicted_price

# Streamlit UI
st.title("Product Price Prediction")
st.write("Enter the details of the product:")
category = st.text_input("Category")
sub_category = st.text_input("Sub-Category")
product = st.text_input("Product")
brand = st.text_input("Brand")
sku_size = st.text_input("SKU Size")

# Load data and preprocess it
df = load_data()
df = preprocess_data(df)

# Train model
rf_model = train_model(df)

if st.button("Predict Price"):
    if category and sub_category and product and brand and sku_size:
        predicted_price = predict_price(df, rf_model, category, sub_category, product, brand, sku_size)
        st.write(f"Predicted Price: {predicted_price}")
    else:
        st.write("Please fill in all the fields.")

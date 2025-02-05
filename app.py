import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the data
st.title("Anomaly Detection in Time-Series Data")

file_path = "Pan002_joined.csv"
df = pd.read_csv(file_path)

df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
df = df.sort_values(by="Timestamp").reset_index(drop=True)

# Scale numerical data
scaler = MinMaxScaler()
df_numeric = df.drop(columns=["Timestamp", "Processed File"], errors='ignore')
df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns, index=df.index)
df_scaled["Timestamp"] = df["Timestamp"]

# Build Autoencoder
input_dim = df_scaled.shape[1] - 1
encoding_dim = input_dim // 2

autoencoder = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(encoding_dim, activation="relu"),
    layers.Dense(input_dim, activation="sigmoid")
])

autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(df_scaled.drop(columns=["Timestamp"], errors='ignore'),
                 df_scaled.drop(columns=["Timestamp"], errors='ignore'),
                 epochs=50, batch_size=64, validation_split=0.1, verbose=0)

# Detect anomalies
reconstructions = autoencoder.predict(df_scaled.drop(columns=["Timestamp"], errors='ignore'))
reconstruction_errors = np.mean(np.abs(df_scaled.drop(columns=["Timestamp"], errors='ignore') - reconstructions), axis=1)
threshold = np.percentile(reconstruction_errors, 99.9)
df_scaled["Anomaly"] = (reconstruction_errors > threshold).astype(int)

# Extract normal and anomalous data
normal_data = df[df_scaled["Anomaly"] == 0]
anomalous_data = df[df_scaled["Anomaly"] == 1]

# Streamlit Sidebar Controls
st.sidebar.header("Filter Data")
start_date = st.sidebar.date_input("Start Date", df["Timestamp"].min().date())
end_date = st.sidebar.date_input("End Date", df["Timestamp"].max().date())

# Filter Data
df_filtered = df[(df["Timestamp"] >= str(start_date)) & (df["Timestamp"] <= str(end_date))]
df_scaled_filtered = df_scaled.loc[df_filtered.index]
normal_data_filtered = df_filtered[df_scaled_filtered["Anomaly"] == 0]
anomalous_data_filtered = df_filtered[df_scaled_filtered["Anomaly"] == 1]

# Plot Data
st.subheader("Anomaly Detection Visualization")
columns_to_plot = df_numeric.columns

for column in columns_to_plot:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(normal_data_filtered["Timestamp"], normal_data_filtered[column], label="Normal", color="green", alpha=0.5)
    ax.scatter(anomalous_data_filtered["Timestamp"], anomalous_data_filtered[column], color="red", marker="x", label="Anomaly")
    ax.set_title(column)
    ax.set_xlabel("Timestamp")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    st.pyplot(fig)

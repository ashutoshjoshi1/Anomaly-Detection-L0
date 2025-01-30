import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def autoencoder_anomaly_detection(df):
    features = [
        'Total duration', 'Integration time', 'Number of cycles', 
        'Saturation index', 'Zenith angle', 'Azimuth', 
        'Temperature columns', 'Humidity', 'Pressure'
    ]
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    input_dim = X_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(int(input_dim/2), activation='relu')(input_layer)
    encoder = Dense(int(input_dim/4), activation='relu')(encoder)
    decoder = Dense(int(input_dim/2), activation='relu')(encoder)
    decoder = Dense(input_dim, activation='linear')(decoder)
    
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, shuffle=True, verbose=0)
    
    reconstructed = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
    
    threshold = np.percentile(mse, 95)
    anomalies = df[mse > threshold]
    return anomalies

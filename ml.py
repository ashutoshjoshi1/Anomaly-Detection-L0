from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def isolation_forest_anomaly_detection(df):
    # Select relevant numerical features
    features = [
        'Total duration', 'Integration time', 'Number of cycles', 
        'Saturation index', 'Zenith angle', 'Azimuth', 
        'Temperature columns', 'Humidity', 'Pressure'
    ]
    
    # Prepare data
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest
    clf = IsolationForest(contamination=0.1, random_state=42)
    y_pred = clf.fit_predict(X_scaled)
    
    # Anomalies are labeled as -1
    anomalies = df[y_pred == -1]
    return anomalies
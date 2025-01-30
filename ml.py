from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def isolation_forest_anomaly_detection(df):
    features = [
        'Total duration', 'Integration time', 'Number of cycles', 
        'Saturation index', 'Zenith angle', 'Azimuth', 
        'Temperature columns', 'Humidity', 'Pressure'
    ]
    
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = IsolationForest(contamination=0.1, random_state=42)
    y_pred = clf.fit_predict(X_scaled)
    
    anomalies = df[y_pred == -1]
    return anomalies
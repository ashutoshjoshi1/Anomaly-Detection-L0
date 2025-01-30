import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats

def statistical_anomaly_detection(df):
    # Focus on numerical columns
    numerical_cols = [
        'Total duration', 'Integration time', 'Number of cycles', 
        'Saturation index', 'Zenith angle', 'Azimuth', 
        'Temperature columns', 'Humidity', 'Pressure'
    ]
    
    # Z-score method
    z_scores = np.abs(stats.zscore(df[numerical_cols]))
    anomalies = (z_scores > 3).any(axis=1)
    return df[anomalies]

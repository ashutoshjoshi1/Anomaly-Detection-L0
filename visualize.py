def visualize_anomalies(df, anomalies):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12,6))
    plt.scatter(df.index, df['Total duration'], label='Normal')
    plt.scatter(anomalies.index, anomalies['Total duration'], color='red', label='Anomalies')
    plt.title('Anomaly Detection')
    plt.legend()
    plt.show()

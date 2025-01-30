import ruptures as rpt

def time_series_change_point_detection(df):
    # Focus on a specific signal (e.g., total duration)
    signal = df['Total duration'].values
    
    # Change point detection
    model = "normal"  # or "rbf", "linear"
    algo = rpt.Pelt(model=model).fit(signal)
    result = algo.predict(pen=10)
    
    # Identify potential anomalous segments
    anomalous_segments = result[:-1]  # Exclude the last trivial segment
    return anomalous_segments

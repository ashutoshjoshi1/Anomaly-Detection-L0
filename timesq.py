import ruptures as rpt

def time_series_change_point_detection(df):
    signal = df['Total duration'].values
    
    model = "normal"
    algo = rpt.Pelt(model=model).fit(signal)
    result = algo.predict(pen=10)
    
    anomalous_segments = result[:-1]
    return anomalous_segments

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def prepare_timestamp_column(df):
    try:
        df['Timestamp'] = pd.to_datetime(df.iloc[:, 1], format='%Y%m%dT%H%M%SZ')
        return df
    except Exception as e:
        print(f"Error converting timestamp: {e}")
        return df

def timestamp_based_anomaly_visualization(df, anomalies):
    df = prepare_timestamp_column(df)
    anomalies = prepare_timestamp_column(anomalies)
    
    plt.figure(figsize=(15, 6))
    plt.scatter(df['Timestamp'], df.index, alpha=0.5, label='Normal', color='blue')
    plt.scatter(anomalies['Timestamp'], anomalies.index, color='red', label='Anomalies', s=100)
    plt.title('Anomalies Distribution Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Data Point Index')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(15, 6))
    df['Hour'] = df['Timestamp'].dt.hour
    anomalies['Hour'] = anomalies['Timestamp'].dt.hour
    
    plt.hist(df['Hour'], bins=24, alpha=0.5, label='Normal', color='blue')
    plt.hist(anomalies['Hour'], bins=24, alpha=0.5, label='Anomalies', color='red')
    plt.title('Anomalies by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(15, 6))
    anomalies_sorted = anomalies.sort_values('Timestamp')
    time_diffs = anomalies_sorted['Timestamp'].diff()
    
    plt.plot(anomalies_sorted['Timestamp'], time_diffs.dt.total_seconds(), marker='o')
    plt.title('Time Differences Between Consecutive Anomalies')
    plt.xlabel('Timestamp')
    plt.ylabel('Time Difference (seconds)')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(15, 6))
    daily_anomalies = anomalies.resample('D', on='Timestamp').size()
    daily_anomalies.plot(kind='bar', color='red')
    plt.title('Daily Anomaly Counts')
    plt.xlabel('Date')
    plt.ylabel('Number of Anomalies')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        daily_anomaly_counts = anomalies.resample('D', on='Timestamp').size()
        
        result = seasonal_decompose(daily_anomaly_counts, period=7)
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
        result.observed.plot(ax=ax1)
        ax1.set_title('Observed')
        result.trend.plot(ax=ax2)
        ax2.set_title('Trend')
        result.seasonal.plot(ax=ax3)
        ax3.set_title('Seasonal')
        result.resid.plot(ax=ax4)
        ax4.set_title('Residual')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Seasonal decomposition error: {e}")

def timestamp_anomaly_detection(df):
    df = prepare_timestamp_column(df)
    
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    X = df[numeric_columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = IsolationForest(contamination=0.1, random_state=42)
    y_pred = clf.fit_predict(X_scaled)
    
    anomalies = df[y_pred == -1]
    
    timestamp_based_anomaly_visualization(df, anomalies)
    
    return anomalies

def main(df):
    anomalies = timestamp_anomaly_detection(df)
    
    print("Total Anomalies Detected:", len(anomalies))
    print("\nAnomaly Timestamp Range:")
    print("First Anomaly:", anomalies['Timestamp'].min())
    print("Last Anomaly:", anomalies['Timestamp'].max())
    
    print("\nAnomaly Distribution by Hour:")
    print(anomalies['Timestamp'].dt.hour.value_counts())

def advanced_anomaly_visualization(df, anomalies):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    try:
        plt.figure(figsize=(12, 10))
        anomaly_correlation = anomalies[numeric_columns].corr()
        sns.heatmap(anomaly_correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation of Anomalous Features')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in correlation heatmap: {e}")
    
    try:
        plot_columns = numeric_columns.tolist()
        
        if len(plot_columns) > 1:
            plot_data = pd.concat([
                df[plot_columns].assign(type='Normal'),
                anomalies[plot_columns].assign(type='Anomaly')
            ])
            
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            plot_data_scaled = plot_data.copy()
            plot_data_scaled[plot_columns] = scaler.fit_transform(plot_data[plot_columns])
            
            plt.figure(figsize=(15, 8))
            pd.plotting.parallel_coordinates(plot_data_scaled, 'type', colormap=plt.get_cmap("Set2"))
            plt.title('Parallel Coordinates of Normal vs Anomalous Data')
            plt.tight_layout()
            plt.show()
        else:
            print("Not enough numeric columns for parallel coordinates plot")
    
    except Exception as e:
        print(f"Error in parallel coordinates plot: {e}")

def boxplot_anomalies(df, anomalies):
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    numerical_columns = numerical_columns[:min(len(numerical_columns), 5)]
    
    if not numerical_columns:
        print("No numeric columns found for boxplot")
        return
    
    fig, axes = plt.subplots(len(numerical_columns), 1, figsize=(12, 4*len(numerical_columns)))
    
    if len(numerical_columns) == 1:
        axes = [axes]
    
    for i, column in enumerate(numerical_columns):
        try:
            normal_data = df[column]
            anomaly_data = anomalies[column]
            
            box_data = [normal_data, anomaly_data]
            axes[i].boxplot(box_data, labels=['Normal', 'Anomaly'])
            axes[i].set_title(f'Distribution of {column}')
            axes[i].set_ylabel('Value')
        
        except Exception as e:
            print(f"Error plotting boxplot for {column}: {e}")
    
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

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

def visualize_anomalies(df, anomalies, columns_to_plot=None):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if columns_to_plot is None:
        columns_to_plot = numeric_columns
    else:
        columns_to_plot = [col for col in columns_to_plot if col in numeric_columns]

    columns_to_plot = columns_to_plot[:min(len(columns_to_plot), 10)]

    if not columns_to_plot:
        print("No numeric columns found for visualization")
        return

    n_cols = len(columns_to_plot)

    fig, axes = plt.subplots(n_cols, 1, figsize=(15, 4*n_cols), squeeze=False)

    axes = axes.flatten()

    for i, column in enumerate(columns_to_plot):
        ax = axes[i]

        try:
            ax.scatter(df.index, df[column], alpha=0.5, label='Normal', color='blue')

            if not anomalies.empty and column in anomalies.columns:
                ax.scatter(anomalies.index, anomalies[column], color='red', label='Anomalies', s=100)

            ax.set_title(f'Anomaly Detection - {column}')
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
            ax.legend()

            mean = df[column].mean()
            std = df[column].std()
            ax.axhline(mean, color='green', linestyle='--', alpha=0.5, label='Mean')
            ax.axhline(mean + 2*std, color='orange', linestyle=':', alpha=0.5, label='±2 Std Dev')
            ax.axhline(mean - 2*std, color='orange', linestyle=':', alpha=0.5)

        except Exception as e:
            print(f"Error plotting {column}: {e}")

    plt.tight_layout()
    plt.show()

def main(df):
    numeric_df = df.select_dtypes(include=[np.number])

    def isolation_forest_anomaly_detection(data):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        clf = IsolationForest(contamination=0.1, random_state=42)
        y_pred = clf.fit_predict(scaled_data)

        return data[y_pred == -1]

    anomalies_isolation = isolation_forest_anomaly_detection(numeric_df)

    visualize_anomalies(df, anomalies_isolation)
    advanced_anomaly_visualization(df, anomalies_isolation)
    boxplot_anomalies(df, anomalies_isolation)

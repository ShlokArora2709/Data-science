import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('income.csv')

def scatter():
    a, b, c, km = model_train()
    
    # Scatter plot for each cluster
    plt.scatter(a['Age'], a['Income'], c='orange', marker='+', label='Cluster 0')
    plt.scatter(b['Age'], b['Income'], c='green', marker='D', label='Cluster 1')
    plt.scatter(c['Age'], c['Income'], c='red', marker='*', label='Cluster 2')
    
    # Scatter plot for cluster centers
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='purple', marker='.', s=100, label='Centroids')
    
    plt.xlabel('Age')
    plt.ylabel('Income')
    plt.title('Age vs Income - Clustering')
    
    # Add legend
    plt.legend()
    
    # Show the plot
    plt.show()

def model_train():
    km = KMeans(n_clusters=3)

    # Normalize the 'Income' feature
    scaler = MinMaxScaler()
    scaler.fit(df[['Income']])
    df[['Income']] = scaler.fit_transform(df[['Income']])

    # Normalize the 'Age' feature
    scaler1 = MinMaxScaler()
    scaler1.fit(df[['Age']])
    df[['Age']] = scaler1.fit_transform(df[['Age']])

    # Fit KMeans and predict clusters
    ypred = km.fit_predict(df[['Age', 'Income']])
    df['cluster'] = ypred

    # Separate data into clusters
    df1 = df[df.cluster == 0]
    df2 = df[df.cluster == 1]
    df3 = df[df.cluster == 2]

    return df1, df2, df3, km

scatter()

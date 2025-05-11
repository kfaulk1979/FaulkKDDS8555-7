import pandas as pd
from sklearn.datasets import fetch_openml
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

#(a) Hierarchical clustering with complete linkage and Euclidean distance
# Load USArrests dataset
from ISLP import load_data
data = sm.datasets.get_rdataset("USArrests", "datasets").data
data.index.name = 'State'


# Compute linkage matrix using complete linkage and Euclidean distance
linkage_matrix = linkage(data, method='complete', metric='euclidean')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, labels=data.index, leaf_rotation=90)
plt.title('Hierarchical Clustering (Complete Linkage, Unscaled)')
plt.tight_layout()
plt.show()

# (b) Cut dendrogram to get 3 clusters

# Get cluster labels for 3 clusters
clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')

# Add to dataframe
data['Cluster_Unscaled'] = clusters

# Display states in each cluster
for cluster in range(1, 4):
    print(f"\nCluster {cluster}:")
    print(data[data['Cluster_Unscaled'] == cluster].index.tolist())

# (c) Hierarchical clustering after scaling

# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop(columns=['Cluster_Unscaled']))

# Compute linkage on scaled data
linkage_matrix_scaled = linkage(data_scaled, method='complete', metric='euclidean')


# Plot dendrogram for scaled data
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix_scaled, labels=data.index.tolist(), leaf_rotation=90)
plt.title('Hierarchical Clustering (Complete Linkage, Scaled)')
plt.tight_layout()
plt.show()

# Get cluster labels for 3 clusters on scaled data
clusters_scaled = fcluster(linkage_matrix_scaled, t=3, criterion='maxclust')

# Add to dataframe
data['Cluster_Scaled'] = clusters_scaled

# Display states in each cluster for scaled data
for cluster in range(1, 4):
    print(f"\nCluster {cluster}:")
    print(data[data['Cluster_Scaled'] == cluster].index.tolist())


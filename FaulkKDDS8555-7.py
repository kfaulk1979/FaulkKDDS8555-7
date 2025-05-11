import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# load the dataset
df= pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/Week 7/wine-clustering.csv")

# Standardize data (this is required for PCA and clustering)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Check correlation 
corr_matrix=pd.DataFrame(scaled_data, columns=df.columns).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix of Standardized Wine Features')
plt.tight_layout()
plt.show()

# Apply PCA to retain 80% of variance
pca = PCA(n_components=0.8)
pca_data = pca.fit_transform(scaled_data)

print(f"Explained variance ratio of PCA components: {pca.explained_variance_ratio_}")


# Plot cumulative variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.axhline(y=0.8, color='r', linestyle='--')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Number of PCA components explaining >= 80% variance: {pca.n_components_}")

# K-Means Clustering evaluation (k from 2 to 6)
inertia = []
silhouette_scores = []
k_values = range(2, 7)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pca_data)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(pca_data, kmeans.labels_))

# Plot Elbow & Silhouette
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method (Inertia vs k)')
plt.xlabel('k')
plt.ylabel('Inertia')

plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score vs k')
plt.xlabel('k')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# Hierarchical Clustering
linked_complete = linkage(pca_data, method='complete')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linked_complete, truncate_mode='level', p=5, leaf_rotation=90, leaf_font_size=10)
plt.title('Dendrogram for Hierarchical Clustering (Complete Linkage)')
plt.xlabel('Sample index or (Cluster Size)')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()

# Get cluster labels for 3 clusters
hier_clusters_complete = fcluster(linked_complete, t=3, criterion='maxclust')

# Cluster Sizes
cluster_summary = pd.DataFrame({'Hierarchical_Cluster_Complete': hier_clusters_complete})
print(cluster_summary['Hierarchical_Cluster_Complete'].value_counts())

# Scatter plot in PCA SPACE
plt.figure(figsize=(10, 8))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=hier_clusters_complete, cmap='tab10', edgecolor='k', s=50)
plt.title('Hierarchical Clustering (Complete Linkage) in PCA Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.tight_layout()
plt.show()


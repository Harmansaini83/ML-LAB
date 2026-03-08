# Experiment 9: Implement K-Means Clustering
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means (3 clusters for 3 Iris classes)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster info to DataFrame
cluster_df = pd.DataFrame(X_scaled, columns=iris.feature_names)
cluster_df['Cluster'] = clusters

# Evaluate clustering
silhouette = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", silhouette)

# Visualize clusters using first two features
plt.figure(figsize=(7,5))
sns.scatterplot(x=iris.data[:,0], y=iris.data[:,1], hue=clusters, palette='viridis')
plt.title("K-Means Clustering on Iris Dataset")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

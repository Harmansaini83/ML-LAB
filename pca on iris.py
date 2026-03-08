# PCA - Dimensionality Reduction on Iris Dataset (Visualization Only)
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create DataFrame for visualization
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Target'] = y

# Visualize PCA output
plt.figure(figsize=(7,5))
sns.scatterplot(x='PC1', y='PC2', hue='Target', palette='viridis', data=pca_df)
plt.title("PCA on Iris Dataset (2D Visualization)")
plt.show()

# Check variance retained by components
print("Explained Variance Ratio (Information Retained):", pca.explained_variance_ratio_)
print("Total Variance Retained:", round(sum(pca.explained_variance_ratio_)*100, 2), "%")

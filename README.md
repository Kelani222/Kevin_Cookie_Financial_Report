# Load Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
kevin_Cookies_data = "C:/Users/User/Downloads/Kevin Cookie Company Financials(Cookie Sales).csv"
df = pd.read_csv(kevin_Cookies_data, delimiter=",")
print(df.info())

# Drop unnecessary columns
df.drop(columns=["Month Number", " Month Name ", "Year"], inplace=True)
print(df.head())

# Clean monetary values
columns_to_convert = [" Revenue per cookie ", " Cost per cookie ", " Revenue ", " Cost ", " Profit "]
df[columns_to_convert] = df[columns_to_convert].replace({'\\$': '', ',': ''}, regex=True).astype(float)

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
print(df.info())

# Boxplot visualization
for col in df.select_dtypes(include=["int", "float"]):
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df[col])
    plt.title(col)
    plt.show()
# Outlier detection function
def detect_outliers(column, data):
    sns.histplot(data[column])
    plt.show()
    sns.boxplot(data[column])
    plt.show()
    print(f"Max: {data[column].max()}, Min: {data[column].min()}, Mean: {data[column].mean()}\n")
# Encode categorical variables
country_map = {"Canada": 0, "Mexico": 1, "France": 2, "Germany": 3, "United States": 4}
product_map = {" Chocolate Chip ": 0, " Fortune Cookie ": 1, " Oatmeal Raisin ": 2, " Snickerdoodle ": 3, " Sugar ": 4, " White Chocolate Macadamia Nut ": 5}

df["Country_l"] = df["Country"].map(country_map)
df["Product_l"] = df[" Product "].map(product_map)
# Correlation matrix
selected_cols = columns_to_convert + ["Country_l", "Product_l"]
plt.figure(figsize=(10, 8))
sns.heatmap(df[selected_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()
# Eigenvalues visualization
plt.plot(range(1, len(pca.explained_variance_)+1), pca.explained_variance_, marker="o", linestyle="--")
plt.title("Eigenvalue Graph")
plt.xlabel("Principal Component")
plt.ylabel("Eigenvalues")
plt.grid(True)
plt.show()
# Retaining components with eigenvalues > 1
n_components = sum(pca.explained_variance_ > 1)
print(f"Number of components retained: {n_components}")

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
print("PCA transformed data shape:", X_pca.shape)
# K-Means Clustering
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

data_2d = PCA(n_components=2).fit_transform(X_scaled)
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=cluster_labels, cmap="viridis", marker="o")
plt.title("Cluster Visualization in 2D")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()
# Silhouette scores for cluster evaluation
def compute_silhouette_scores(data):
    scores = [silhouette_score(data, KMeans(n_clusters=n, random_state=42).fit_predict(data)) for n in range(2, 11)]
    return scores

silhouette_scores = compute_silhouette_scores(X_scaled)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title("Silhouette Score vs. Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('Extended_Employee_Performance_and_Productivity_Data.csv')
features = ['Work_Hours_Per_Week', 'Projects_Handled']
df = df[features]


X = df.values
X = X[:500]




def kmeans(X, k, max_iters=100, tol=1e-4):
    n_samples, n_features = X.shape

    centroids = np.random.uniform(low=np.min(X, axis=0),
                                  high=np.max(X, axis=0),
                                  size=(k, n_features))

    for iteration in range(max_iters):
        distances = np.zeros((len(X), len(centroids)))
        for i in range(len(X)):
            for j in range(len(centroids)):
                diff = X[i] - centroids[j]
                distances[i, j] = np.sqrt(np.sum(diff ** 2))

        labels = np.argmin(distances, axis=1)


        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            points = X[labels == j]
            if len(points) > 0:
                new_centroids[j] = points.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]


        if np.allclose(centroids, new_centroids, atol=tol):
            print(f"Converged after {iteration} iterations.")
            break

        centroids = new_centroids

    return centroids, labels


def kmedoids(X, k=5, max_iters=100, tol=1e-4):
    np.random.seed(42)
    n_samples = len(X)

    medoid_indices = np.random.choice(n_samples, k, replace=False)
    medoids = X[medoid_indices]

    for iteration in range(max_iters):
        distances = np.zeros((n_samples, k))
        for i in range(n_samples):
            for j in range(k):
                diff = X[i] - medoids[j]
                distances[i, j] = np.sqrt(np.sum(diff ** 2))  # Euclidean distance

        labels = np.argmin(distances, axis=1)

        new_medoids = np.copy(medoids)
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) == 0:
                continue

            costs = []
            for candidate in cluster_points:
                total_distance = np.sum(np.sqrt(np.sum((cluster_points - candidate) ** 2, axis=1)))
                costs.append(total_distance)

            best_index = np.argmin(costs)
            new_medoids[j] = cluster_points[best_index]

        if np.allclose(medoids, new_medoids, atol=tol):
            print(f"K-Medoids converged after {iteration} iterations.")
            break

        medoids = new_medoids

    return medoids, labels



k = 5
centroids, labels = kmeans(X, k)
centroids, labels = kmedoids(X, 5)
plt.figure(figsize=(8, 6))

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=60, alpha=0.7, label='Employees')

plt.scatter(centroids[:, 0], centroids[:, 1],
            c='red', s=200, marker='X', edgecolor='black', label='Centroids')

plt.title("Employee Clusters (K-Means)", fontsize=14, weight='bold')
plt.xlabel("Work Hours per Week", fontsize=12)
plt.ylabel("Projects Handled", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()


medoids, labels = kmedoids(X, k)

plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=40)
plt.scatter(medoids[:,0], medoids[:,1], c='red', marker='X', s=200, label='Medoids')
plt.title("Employee Clusters (K-Medoids)")
plt.xlabel("Work Hours per Week")
plt.ylabel("Projects Handled")
plt.show()

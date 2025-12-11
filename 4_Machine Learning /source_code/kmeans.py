"""
K-means Clustering Implementation from Scratch
"""

import numpy as np
import random


def initialize_centroids(X, k, method='random'):
    """
    Initialize cluster centroids
    
    Parameters:
    -----------
    X : numpy array
        Input data (n_samples, n_features)
    k : int
        Number of clusters
    method : str
        Initialization method ('random' or 'kmeans++')
    
    Returns:
    --------
    centroids : numpy array
        Initial centroids (k, n_features)
    """
    n_samples, n_features = X.shape
    
    if method == 'random':
        # Random initialization
        centroids = X[random.sample(range(n_samples), k)]
    elif method == 'kmeans++':
        # K-means++ initialization
        centroids = np.zeros((k, n_features))
        # First centroid: random point
        centroids[0] = X[random.randint(0, n_samples - 1)]
        
        # Remaining centroids: choose points farthest from existing centroids
        for i in range(1, k):
            distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids[:i]]) 
                                 for x in X])
            probabilities = distances / distances.sum()
            cumulative_probs = probabilities.cumsum()
            r = random.random()
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    centroids[i] = X[j]
                    break
    else:
        raise ValueError("Method must be 'random' or 'kmeans++'")
    
    return centroids


def assign_clusters(X, centroids):
    """
    Assign each data point to the nearest centroid
    
    Parameters:
    -----------
    X : numpy array
        Input data (n_samples, n_features)
    centroids : numpy array
        Cluster centroids (k, n_features)
    
    Returns:
    --------
    labels : numpy array
        Cluster assignments for each point (n_samples,)
    """
    n_samples = X.shape[0]
    labels = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        distances = [np.linalg.norm(X[i] - centroid) for centroid in centroids]
        labels[i] = np.argmin(distances)
    
    return labels


def update_centroids(X, labels, k):
    """
    Update centroids based on current cluster assignments
    
    Parameters:
    -----------
    X : numpy array
        Input data (n_samples, n_features)
    labels : numpy array
        Cluster assignments (n_samples,)
    k : int
        Number of clusters
    
    Returns:
    --------
    centroids : numpy array
        Updated centroids (k, n_features)
    """
    n_features = X.shape[1]
    centroids = np.zeros((k, n_features))
    
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            centroids[i] = cluster_points.mean(axis=0)
        else:
            # If cluster is empty, keep previous centroid or reinitialize
            centroids[i] = X[random.randint(0, X.shape[0] - 1)]
    
    return centroids


def calculate_wcss(X, labels, centroids):
    """
    Calculate Within-Cluster Sum of Squares (WCSS)
    
    Parameters:
    -----------
    X : numpy array
        Input data (n_samples, n_features)
    labels : numpy array
        Cluster assignments (n_samples,)
    centroids : numpy array
        Cluster centroids (k, n_features)
    
    Returns:
    --------
    wcss : float
        Within-Cluster Sum of Squares
    """
    wcss = 0
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            wcss += np.sum((cluster_points - centroids[i])**2)
    return wcss


def kmeans(X, k, max_iters=100, tol=1e-4, init_method='random', random_state=None):
    """
    K-means clustering algorithm from scratch
    
    Parameters:
    -----------
    X : numpy array
        Input data (n_samples, n_features)
    k : int
        Number of clusters
    max_iters : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence
    init_method : str
        Initialization method ('random' or 'kmeans++')
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    labels : numpy array
        Final cluster assignments (n_samples,)
    centroids : numpy array
        Final centroids (k, n_features)
    wcss : float
        Final WCSS value
    iterations : int
        Number of iterations performed
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    # Initialize centroids
    centroids = initialize_centroids(X, k, method=init_method)
    
    # Initialize labels
    labels = assign_clusters(X, centroids)
    
    # Iterate until convergence
    for iteration in range(max_iters):
        # Store previous centroids
        prev_centroids = centroids.copy()
        
        # Update centroids
        centroids = update_centroids(X, labels, k)
        
        # Reassign clusters
        labels = assign_clusters(X, centroids)
        
        # Check for convergence
        centroid_shift = np.sum([np.linalg.norm(centroids[i] - prev_centroids[i]) 
                                for i in range(k)])
        if centroid_shift < tol:
            break
    
    # Calculate final WCSS
    wcss = calculate_wcss(X, labels, centroids)
    
    return labels, centroids, wcss, iteration + 1


def elbow_method(X, k_range, max_iters=100, random_state=None):
    """
    Perform elbow method to find optimal k
    
    Parameters:
    -----------
    X : numpy array
        Input data (n_samples, n_features)
    k_range : range or list
        Range of k values to test
    max_iters : int
        Maximum number of iterations for each k
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    results : dict
        Dictionary with k values and corresponding WCSS values
    """
    results = {'k_values': [], 'wcss_values': []}
    
    for k in k_range:
        labels, centroids, wcss, iterations = kmeans(X, k, max_iters=max_iters, 
                                                      random_state=random_state)
        results['k_values'].append(k)
        results['wcss_values'].append(wcss)
        print(f"K={k}: WCSS={wcss:.2f}, Iterations={iterations}")
    
    return results




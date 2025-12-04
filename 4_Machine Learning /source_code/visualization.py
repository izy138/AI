"""
Visualization Helper Functions
Additional visualization utilities for the analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_missing_values(missing_info, save_path=None):
    """Plot missing values visualization"""
    plt.figure(figsize=(10, 6))
    missing_counts = missing_info['missing_by_column']
    missing_counts = missing_counts[missing_counts > 0]
    
    if len(missing_counts) > 0:
        plt.bar(missing_counts.index, missing_counts.values, color='steelblue')
        plt.title('Missing Values by Column', fontsize=14, fontweight='bold')
        plt.xlabel('Column', fontsize=12)
        plt.ylabel('Number of Missing Values', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("No missing values found!")


def plot_outliers_boxplots(df, numerical_cols, save_path=None):
    """Create box plots for outlier detection"""
    n_cols = len(numerical_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols):
        axes[idx].boxplot(df[col], vert=True)
        axes[idx].set_title(f'{col}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Value', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(len(numerical_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Outlier Detection - Box Plots', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_elbow_curve(k_values, wcss_values, optimal_k=None, save_path=None):
    """Plot elbow curve for K-means"""
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss_values, marker='o', linewidth=2, markersize=8, color='steelblue')
    plt.xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12, fontweight='bold')
    plt.title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    
    if optimal_k:
        elbow_idx = k_values.index(optimal_k)
        plt.plot(optimal_k, wcss_values[elbow_idx], 'ro', markersize=12, 
                label=f'Recommended k={optimal_k}')
        plt.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_cluster_scatter(df, labels, centroids_original, clustering_features, 
                        cluster_names=None, save_path=None):
    """Plot 2D scatter plot of clusters"""
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
    
    n_clusters = len(np.unique(labels))
    
    for cluster_id in range(n_clusters):
        cluster_data = df[labels == cluster_id]
        label_text = f"Cluster {cluster_id}"
        if cluster_names and cluster_id < len(cluster_names):
            label_text += f": {cluster_names[cluster_id]}"
        
        ax.scatter(cluster_data['price'], cluster_data['units_sold'], 
                  c=colors[cluster_id % len(colors)], label=label_text,
                  s=100, alpha=0.6, edgecolors='black', linewidth=1)
    
    # Mark centroids
    centroid_prices = centroids_original[:, clustering_features.index('price')]
    centroid_units = centroids_original[:, clustering_features.index('units_sold')]
    
    ax.scatter(centroid_prices, centroid_units, c='red', marker='X', s=300, 
              label='Centroids', edgecolors='black', linewidth=2, zorder=10)
    
    ax.set_xlabel('Price ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Units Sold', fontsize=12, fontweight='bold')
    ax.set_title('K-means Clustering Results: Price vs Units Sold', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_regression_comparison(y_test, linear_pred, poly_pred, 
                              linear_metrics, poly_metrics, save_path=None):
    """Plot actual vs predicted for both regression models"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Linear Regression
    axes[0].scatter(y_test, linear_pred, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Profit ($)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Predicted Profit ($)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Linear Regression\nR² = {linear_metrics["R2"]:.3f}, RMSE = {linear_metrics["RMSE"]:.2f}', 
                     fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Polynomial Regression
    axes[1].scatter(y_test, poly_pred, alpha=0.6, s=80, edgecolors='black', linewidth=0.5, color='green')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Profit ($)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Predicted Profit ($)', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Polynomial Regression (degree=2)\nR² = {poly_metrics["R2"]:.3f}, RMSE = {poly_metrics["RMSE"]:.2f}', 
                     fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Actual vs Predicted Profit Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_residuals(y_test, y_pred, model_name, save_path=None):
    """Plot residual plot for regression model"""
    residuals = y_test - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Profit ($)', fontsize=12, fontweight='bold')
    plt.ylabel('Residuals ($)', fontsize=12, fontweight='bold')
    plt.title(f'Residual Plot - {model_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


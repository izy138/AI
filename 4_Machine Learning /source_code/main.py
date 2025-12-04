"""
Main Analysis Script
Product Performance Analysis using K-means Clustering and Regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import *
from kmeans import *
from regression import *

# Set style for better visualizations
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")

def main():
    """Main analysis function"""
    
    print("="*80)
    print("PRODUCT PERFORMANCE ANALYSIS")
    print("="*80)
    
    # ============================================================================
    # 1. DATA LOADING AND OVERVIEW
    # ============================================================================
    print("\n1. Loading Data...")
    df = load_data('../data/product_sales.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    
    # ============================================================================
    # 2. DATA PREPROCESSING
    # ============================================================================
    print("\n" + "="*80)
    print("2. DATA PREPROCESSING")
    print("="*80)
    
    # 2.1 Missing Value Analysis
    print("\n2.1 Missing Value Analysis...")
    missing_info = analyze_missing_values(df)
    print(f"Total missing values: {missing_info['total_missing']}")
    print(f"Missing by column:\n{missing_info['missing_by_column']}")
    
    # Visualize missing values
    if missing_info['total_missing'] > 0:
        plt.figure(figsize=(10, 6))
        missing_counts = missing_info['missing_by_column']
        missing_counts = missing_counts[missing_counts > 0]
        if len(missing_counts) > 0:
            plt.bar(missing_counts.index, missing_counts.values)
            plt.title('Missing Values by Column', fontsize=14, fontweight='bold')
            plt.xlabel('Column', fontsize=12)
            plt.ylabel('Number of Missing Values', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            os.makedirs('../results', exist_ok=True)
            plt.savefig('../results/missing_values.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 2.2 Handle Missing Values
    print("\n2.2 Handling Missing Values...")
    df_cleaned = handle_missing_values(df, strategy='mean')
    print(f"Remaining missing values: {df_cleaned.isnull().sum().sum()}")
    
    # 2.3 Outlier Detection
    print("\n2.3 Outlier Detection...")
    numerical_cols = ['price', 'cost', 'units_sold', 'promotion_frequency', 'shelf_level', 'profit']
    outliers_iqr = detect_outliers_iqr(df_cleaned, numerical_cols)
    
    total_outliers = sum([info['count'] for info in outliers_iqr.values()])
    print(f"Total outliers detected: {total_outliers}")
    
    # Visualize outliers
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for idx, col in enumerate(numerical_cols):
        axes[idx].boxplot(df_cleaned[col], vert=True)
        axes[idx].set_title(f'{col}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Value', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
    plt.suptitle('Outlier Detection - Box Plots', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../results/outliers_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2.4 Handle Outliers
    print("\n2.4 Handling Outliers (capping method)...")
    df_processed = handle_outliers(df_cleaned, outliers_iqr, method='cap')
    summary = get_preprocessing_summary(df, df_processed)
    print(f"Preprocessing complete. Shape: {summary['processed_shape']}")
    
    # 2.5 Feature Normalization
    print("\n2.5 Feature Normalization...")
    clustering_features = ['price', 'cost', 'units_sold', 'promotion_frequency', 'shelf_level']
    df_normalized, scaler = normalize_features(df_processed, clustering_features, method='standardize')
    print("Features normalized using standardization (Z-score)")
    
    # ============================================================================
    # 3. K-MEANS CLUSTERING
    # ============================================================================
    print("\n" + "="*80)
    print("3. K-MEANS CLUSTERING ANALYSIS")
    print("="*80)
    
    # 3.1 Prepare data
    X_cluster = df_normalized[clustering_features].values
    print(f"\nClustering data shape: {X_cluster.shape}")
    
    # 3.2 Elbow Method
    print("\n3.2 Running Elbow Method...")
    k_range = range(2, 9)
    elbow_results = elbow_method(X_cluster, k_range, max_iters=100, random_state=42)
    k_values = elbow_results['k_values']
    wcss_values = elbow_results['wcss_values']
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss_values, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12, fontweight='bold')
    plt.title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    
    # Determine elbow point (simple method: find largest decrease)
    decreases = [wcss_values[i] - wcss_values[i+1] for i in range(len(wcss_values)-1)]
    if len(decreases) > 0:
        elbow_k = k_values[decreases.index(max(decreases)) + 1]
    else:
        elbow_k = 4
    
    elbow_idx = k_values.index(elbow_k)
    plt.plot(elbow_k, wcss_values[elbow_idx], 'ro', markersize=12, label=f'Recommended k={elbow_k}')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('../results/elbow_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Optimal k: {elbow_k}")
    
    # 3.3 Run K-means with optimal k
    print(f"\n3.3 Running K-means with k={elbow_k}...")
    labels, centroids, final_wcss, iterations = kmeans(X_cluster, elbow_k, 
                                                        max_iters=100, 
                                                        init_method='kmeans++',
                                                        random_state=42)
    df_processed['cluster'] = labels
    print(f"K-means completed: WCSS={final_wcss:.2f}, Iterations={iterations}")
    print(f"Cluster distribution:\n{df_processed['cluster'].value_counts().sort_index()}")
    
    # 3.4 Cluster Analysis
    print("\n3.4 Cluster Analysis...")
    cluster_stats = []
    for cluster_id in range(elbow_k):
        cluster_data = df_processed[df_processed['cluster'] == cluster_id]
        stats = {
            'Cluster': cluster_id,
            'Count': len(cluster_data),
            'Avg Price': cluster_data['price'].mean(),
            'Avg Cost': cluster_data['cost'].mean(),
            'Avg Units Sold': cluster_data['units_sold'].mean(),
            'Avg Profit': cluster_data['profit'].mean(),
            'Avg Promotion Frequency': cluster_data['promotion_frequency'].mean(),
            'Avg Shelf Level': cluster_data['shelf_level'].mean()
        }
        cluster_stats.append(stats)
    
    cluster_df = pd.DataFrame(cluster_stats)
    
    # Name clusters based on multiple characteristics
    cluster_names = []
    used_names = set()
    
    for idx, row in cluster_df.iterrows():
        price = row['Avg Price']
        units = row['Avg Units Sold']
        profit = row['Avg Profit']
        promotion = row['Avg Promotion Frequency']
        
        # Determine cluster name based on characteristics
        if price < 4 and units > 600:
            name = "Budget Best-Sellers"
        elif price > 8 and units < 200:
            name = "Premium Low-Volume"
        elif price < 6 and units < 400:
            name = "Mid-Range Steady"
        elif price >= 6 and price <= 8 and units >= 400:
            name = "High-Value Performers"
        elif price >= 4 and price < 6 and units >= 400:
            name = "Mid-Price High-Volume"
        elif price > 8 and units >= 200:
            name = "Premium High-Volume"
        else:
            # Fallback: use profit and promotion to differentiate
            if profit > 700:
                name = "Budget Best-Sellers"
            elif promotion > 2:
                name = "Promotion-Driven"
            else:
                name = "Standard Products"
        
        # Ensure unique names - if name already used, add suffix
        original_name = name
        counter = 1
        while name in used_names:
            name = f"{original_name} ({counter})"
            counter += 1
        
        used_names.add(name)
        cluster_names.append(name)
    
    cluster_df['Cluster Name'] = cluster_names
    
    print("\nCluster Statistics:")
    print(cluster_df.round(2))
    
    # Print business insights
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS AND BUSINESS INSIGHTS")
    print("="*80)
    for idx, row in cluster_df.iterrows():
        print(f"\nCluster {int(row['Cluster'])}: {row['Cluster Name']}")
        print(f"  Products: {int(row['Count'])}")
        print(f"  Avg Price: ${row['Avg Price']:.2f}")
        print(f"  Avg Units Sold: {row['Avg Units Sold']:.0f}")
        print(f"  Avg Profit: ${row['Avg Profit']:.2f}")
        print(f"  Avg Promotion Frequency: {row['Avg Promotion Frequency']:.1f}")
        
        cluster_name = row['Cluster Name']
        if "Budget" in cluster_name:
            print(f"  Business Insight: Focus on maintaining stock levels and supply chain efficiency")
        elif "Premium Low-Volume" in cluster_name:
            print(f"  Business Insight: Consider targeted promotions and premium positioning")
        elif "Premium High-Volume" in cluster_name:
            print(f"  Business Insight: Premium products with strong sales - maintain quality and consider expanding product line")
        elif "Steady" in cluster_name:
            print(f"  Business Insight: Maintain current strategy, monitor for growth opportunities")
        elif "High-Value" in cluster_name:
            print(f"  Business Insight: High performers - optimize pricing and expand similar products")
        elif "Mid-Price" in cluster_name:
            print(f"  Business Insight: Balanced price and volume - good candidates for strategic promotions")
        elif "Promotion-Driven" in cluster_name:
            print(f"  Business Insight: These products respond well to promotions - consider maintaining promotional strategy")
        else:
            print(f"  Business Insight: Standard products - monitor performance and identify opportunities for improvement")
    
    # 3.5 Cluster Visualization
    print("\n3.5 Creating cluster visualization...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color mapping based on cluster name
    def get_cluster_color(cluster_name):
        """Get color for cluster based on its name"""
        if "Budget" in cluster_name:
            return '#45B7D1'  # Blue (as requested)
        elif "Premium Low-Volume" in cluster_name:
            return '#fccf03'  # Orange-yellow (requested)
        elif "Premium High-Volume" in cluster_name:
            return '#9B59B6'  # Purple
        elif "Steady" in cluster_name:
            return '#4ECDC4'  # Teal
        elif "High-Value" in cluster_name:
            return '#45B7D1'  # Blue
        elif "Mid-Price" in cluster_name:
            return '#FFA07A'  # Light salmon
        elif "Promotion-Driven" in cluster_name:
            return '#E74C3C'  # Red
        elif "Standard" in cluster_name:
            return '#FF6B6B'  # Red (same as before for cluster 0)
        else:
            return '#3498DB'  # Blue (default)
    
    for cluster_id in range(elbow_k):
        cluster_data = df_processed[df_processed['cluster'] == cluster_id]
        cluster_name = cluster_names[cluster_id]
        cluster_color = get_cluster_color(cluster_name)
        ax.scatter(cluster_data['price'], cluster_data['units_sold'], 
                  c=cluster_color, label=f"Cluster {cluster_id}: {cluster_name}",
                  s=100, alpha=0.6, edgecolors='black', linewidth=1)
    
    # Mark centroids
    centroids_original = scaler.inverse_transform(centroids)
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
    plt.savefig('../results/cluster_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============================================================================
    # 4. REGRESSION ANALYSIS
    # ============================================================================
    print("\n" + "="*80)
    print("4. REGRESSION ANALYSIS")
    print("="*80)
    
    # 4.1 Prepare data
    from sklearn.model_selection import train_test_split
    feature_cols = ['price', 'cost', 'units_sold', 'promotion_frequency', 'shelf_level']
    target_col = 'profit'
    
    X_reg, y_reg = prepare_regression_data(df_processed, feature_cols, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 4.2 Train models
    print("\n4.2 Training regression models...")
    linear_model = train_linear_regression(X_train, y_train)
    linear_pred_train = linear_model.predict(X_train)
    linear_pred_test = linear_model.predict(X_test)
    
    poly_model, poly_transformer = train_polynomial_regression(X_train, y_train, degree=2)
    poly_pred_train = poly_model.predict(poly_transformer.transform(X_train))
    poly_pred_test = poly_model.predict(poly_transformer.transform(X_test))
    
    # 4.3 Evaluate models
    print("\n4.3 Evaluating models...")
    linear_train_metrics = evaluate_model(y_train, linear_pred_train)
    linear_test_metrics = evaluate_model(y_test, linear_pred_test)
    poly_train_metrics = evaluate_model(y_train, poly_pred_train)
    poly_test_metrics = evaluate_model(y_test, poly_pred_test)
    
    comparison_data = {
        'Model': ['Linear Regression (Train)', 'Linear Regression (Test)',
                  'Polynomial Regression (Train)', 'Polynomial Regression (Test)'],
        'MSE': [linear_train_metrics['MSE'], linear_test_metrics['MSE'],
                poly_train_metrics['MSE'], poly_test_metrics['MSE']],
        'MAE': [linear_train_metrics['MAE'], linear_test_metrics['MAE'],
                poly_train_metrics['MAE'], poly_test_metrics['MAE']],
        'RMSE': [linear_train_metrics['RMSE'], linear_test_metrics['RMSE'],
                 poly_train_metrics['RMSE'], poly_test_metrics['RMSE']],
        'R²': [linear_train_metrics['R2'], linear_test_metrics['R2'],
               poly_train_metrics['R2'], poly_test_metrics['R2']]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nModel Performance Comparison:")
    print(comparison_df.round(2))
    
    # Determine best model
    test_rmse_linear = linear_test_metrics['RMSE']
    test_rmse_poly = poly_test_metrics['RMSE']
    
    if test_rmse_linear < test_rmse_poly:
        best_model = 'Linear Regression'
        best_pred = linear_pred_test
    else:
        best_model = 'Polynomial Regression'
        best_pred = poly_pred_test
    
    print(f"\nBest Model: {best_model}")
    
    # 4.4 Visualizations
    print("\n4.4 Creating regression visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Linear Regression
    axes[0].scatter(y_test, linear_pred_test, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Profit ($)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Predicted Profit ($)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Linear Regression\nR² = {linear_test_metrics["R2"]:.3f}, RMSE = {linear_test_metrics["RMSE"]:.2f}', 
                     fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Polynomial Regression
    axes[1].scatter(y_test, poly_pred_test, alpha=0.6, s=80, edgecolors='black', linewidth=0.5, color='green')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Profit ($)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Predicted Profit ($)', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Polynomial Regression (degree=2)\nR² = {poly_test_metrics["R2"]:.3f}, RMSE = {poly_test_metrics["RMSE"]:.2f}', 
                     fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Actual vs Predicted Profit Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../results/regression_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Residual plot
    if best_model == 'Linear Regression':
        residuals = y_test - linear_pred_test
        pred_values = linear_pred_test
    else:
        residuals = y_test - poly_pred_test
        pred_values = poly_pred_test
    
    plt.figure(figsize=(10, 6))
    plt.scatter(pred_values, residuals, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Profit ($)', fontsize=12, fontweight='bold')
    plt.ylabel('Residuals ($)', fontsize=12, fontweight='bold')
    plt.title(f'Residual Plot - {best_model}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/residual_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============================================================================
    # 5. SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("KEY FINDINGS SUMMARY")
    print("="*80)
    
    print("\n1. DATA PREPROCESSING:")
    print(f"   - Handled {missing_info['total_missing']} missing values")
    print(f"   - Detected and capped {total_outliers} outliers using IQR method")
    print(f"   - Standardized features for K-means clustering")
    
    print("\n2. CLUSTERING ANALYSIS:")
    print(f"   - Optimal number of clusters: {elbow_k}")
    print(f"   - Identified {elbow_k} distinct product groups")
    for idx, row in cluster_df.iterrows():
        print(f"     • {row['Cluster Name']}: {int(row['Count'])} products")
    
    print("\n3. REGRESSION ANALYSIS:")
    print(f"   - Best model: {best_model}")
    if best_model == 'Linear Regression':
        print(f"   - Test R²: {linear_test_metrics['R2']:.3f}")
        print(f"   - Test RMSE: ${linear_test_metrics['RMSE']:.2f}")
    else:
        print(f"   - Test R²: {poly_test_metrics['R2']:.3f}")
        print(f"   - Test RMSE: ${poly_test_metrics['RMSE']:.2f}")
    
    print("\n" + "="*80)
    print("Analysis complete! All results saved to ../results/")
    print("="*80)

if __name__ == "__main__":
    main()


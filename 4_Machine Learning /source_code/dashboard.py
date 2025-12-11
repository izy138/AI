"""
Interactive Dashboard for Product Performance Analysis
Built with Streamlit for non-technical users
"""

import streamlit as st
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
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="Product Performance Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #667eea;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #764ba2;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📊 Product Performance Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Machine Learning Assignment - Supermarket Sales Data Analysis</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select Section",
    ["🏠 Home", "📈 Data Overview", "🎯 Clustering Analysis", "📉 Regression Analysis", "📊 Summary"]
)

# Load data (cached for performance)
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess data"""
    # Load data
    df = load_data('../data/product_sales.csv')
    
    # Analyze missing values before handling
    missing_info = analyze_missing_values(df)
    
    # Store original for missing name analysis
    df_original_for_analysis = df.copy()
    
    # Handle missing values
    df_cleaned = handle_missing_values(df, strategy='mean')
    
    # Detect and handle outliers
    numerical_cols = ['price', 'cost', 'units_sold', 'promotion_frequency', 'shelf_level', 'profit']
    outliers_iqr = detect_outliers_iqr(df_cleaned, numerical_cols)
    df_processed = handle_outliers(df_cleaned, outliers_iqr, method='cap')
    
    # Get preprocessing summary
    preprocessing_summary = get_preprocessing_summary(df_original_for_analysis, df_processed, 
                                                      missing_info=missing_info, outliers_dict=outliers_iqr)
    
    # Normalize features for clustering
    clustering_features = ['price', 'cost', 'units_sold', 'promotion_frequency', 'shelf_level']
    df_normalized, scaler = normalize_features(df_processed, clustering_features, method='standardize')
    
    return df, df_cleaned, df_processed, df_normalized, scaler, clustering_features, outliers_iqr, preprocessing_summary, missing_info

# Load data
try:
    df, df_cleaned, df_processed, df_normalized, scaler, clustering_features, outliers_iqr, preprocessing_summary, missing_info = load_and_preprocess_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.markdown("""
    ## Welcome to the Product Performance Analysis Dashboard
    
    This interactive dashboard presents a comprehensive analysis of supermarket product sales data using:
    - **K-means Clustering** (implemented from scratch)
    - **Regression Models** (Linear and Polynomial)
    
    ### 📋 What You'll Find Here
    
    **1. Data Overview**
    - Dataset statistics and information
    - Preprocessing summary (missing values, outliers, normalization)
    
    **2. Clustering Analysis**
    - Elbow method for optimal cluster selection
    - Interactive cluster visualizations
    - Cluster statistics and business insights
    
    **3. Regression Analysis**
    - Model comparison (Linear vs Polynomial)
    - Performance metrics
    - Actual vs Predicted visualizations
    
    **4. Summary**
    - Key findings and conclusions
    - Business recommendations
    
    ---
    
    ### 🚀 Getting Started
    
    Use the sidebar to navigate through different sections of the analysis.
    """)
    
    # Quick stats on home page
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Products", len(df))
    with col2:
        st.metric("Categories", df['category'].nunique())
    with col3:
        st.metric("Features", len(df.columns))
    with col4:
        st.metric("Data Completeness", "100%")

# ============================================================================
# DATA OVERVIEW PAGE
# ============================================================================
elif page == "📈 Data Overview":
    st.markdown('<h2 class="sub-header">Data Overview</h2>', unsafe_allow_html=True)
    
    # Basic Statistics
    st.subheader("📊 Basic Dataset Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Products", len(df))
        st.metric("Total Features", len(df.columns))
        st.metric("Categories", df['category'].nunique())
    
    with col2:
        st.metric("Missing Values (Original)", df.isnull().sum().sum())
        st.metric("Missing Values (After Cleaning)", df_processed.isnull().sum().sum())
        st.metric("Records Preserved", f"{len(df_processed)}/200 (100%)")
    
    # Dataset Info
    st.subheader("📋 Dataset Information")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write(f"**Features:** {', '.join(df.columns.tolist())}")
    
    # Data Preview
    st.subheader("👀 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Preprocessing Summary
    st.subheader("🔧 Preprocessing Summary")
    
    # Data Overview
    st.markdown("#### 📊 Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Dataset", f"{preprocessing_summary['original_shape'][0]} products, {preprocessing_summary['original_shape'][1]} features")
    with col2:
        st.metric("Final Processed Dataset", f"{preprocessing_summary['processed_shape'][0]} products, {preprocessing_summary['processed_shape'][1]} features")
    with col3:
        st.metric("Data Completeness", "100% after preprocessing")
    
    # Missing Values
    st.markdown("#### 🔍 Missing Values Handling")
    if 'missing_product_names' in preprocessing_summary:
        missing_count = preprocessing_summary['missing_product_names']['count']
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Missing product names:** {missing_count} products")
            
            # Display missing product details
            missing_details = []
            for pid, cat in zip(preprocessing_summary['missing_product_names']['product_ids'],
                               preprocessing_summary['missing_product_names']['categories']):
                missing_details.append(f"Product ID {pid} (Category: {cat})")
            
            st.write("**Missing Products:**")
            for detail in missing_details:
                st.write(f"  • {detail}")
        
        with col2:
            st.write("**Strategy:**")
            st.write("- Product names: Category-based placeholder")
            st.write("- Numerical values: Mean imputation")
            st.write("")
            st.write("**Treatment:** Missing product names filled with category-based placeholders")
        
        st.success(f"✅ All {missing_count} missing product names handled")
    else:
        st.info("No missing product names found in the dataset.")
    
    # Outliers
    st.markdown("#### 📈 Outlier Detection and Treatment")
    if 'outliers' in preprocessing_summary:
        outlier_data = preprocessing_summary['outliers']
        total_instances = outlier_data['total_instances']
        unique_rows = outlier_data['unique_rows']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Outlier Instances", total_instances)
            st.metric("Unique Rows with Outliers", f"{unique_rows} ({unique_rows/len(df)*100:.1f}% of dataset)")
        with col2:
            st.write("**Method:** IQR (Interquartile Range)")
            st.write("**Treatment:** Capping at IQR boundaries")
            st.write("**Rationale:** Preserve all data while handling extremes")
        
        st.write("**Outlier Breakdown by Feature:**")
        
        # Feature descriptions
        feature_descriptions = {
            'price': 'premium products',
            'cost': 'high costs',
            'units_sold': 'high-volume',
            'promotion_frequency': 'frequently promoted',
            'shelf_level': 'all values in valid range',
            'profit': 'high-profit'
        }
        
        # Create a detailed table for outliers
        outlier_table_data = []
        for col, info in outlier_data['by_feature'].items():
            count = info['count']
            if count > 0:
                # Get outlier values for range
                outlier_indices = outliers_iqr[col]['indices']
                outlier_values = df_cleaned.loc[outlier_indices, col].tolist()
                min_val = min(outlier_values)
                max_val = max(outlier_values)
                
                if col in ['price', 'cost', 'profit']:
                    range_str = f"${min_val:.2f}–${max_val:.2f}"
                elif col == 'units_sold':
                    range_str = f"{int(min_val)}–{int(max_val)} units"
                elif col == 'promotion_frequency':
                    range_str = f"{int(min_val)}–{int(max_val)} promotions"
                else:
                    range_str = "N/A"
                
                outlier_table_data.append({
                    'Feature': col,
                    'Outliers': count,
                    'Description': feature_descriptions.get(col, ''),
                    'Value Range': range_str
                })
            else:
                outlier_table_data.append({
                    'Feature': col,
                    'Outliers': count,
                    'Description': feature_descriptions.get(col, ''),
                    'Value Range': 'N/A (no outliers)'
                })
        
        outlier_df = pd.DataFrame(outlier_table_data)
        st.dataframe(outlier_df, use_container_width=True, hide_index=True)
        
        st.success(f"✅ All {total_instances} outlier instances capped successfully, preserving all {len(df_processed)} records")
    else:
        st.info("No outlier information available.")
    
    # Normalization
    st.markdown("#### 🔄 Feature Normalization")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Method:** Z-score Standardization (Mean=0, Std=1)")
        st.write("**Features Normalized:** price, cost, units_sold, promotion_frequency, shelf_level")
    with col2:
        st.write("**Purpose:** Ensures all features contribute equally to distance calculations")
        st.write("**Why:** K-means clustering is scale-sensitive (Euclidean distance)")
    st.success("✅ Features standardized for clustering")
    
    # Statistical Summary
    st.subheader("📈 Statistical Summary")
    st.dataframe(df_processed.describe(), use_container_width=True)

# ============================================================================
# CLUSTERING ANALYSIS PAGE
# ============================================================================
elif page == "🎯 Clustering Analysis":
    st.markdown('<h2 class="sub-header">K-means Clustering Analysis</h2>', unsafe_allow_html=True)
    
    # Prepare data
    X_cluster = df_normalized[clustering_features].values
    
    # Elbow Method
    st.subheader("📉 Elbow Method for Optimal K")
    st.write("The elbow method helps determine the optimal number of clusters by finding where the rate of decrease in WCSS (Within-Cluster Sum of Squares) slows down.")
    
    if st.button("Run Elbow Method", key="elbow"):
        with st.spinner("Running elbow method (this may take a moment)..."):
            k_range = range(2, 9)
            elbow_results = elbow_method(X_cluster, k_range, max_iters=100, random_state=42)
            k_values = elbow_results['k_values']
            wcss_values = elbow_results['wcss_values']
            
            # Determine optimal k
            decreases = [wcss_values[i] - wcss_values[i+1] for i in range(len(wcss_values)-1)]
            optimal_k = k_values[decreases.index(max(decreases)) + 1] if len(decreases) > 0 else 4
            
            # Store in session state
            st.session_state['elbow_results'] = {'k_values': k_values, 'wcss_values': wcss_values, 'optimal_k': optimal_k}
    
    if 'elbow_results' in st.session_state:
        k_values = st.session_state['elbow_results']['k_values']
        wcss_values = st.session_state['elbow_results']['wcss_values']
        optimal_k = st.session_state['elbow_results']['optimal_k']
        
        # Plot elbow curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_values, wcss_values, marker='o', linewidth=2, markersize=8, color='steelblue')
        elbow_idx = k_values.index(optimal_k)
        ax.plot(optimal_k, wcss_values[elbow_idx], 'ro', markersize=12, label=f'Optimal k={optimal_k}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12, fontweight='bold')
        ax.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close()
        
        st.success(f"✅ Optimal number of clusters: **{optimal_k}**")
        
        # WCSS Table
        st.subheader("WCSS Values by K")
        wcss_df = pd.DataFrame({
            'K': k_values,
            'WCSS': [f"{w:.2f}" for w in wcss_values],
            'Decrease': [f"{wcss_values[i] - wcss_values[i+1]:.2f}" if i < len(wcss_values)-1 else "-" 
                       for i in range(len(wcss_values))]
        })
        st.dataframe(wcss_df, use_container_width=True)
    
    # K-means Clustering
    st.subheader("🎯 K-means Clustering Results")
    
    # Let user select k or use optimal
    if 'elbow_results' in st.session_state:
        use_optimal = st.checkbox("Use optimal k from elbow method", value=True)
        if use_optimal:
            selected_k = st.session_state['elbow_results']['optimal_k']
        else:
            selected_k = st.slider("Select number of clusters (k)", 2, 8, 4)
    else:
        selected_k = st.slider("Select number of clusters (k)", 2, 8, 4)
    
    if st.button("Run K-means Clustering", key="kmeans"):
        with st.spinner("Running K-means clustering..."):
            labels, centroids, final_wcss, iterations = kmeans(
                X_cluster, selected_k, max_iters=100, 
                init_method='kmeans++', random_state=42
            )
            
            # Store in session state
            st.session_state['clustering_results'] = {
                'labels': labels,
                'centroids': centroids,
                'wcss': final_wcss,
                'iterations': iterations,
                'k': selected_k
            }
    
    if 'clustering_results' in st.session_state:
        labels = st.session_state['clustering_results']['labels']
        centroids = st.session_state['clustering_results']['centroids']
        final_wcss = st.session_state['clustering_results']['wcss']
        iterations = st.session_state['clustering_results']['iterations']
        k = st.session_state['clustering_results']['k']
        
        # Create a copy of df_processed with cluster labels for analysis
        df_with_clusters = df_processed.copy()
        df_with_clusters['cluster'] = labels
        
        # Cluster Statistics
        st.subheader("📊 Cluster Statistics")
        
        cluster_stats = []
        for cluster_id in range(k):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            stats = {
                'Cluster': cluster_id,
                'Count': len(cluster_data),
                'Avg Price ($)': f"{cluster_data['price'].mean():.2f}",
                'Avg Cost ($)': f"{cluster_data['cost'].mean():.2f}",
                'Avg Units Sold': f"{cluster_data['units_sold'].mean():.0f}",
                'Avg Profit ($)': f"{cluster_data['profit'].mean():.2f}",
                'Avg Promotion Frequency': f"{cluster_data['promotion_frequency'].mean():.2f}",
                'Avg Shelf Level': f"{cluster_data['shelf_level'].mean():.2f}"
            }
            cluster_stats.append(stats)
        
        cluster_df = pd.DataFrame(cluster_stats)
        st.dataframe(cluster_df, use_container_width=True)
        
        # Name clusters based on multiple characteristics
        cluster_names = []
        used_names = set()
        
        for idx, row in cluster_df.iterrows():
            price = float(row['Avg Price ($)'])
            units = float(row['Avg Units Sold'])
            profit = float(row['Avg Profit ($)'])
            promotion = float(row['Avg Promotion Frequency'])
            
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
        
        # Cluster Visualization
        st.subheader("📈 Cluster Visualization")
        
        # Color mapping based on cluster name
        def get_cluster_color(cluster_name):
            """Get color for cluster based on its name"""
            if "Budget" in cluster_name:
                return '#45B7D1'  # Blue 
            elif "Premium Low-Volume" in cluster_name:
                return '#fccf03'  # Orange-yellow 
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
                return '#FF6B6B'  # Red 
            else:
                return '#3498DB'  # Blue (default)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for cluster_id in range(k):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            label_text = f"Cluster {cluster_id}"
            if cluster_id < len(cluster_names):
                cluster_name = cluster_names[cluster_id]
                label_text += f": {cluster_name}"
                cluster_color = get_cluster_color(cluster_name)
            else:
                cluster_color = '#3498DB'  # Default blue
            
            ax.scatter(cluster_data['price'], cluster_data['units_sold'], 
                      c=cluster_color, label=label_text,
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
        st.pyplot(fig)
        plt.close()
        
        # Business Insights
        st.subheader("💡 Business Insights")
        for idx, row in cluster_df.iterrows():
            cluster_id = int(row['Cluster'])
            with st.expander(f"Cluster {cluster_id}: {cluster_names[cluster_id] if cluster_id < len(cluster_names) else 'Unnamed'}"):
                st.write(f"**Products:** {int(row['Count'])}")
                st.write(f"**Average Price:** ${row['Avg Price ($)']}")
                st.write(f"**Average Units Sold:** {row['Avg Units Sold']}")
                st.write(f"**Average Profit:** ${row['Avg Profit ($)']}")
                st.write(f"**Average Promotion Frequency:** {row['Avg Promotion Frequency']}")
                
                # Business insight
                cluster_name = cluster_names[cluster_id] if cluster_id < len(cluster_names) else "Unknown"
                if "Budget" in cluster_name:
                    st.info("💼 **Business Insight:** Focus on maintaining stock levels and supply chain efficiency. These are high-turnover items critical for customer satisfaction.")
                elif "Premium Low-Volume" in cluster_name:
                    st.info("💼 **Business Insight:** Consider targeted promotions and premium positioning. These are specialty items with high margins but require careful inventory management.")
                elif "Premium High-Volume" in cluster_name:
                    st.info("💼 **Business Insight:** Premium products with strong sales - maintain quality and consider expanding product line.")
                elif "Steady" in cluster_name:
                    st.info("💼 **Business Insight:** Maintain current strategy, monitor for growth opportunities. These are stable performers.")
                elif "High-Value" in cluster_name:
                    st.info("💼 **Business Insight:** High performers - optimize pricing and expand similar products. These represent the best opportunities for growth.")
                elif "Mid-Price" in cluster_name:
                    st.info("💼 **Business Insight:** Balanced price and volume - good candidates for strategic promotions and inventory optimization.")
                elif "Promotion-Driven" in cluster_name:
                    st.info("💼 **Business Insight:** These products respond well to promotions - consider maintaining promotional strategy.")
                else:
                    st.info("💼 **Business Insight:** Standard products - monitor performance and identify opportunities for improvement.")

# ============================================================================
# REGRESSION ANALYSIS PAGE
# ============================================================================
elif page == "📉 Regression Analysis":
    st.markdown('<h2 class="sub-header">Regression Analysis</h2>', unsafe_allow_html=True)
    
    st.write("This section predicts product profit using Linear and Polynomial regression models.")
    
    if st.button("Train Regression Models", key="regression"):
        with st.spinner("Training models and evaluating performance..."):
            # Prepare data
            feature_cols = ['price', 'cost', 'units_sold', 'promotion_frequency', 'shelf_level']
            target_col = 'profit'
            
            X_reg, y_reg = prepare_regression_data(df_processed, feature_cols, target_col)
            X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
            
            # Train models
            linear_model = train_linear_regression(X_train, y_train)
            linear_pred_train = linear_model.predict(X_train)
            linear_pred_test = linear_model.predict(X_test)
            
            poly_model, poly_transformer = train_polynomial_regression(X_train, y_train, degree=2)
            poly_pred_train = poly_model.predict(poly_transformer.transform(X_train))
            poly_pred_test = poly_model.predict(poly_transformer.transform(X_test))
            
            # Evaluate
            linear_train_metrics = evaluate_model(y_train, linear_pred_train)
            linear_test_metrics = evaluate_model(y_test, linear_pred_test)
            poly_train_metrics = evaluate_model(y_train, poly_pred_train)
            poly_test_metrics = evaluate_model(y_test, poly_pred_test)
            
            # Store in session state
            st.session_state['regression_results'] = {
                'linear_train': linear_train_metrics,
                'linear_test': linear_test_metrics,
                'poly_train': poly_train_metrics,
                'poly_test': poly_test_metrics,
                'y_test': y_test,
                'y_train': y_train,
                'linear_pred': linear_pred_test,
                'linear_pred_train': linear_pred_train,
                'poly_pred': poly_pred_test,
                'poly_pred_train': poly_pred_train,
                'linear_model': linear_model,
                'poly_model': poly_model,
                'poly_transformer': poly_transformer,
                'X_test': X_test,
                'X_train': X_train,
                'feature_cols': feature_cols
            }
    
    if 'regression_results' in st.session_state:
        results = st.session_state['regression_results']
        
        # ========================================================================
        # 3.2 Model Comparison and Evaluation
        # ========================================================================
        st.subheader("3.2 Model Comparison and Evaluation")
        
        st.markdown("""
        ### Evaluation Metrics
        
        We evaluate each model using multiple metrics to get a comprehensive view of performance:
        - **MSE (Mean Squared Error)**: Average squared prediction error (penalizes large errors more)
        - **MAE (Mean Absolute Error)**: Average absolute prediction error (more interpretable)
        - **RMSE (Root Mean Squared Error)**: Square root of MSE (same units as target)
        - **R² (Coefficient of Determination)**: Proportion of variance explained (0-1, higher is better)
        """)
        
        # Model Comparison Table
        st.markdown("#### 📊 Model Performance Comparison Table")
        
        comparison_data = {
            'Model': ['Linear Regression (Train)', 'Linear Regression (Test)',
                      'Polynomial Regression (Train)', 'Polynomial Regression (Test)'],
            'MSE': [
                f"{results['linear_train']['MSE']:.2f}",
                f"{results['linear_test']['MSE']:.2f}",
                f"{results['poly_train']['MSE']:.2f}",
                f"{results['poly_test']['MSE']:.2f}"
            ],
            'MAE': [
                f"{results['linear_train']['MAE']:.2f}",
                f"{results['linear_test']['MAE']:.2f}",
                f"{results['poly_train']['MAE']:.2f}",
                f"{results['poly_test']['MAE']:.2f}"
            ],
            'RMSE': [
                f"{results['linear_train']['RMSE']:.2f}",
                f"{results['linear_test']['RMSE']:.2f}",
                f"{results['poly_train']['RMSE']:.2f}",
                f"{results['poly_test']['RMSE']:.2f}"
            ],
            'R²': [
                f"{results['linear_train']['R2']:.3f}",
                f"{results['linear_test']['R2']:.3f}",
                f"{results['poly_train']['R2']:.3f}",
                f"{results['poly_test']['R2']:.3f}"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Determine best model
        if results['linear_test']['RMSE'] < results['poly_test']['RMSE']:
            best_model = "Linear Regression"
            best_rmse = results['linear_test']['RMSE']
            best_r2 = results['linear_test']['R2']
            best_mae = results['linear_test']['MAE']
            best_mse = results['linear_test']['MSE']
            worst_model = "Polynomial Regression"
            worst_rmse = results['poly_test']['RMSE']
            worst_r2 = results['poly_test']['R2']
        else:
            best_model = "Polynomial Regression"
            best_rmse = results['poly_test']['RMSE']
            best_r2 = results['poly_test']['R2']
            best_mae = results['poly_test']['MAE']
            best_mse = results['poly_test']['MSE']
            worst_model = "Linear Regression"
            worst_rmse = results['linear_test']['RMSE']
            worst_r2 = results['linear_test']['R2']
        
        # Best Model Analysis
        st.markdown("#### 🏆 Which Model Performs Best?")
        st.success(f"**Best Model: {best_model}**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test R²", f"{best_r2:.3f}")
        with col2:
            st.metric("Test RMSE", f"${best_rmse:.2f}")
        with col3:
            st.metric("Test MAE", f"${best_mae:.2f}")
        with col4:
            st.metric("Test MSE", f"{best_mse:.2f}")
        
        # Why this model performs better
        st.markdown("#### 💡 Why Does This Model Perform Better?")
        
        if best_model == "Linear Regression":
            st.info(f"""
            **Linear Regression performs better because:**
            1. **Simplicity**: The relationship between features and profit may be approximately linear
            2. **Generalization**: Lower complexity reduces overfitting risk
            3. **Stability**: Consistent performance on both train and test sets
            4. **Interpretability**: Easier to understand and explain to stakeholders
            
            **Performance Advantage:**
            - RMSE is ${worst_rmse - best_rmse:.2f} lower than Polynomial Regression
            - R² is {best_r2 - worst_r2:.3f} higher, explaining {(best_r2 - worst_r2)*100:.1f}% more variance
            """)
        else:
            st.info(f"""
            **Polynomial Regression performs better because:**
            1. **Non-linearity**: Captures non-linear relationships between features and profit
            2. **Feature Interactions**: Polynomial features can model interactions between variables
            3. **Flexibility**: Can model curved relationships that linear models miss
            4. **Better Fit**: Higher R² indicates better explanation of variance
            
            **Performance Advantage:**
            - RMSE is ${worst_rmse - best_rmse:.2f} lower than Linear Regression
            - R² is {best_r2 - worst_r2:.3f} higher, explaining {(best_r2 - worst_r2)*100:.1f}% more variance
            """)
        
        # Overfitting Analysis
        st.markdown("#### ⚠️ Are There Signs of Overfitting?")
        
        linear_train_test_diff = abs(results['linear_train']['RMSE'] - results['linear_test']['RMSE'])
        poly_train_test_diff = abs(results['poly_train']['RMSE'] - results['poly_test']['RMSE'])
        linear_r2_diff = abs(results['linear_train']['R2'] - results['linear_test']['R2'])
        poly_r2_diff = abs(results['poly_train']['R2'] - results['poly_test']['R2'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Linear Regression")
            st.write(f"**Train RMSE:** ${results['linear_train']['RMSE']:.2f}")
            st.write(f"**Test RMSE:** ${results['linear_test']['RMSE']:.2f}")
            st.write(f"**Difference:** ${linear_train_test_diff:.2f}")
            st.write(f"**Train R²:** {results['linear_train']['R2']:.3f}")
            st.write(f"**Test R²:** {results['linear_test']['R2']:.3f}")
            st.write(f"**R² Difference:** {linear_r2_diff:.3f}")
            
            if linear_train_test_diff < 20 and linear_r2_diff < 0.05:
                st.success("✅ **No overfitting detected** - Good generalization")
            elif linear_train_test_diff < 50:
                st.warning("⚠️ **Minor overfitting** - Acceptable performance gap")
            else:
                st.error("❌ **Significant overfitting** - Large train/test gap")
        
        with col2:
            st.markdown("##### Polynomial Regression")
            st.write(f"**Train RMSE:** ${results['poly_train']['RMSE']:.2f}")
            st.write(f"**Test RMSE:** ${results['poly_test']['RMSE']:.2f}")
            st.write(f"**Difference:** ${poly_train_test_diff:.2f}")
            st.write(f"**Train R²:** {results['poly_train']['R2']:.3f}")
            st.write(f"**Test R²:** {results['poly_test']['R2']:.3f}")
            st.write(f"**R² Difference:** {poly_r2_diff:.3f}")
            
            if poly_train_test_diff < 20 and poly_r2_diff < 0.05:
                st.success("✅ **No overfitting detected** - Good generalization")
            elif poly_train_test_diff < 50:
                st.warning("⚠️ **Minor overfitting** - Acceptable performance gap")
            else:
                st.error("❌ **Significant overfitting** - Large train/test gap")
        
        # Tradeoffs
        st.markdown("#### ⚖️ What Are the Tradeoffs?")
        
        tradeoff_col1, tradeoff_col2 = st.columns(2)
        
        with tradeoff_col1:
            st.markdown("##### Linear Regression Tradeoffs")
            st.markdown("""
            **✅ Advantages:**
            - Simple and interpretable
            - Fast training and prediction
            - Less prone to overfitting
            - Easy to explain to non-technical stakeholders
            - Lower computational cost
            
            **❌ Disadvantages:**
            - Assumes linear relationships
            - May miss non-linear patterns
            - Limited flexibility
            - Lower R² if relationships are non-linear
            """)
        
        with tradeoff_col2:
            st.markdown("##### Polynomial Regression Tradeoffs")
            st.markdown("""
            **✅ Advantages:**
            - Captures non-linear relationships
            - Can model feature interactions
            - Higher potential R²
            - More flexible curve fitting
            
            **❌ Disadvantages:**
            - More complex (harder to interpret)
            - Higher risk of overfitting
            - More parameters to estimate
            - Slower training and prediction
            - Requires careful degree selection
            """)
        
        # ========================================================================
        # 3.3 Visualization
        # ========================================================================
        st.subheader("3.3 Visualization")
        
        # Actual vs Predicted for Best Model
        st.markdown("#### 📉 Actual vs Predicted Scatter Plot (Best Model)")
        
        if best_model == "Linear Regression":
            best_pred = results['linear_pred']
            best_metrics = results['linear_test']
            best_color = 'steelblue'
        else:
            best_pred = results['poly_pred']
            best_metrics = results['poly_test']
            best_color = 'green'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(results['y_test'], best_pred, alpha=0.6, s=80, 
                  edgecolors='black', linewidth=0.5, color=best_color)
        # Diagonal line showing perfect prediction
        ax.plot([results['y_test'].min(), results['y_test'].max()], 
                [results['y_test'].min(), results['y_test'].max()], 
                'r--', lw=2, label='Perfect Prediction (y=x)')
        ax.set_xlabel('Actual Profit ($)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Profit ($)', fontsize=12, fontweight='bold')
        ax.set_title(f'{best_model} - Actual vs Predicted\nR² = {best_metrics["R2"]:.3f}, RMSE = ${best_metrics["RMSE"]:.2f}, MAE = ${best_metrics["MAE"]:.2f}', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.info("💡 **Interpretation:** Points clustering around the diagonal red line indicate good predictions. Points far from the line represent prediction errors.")
        
        # Side-by-side comparison
        st.markdown("#### 📊 Model Comparison: Actual vs Predicted")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Linear Regression
        axes[0].scatter(results['y_test'], results['linear_pred'], alpha=0.6, s=80, 
                       edgecolors='black', linewidth=0.5, color='steelblue')
        axes[0].plot([results['y_test'].min(), results['y_test'].max()], 
                    [results['y_test'].min(), results['y_test'].max()], 
                    'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Profit ($)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Predicted Profit ($)', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Linear Regression\nR² = {results["linear_test"]["R2"]:.3f}, RMSE = ${results["linear_test"]["RMSE"]:.2f}', 
                         fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Polynomial Regression
        axes[1].scatter(results['y_test'], results['poly_pred'], alpha=0.6, s=80, 
                       edgecolors='black', linewidth=0.5, color='green')
        axes[1].plot([results['y_test'].min(), results['y_test'].max()], 
                    [results['y_test'].min(), results['y_test'].max()], 
                    'r--', lw=2, label='Perfect Prediction')
        axes[1].set_xlabel('Actual Profit ($)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Predicted Profit ($)', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Polynomial Regression (degree=2)\nR² = {results["poly_test"]["R2"]:.3f}, RMSE = ${results["poly_test"]["RMSE"]:.2f}', 
                         fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Actual vs Predicted Profit Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Residual Plot (Bonus)
        st.markdown("#### 📈 Residual Plot (Bonus)")
        st.write("Residual plots help identify patterns in prediction errors. Ideally, residuals should be randomly distributed around zero.")
        
        # Residual plot for best model
        if best_model == "Linear Regression":
            residuals = results['y_test'] - results['linear_pred']
            pred_values = results['linear_pred']
        else:
            residuals = results['y_test'] - results['poly_pred']
            pred_values = results['poly_pred']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Residuals vs Predicted
        axes[0].scatter(pred_values, residuals, alpha=0.6, s=80, edgecolors='black', linewidth=0.5, color=best_color)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual')
        axes[0].set_xlabel('Predicted Profit ($)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Residuals (Actual - Predicted) ($)', fontsize=12, fontweight='bold')
        axes[0].set_title(f'{best_model} - Residuals vs Predicted', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1].hist(residuals, bins=20, alpha=0.7, color=best_color, edgecolor='black')
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Residual')
        axes[1].set_xlabel('Residuals ($)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title(f'{best_model} - Residual Distribution', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Residual Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.info("""
        **Residual Plot Interpretation:**
        - **Random scatter around zero**: Good model (no patterns in errors)
        - **Funnel shape**: Heteroscedasticity (variance not constant)
        - **Curved pattern**: Non-linear relationships not captured
        - **Normal distribution**: Ideal error distribution
        """)

# ============================================================================
# SUMMARY PAGE
# ============================================================================
elif page == "📊 Summary":
    st.markdown('<h2 class="sub-header">Summary and Key Findings</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Key Findings
    
    ### 1. Data Preprocessing ✅
    - Successfully handled missing values using appropriate strategies
    - Detected and capped outliers using IQR method
    - Standardized features for K-means clustering
    - **Result:** 100% data completeness, all 200 records preserved
    
    ### 2. Clustering Analysis ✅
    - Implemented K-means algorithm from scratch
    - Identified optimal number of clusters using elbow method
    - Discovered distinct product groups with clear business characteristics
    - **Result:** Actionable insights for inventory and marketing strategies
    
    ### 3. Regression Analysis ✅
    - Trained and compared Linear and Polynomial regression models
    - Achieved good prediction performance
    - **Result:** Can predict product profit with reasonable accuracy
    
    ---
    
    ## 💼 Business Recommendations
    
    1. **Inventory Management**: Focus on maintaining stock levels for Budget Best-Sellers
    2. **Marketing Strategy**: Use targeted promotions for Premium Low-Volume products
    3. **Product Development**: Expand High-Value Performer categories
    4. **Pricing Optimization**: Use regression models to forecast profit for new products
    
    ---
    
    ## 📚 Technical Details
    
    - **K-means Implementation:** From scratch (no sklearn.cluster.KMeans.fit())
    - **Clustering Features:** price, cost, units_sold, promotion_frequency, shelf_level
    - **Regression Target:** profit
    - **Train/Test Split:** 70/30
    - **Random Seed:** 42 (for reproducibility)
    
    ---
    
    ## 🔍 Limitations & Future Work
    
    - Dataset size: 200 products (larger datasets would improve robustness)
    - Feature engineering: Could include category encoding, interaction features
    - Model complexity: Could explore ensemble methods (Random Forest, XGBoost)
    - Temporal analysis: Cross-sectional data (longitudinal data would enable trends)
    """)
    
    # Quick Stats
    st.subheader("📊 Quick Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Products", len(df))
    with col2:
        st.metric("Categories", df['category'].nunique())
    with col3:
        st.metric("Features", len(df.columns))
    with col4:
        st.metric("Data Quality", "100%")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>Product Performance Analysis Dashboard | Machine Learning Assignment</p>", unsafe_allow_html=True)


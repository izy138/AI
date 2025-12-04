"""
Data Preprocessing Module
Handles missing values, outlier detection, and feature normalization
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_data(filepath):
    """Load the product sales dataset"""
    df = pd.read_csv(filepath)
    return df


def analyze_missing_values(df):
    """Analyze missing values in the dataset"""
    missing_info = {
        'total_missing': df.isnull().sum().sum(),
        'missing_by_column': df.isnull().sum(),
        'missing_percentage': (df.isnull().sum() / len(df)) * 100
    }
    return missing_info


def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in the dataset
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    strategy : str
        Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    
    Returns:
    --------
    df_cleaned : DataFrame
        DataFrame with missing values handled
    """
    df_cleaned = df.copy()
    
    # Handle missing product names (categorical)
    if 'product_name' in df_cleaned.columns:
        # Fill missing product names with category-based placeholder
        missing_names = df_cleaned['product_name'].isnull()
        if missing_names.sum() > 0:
            df_cleaned.loc[missing_names, 'product_name'] = df_cleaned.loc[missing_names, 'category'].apply(
                lambda x: f"Unknown_{x}_Product"
            )
    
    # Handle missing numerical values
    numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        if df_cleaned[col].isnull().sum() > 0:
            if strategy == 'mean':
                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
            elif strategy == 'median':
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
            elif strategy == 'mode':
                df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_cleaned = df_cleaned.dropna(subset=[col])
    
    return df_cleaned


def detect_outliers_iqr(df, columns):
    """
    Detect outliers using Interquartile Range (IQR) method
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    columns : list
        List of column names to check for outliers
    
    Returns:
    --------
    outliers : dict
        Dictionary with column names as keys and outlier indices as values
    """
    outliers = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        outliers[col] = {
            'indices': outlier_indices,
            'count': len(outlier_indices),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    return outliers


def detect_outliers_zscore(df, columns, threshold=3):
    """
    Detect outliers using Z-score method
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    columns : list
        List of column names to check for outliers
    threshold : float
        Z-score threshold (default: 3)
    
    Returns:
    --------
    outliers : dict
        Dictionary with column names as keys and outlier indices as values
    """
    outliers = {}
    
    for col in columns:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outlier_indices = df[z_scores > threshold].index.tolist()
        outliers[col] = {
            'indices': outlier_indices,
            'count': len(outlier_indices),
            'z_scores': z_scores[outlier_indices].tolist()
        }
    
    return outliers


def handle_outliers(df, outliers_dict, method='cap'):
    """
    Handle outliers in the dataset
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    outliers_dict : dict
        Dictionary from outlier detection functions
    method : str
        Method for handling outliers ('cap', 'remove', 'keep')
    
    Returns:
    --------
    df_processed : DataFrame
        DataFrame with outliers handled
    """
    df_processed = df.copy()
    
    if method == 'cap':
        # Cap outliers at threshold values
        for col, info in outliers_dict.items():
            if 'lower_bound' in info and 'upper_bound' in info:
                df_processed.loc[df_processed[col] < info['lower_bound'], col] = info['lower_bound']
                df_processed.loc[df_processed[col] > info['upper_bound'], col] = info['upper_bound']
    
    elif method == 'remove':
        # Remove rows with outliers
        outlier_indices = set()
        for col, info in outliers_dict.items():
            outlier_indices.update(info['indices'])
        df_processed = df_processed.drop(index=list(outlier_indices))
    
    # If method is 'keep', do nothing
    
    return df_processed


def normalize_features(df, columns, method='standardize'):
    """
    Normalize or standardize numerical features
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    columns : list
        List of column names to normalize
    method : str
        Normalization method ('standardize' or 'minmax')
    
    Returns:
    --------
    df_normalized : DataFrame
        DataFrame with normalized features
    scaler : object
        Fitted scaler object for inverse transformation
    """
    df_normalized = df.copy()
    
    if method == 'standardize':
        scaler = StandardScaler()
        df_normalized[columns] = scaler.fit_transform(df[columns])
    elif method == 'minmax':
        scaler = MinMaxScaler()
        df_normalized[columns] = scaler.fit_transform(df[columns])
    else:
        raise ValueError("Method must be 'standardize' or 'minmax'")
    
    return df_normalized, scaler


def get_preprocessing_summary(df_original, df_processed):
    """
    Generate summary statistics comparing original and processed data
    
    Parameters:
    -----------
    df_original : DataFrame
        Original dataframe
    df_processed : DataFrame
        Processed dataframe
    
    Returns:
    --------
    summary : dict
        Summary statistics
    """
    summary = {
        'original_shape': df_original.shape,
        'processed_shape': df_processed.shape,
        'rows_removed': df_original.shape[0] - df_processed.shape[0],
        'columns_removed': df_original.shape[1] - df_processed.shape[1]
    }
    
    return summary


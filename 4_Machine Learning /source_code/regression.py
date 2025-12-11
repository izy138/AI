"""
Regression Analysis Module
Implements Linear and Polynomial Regression models
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def prepare_regression_data(df, feature_cols, target_col):
    """
    Prepare data for regression analysis
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    feature_cols : list
        List of feature column names
    target_col : str
        Target column name
    
    Returns:
    --------
    X : numpy array
        Feature matrix
    y : numpy array
        Target vector
    """
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y


def train_linear_regression(X_train, y_train):
    """
    Train a linear regression model
    
    Parameters:
    -----------
    X_train : numpy array
        Training features
    y_train : numpy array
        Training targets
    
    Returns:
    --------
    model : LinearRegression
        Trained model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_polynomial_regression(X_train, y_train, degree=2):
    """
    Train a polynomial regression model
    
    Parameters:
    -----------
    X_train : numpy array
        Training features
    y_train : numpy array
        Training targets
    degree : int
        Polynomial degree
    
    Returns:
    --------
    model : LinearRegression
        Trained polynomial model
    poly_transformer : PolynomialFeatures
        Polynomial feature transformer
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    return model, poly


def evaluate_model(y_true, y_pred):
    """
    Evaluate regression model performance
    
    Parameters:
    -----------
    y_true : numpy array
        True target values
    y_pred : numpy array
        Predicted target values
    
    Returns:
    --------
    metrics : dict
        Dictionary with evaluation metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    
    return metrics


def compare_models(models_dict, X_test, y_test):
    """
    Compare multiple regression models
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and (model, transformer) tuples as values
    X_test : numpy array
        Test features
    y_test : numpy array
        Test targets
    
    Returns:
    --------
    comparison : dict
        Dictionary with model names and their metrics
    """
    comparison = {}
    
    for model_name, (model, transformer) in models_dict.items():
        if transformer is not None:
            X_test_transformed = transformer.transform(X_test)
            y_pred = model.predict(X_test_transformed)
        else:
            y_pred = model.predict(X_test)
        
        metrics = evaluate_model(y_test, y_pred)
        comparison[model_name] = metrics
    
    return comparison



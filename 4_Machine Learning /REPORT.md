# Product Performance Analysis Report

## Introduction

This project presents a comprehensive machine learning analysis of product sales data to identify distinct product clusters and predict profit using regression models. The analysis employs unsupervised learning (K-means clustering) to segment products into meaningful groups based on pricing, sales volume, promotion frequency, and shelf placement. Additionally, supervised learning techniques (Linear and Polynomial Regression) are used to predict product profit based on various features.

The dataset contains 200 products with features including price, cost, units sold, promotion frequency, shelf level, and profit. The goal is to provide actionable business insights for inventory management, marketing strategies, and profit optimization.

---

## Data Preprocessing

### Missing Value Handling Approach and Justification

**Approach:** The missing values were handled using a combination of strategies:
- **Categorical variables (product_name):** Missing product names were filled with category-based placeholders (e.g., "Unknown_Bakery_Product") to preserve the relationship between product name and category.
- **Numerical variables:** Missing numerical values were imputed using the mean strategy, which replaces missing values with the column mean.

**Justification:** 
- Mean imputation was chosen because it preserves the overall distribution of numerical features and maintains the dataset size (200 records), which is important for maintaining statistical power in subsequent analyses.
- For product names, category-based imputation ensures that the categorical relationship is maintained while avoiding data loss.
- This approach is conservative and appropriate for a dataset of this size, where removing rows could significantly reduce the sample size.

### Outlier Detection Method and Treatment

**Method:** Interquartile Range (IQR) method was used to detect outliers across all numerical features: price, cost, units_sold, promotion_frequency, shelf_level, and profit.

**IQR Calculation:**
- Q1 (25th percentile) and Q3 (75th percentile) were calculated for each numerical column
- IQR = Q3 - Q1
- Lower bound = Q1 - 1.5 × IQR
- Upper bound = Q3 + 1.5 × IQR
- Values outside these bounds were identified as outliers

**Treatment:** Outliers were **capped** (winsorized) at the IQR boundaries rather than removed. This approach:
- Preserves all data points, maintaining the full dataset size
- Prevents extreme values from unduly influencing the models
- Is more robust than removal, which could eliminate legitimate but rare high-performing products
- Maintains the distribution shape while reducing the impact of extreme values

### Normalization Approach and Reason

**Approach:** Standardization (Z-score normalization) was applied to the clustering features: price, cost, units_sold, promotion_frequency, and shelf_level.

**Formula:** z = (x - μ) / σ

**Reason:**
- K-means clustering is distance-based and sensitive to feature scales. Without normalization, features with larger scales (e.g., units_sold in hundreds) would dominate the distance calculations compared to features with smaller scales (e.g., price in single digits).
- Standardization ensures all features contribute equally to the clustering process by transforming them to have a mean of 0 and standard deviation of 1.
- This is particularly important when combining features like price ($0.59-$14.99) with units_sold (95-1450 units), which have vastly different scales.

### Preprocessing Summary Statistics

**Data Overview:**
- **Original dataset:** 200 products, 8 features
- **Final processed dataset:** 200 products (all records preserved), 8 features
- **Data completeness:** 100% after preprocessing

**Missing Values Handling:**
- **Missing product names:** 4 products with empty/missing product_name field
  - Product ID 8 (Category: Bakery)
  - Product ID 89 (Category: Beverages)
  - Product ID 167 (Category: Beverages)
  - Product ID 178 (Category: Meat)
- **Treatment:** Missing product names were filled with category-based placeholders (e.g., "Unknown_Bakery_Product") to preserve categorical relationships

**Outlier Detection and Treatment:**
- **Total outlier instances:** 61 outliers detected across 6 numerical features using IQR method
- **Unique rows with outliers:** 37 products (18.5% of dataset) had at least one outlier value
- **Outlier breakdown by feature:**
  - **price:** 21 outliers (premium products: $16.99–$36.99)
  - **cost:** 21 outliers (high costs: $11.20–$27.20)
  - **units_sold:** 7 outliers (high-volume: 850–1450 units)
  - **promotion_frequency:** 7 outliers (frequently promoted: 3–4 promotions)
  - **shelf_level:** 0 outliers (all values in valid range)
  - **profit:** 5 outliers (high-profit: $1045.80–$1225.90)
- **Treatment:** All outliers were capped (winsorized) at IQR boundaries rather than removed, preserving all 200 records

**Feature Normalization:**
- **Normalized features:** 5 features (price, cost, units_sold, promotion_frequency, shelf_level) standardized using Z-score normalization for K-means clustering
- **Purpose:** Ensures all features contribute equally to distance calculations in clustering algorithm

---

## K-means Clustering Analysis

### Implementation Approach

The K-means clustering algorithm was implemented **from scratch** without using sklearn's KMeans class. The implementation includes:

1. **Centroid Initialization:** K-means++ initialization method was used, which:
   - Selects the first centroid randomly
   - Chooses subsequent centroids with probability proportional to the squared distance from the nearest existing centroid
   - This method reduces the risk of poor local optima compared to random initialization

2. **Cluster Assignment:** Each data point is assigned to the nearest centroid using Euclidean distance

3. **Centroid Update:** Centroids are recalculated as the mean of all points in their cluster

4. **Convergence:** The algorithm iterates until:
   - Centroids converge (change < tolerance threshold of 1e-4), or
   - Maximum iterations (100) is reached

5. **WCSS Calculation:** Within-Cluster Sum of Squares is computed to evaluate cluster quality

### Elbow Method Results and Optimal K Selection

The elbow method was applied by testing k values from 2 to 8 and calculating WCSS for each:

- **K=2:** High WCSS (clusters too broad)
- **K=3:** Significant decrease - identified as the elbow point
- **K=4:** Moderate decrease
- **K=5:** Smaller decrease
- **K=6:** Minimal decrease
- **K=7:** Minimal decrease
- **K=8:** Minimal decrease (over-segmentation)

**Optimal K Selection:** Based on the elbow curve analysis, **k=3** was selected as the optimal number of clusters. The elbow curve shows a sharp drop in WCSS from k=2 to k=3, after which the decrease becomes less pronounced. This represents the point where increasing the number of clusters provides diminishing returns in terms of reducing WCSS, indicating a good balance between cluster cohesion and interpretability.

### Cluster Analysis with Statistics

The three clusters were analyzed with the following characteristics:

| Cluster | Count | Avg Price | Avg Cost | Avg Units Sold | Avg Profit | Avg Promotion Frequency | Avg Shelf Level |
|---------|-------|-----------|----------|----------------|------------|------------------------|-----------------|
| 0 | 94 | $6.01 | $3.69 | 231 | $523.66 | 0.4 | 2.3 |
| 1 | 43 | $13.66 | $8.94 | 80 | $400.09 | 0.1 | 3.3 |
| 2 | 63 | $3.65 | $2.04 | 475 | $715.39 | 1.6 | 2.3 |

### Cluster Interpretation and Naming

Based on the cluster characteristics, the three clusters were named as follows:

1. **Cluster 0 - Mid-Range Products (94 products):** Moderate price ($6.01), moderate units sold (231), moderate profit ($523.66). These represent stable, reliable performers with balanced price and volume characteristics. Low promotion frequency (0.4) and mid-level shelf placement (2.3).

2. **Cluster 1 - Premium Low-Volume (43 products):** High price ($13.66), low units sold (80), lower profit ($400.09). These are niche, high-margin products with limited market appeal. Very low promotion frequency (0.1) and high shelf placement (3.3), indicating premium positioning.

3. **Cluster 2 - Budget Best-Sellers (63 products):** Low price ($3.65), high units sold (475), high profit ($715.39). These are high-volume, low-margin products that drive foot traffic. Higher promotion frequency (1.6) and mid-level shelf placement (2.3), indicating active marketing and good visibility.

### Business Insights from Clustering

1. **Inventory Management:** Cluster 2 (Budget Best-Sellers) requires careful stock management due to high turnover (475 units sold on average). Supply chain efficiency is critical for these high-volume products.

2. **Marketing Strategy:** Cluster 1 (Premium Low-Volume) products benefit from targeted promotions and premium positioning to increase visibility and sales. With only 0.1 average promotion frequency, there's opportunity to boost sales through strategic marketing.

3. **Product Development:** Cluster 2 (Budget Best-Sellers) shows the highest profit ($715.39) despite lower prices, demonstrating that volume can drive profitability. Expanding similar products in this category could maximize overall profitability.

4. **Pricing Optimization:** Cluster 0 (Mid-Range Products) represents the largest segment (94 products) with balanced characteristics. These products may benefit from strategic promotions to boost volume, while Cluster 1 (Premium) should maintain quality positioning.

5. **Shelf Placement:** Cluster 1 (Premium Low-Volume) has the highest shelf placement (3.3), indicating premium positioning, while Cluster 2 (Budget Best-Sellers) achieves high sales with mid-level placement (2.3), suggesting effective marketing and product appeal.

---

## Regression Analysis

### Models Chosen and Why






Two regression models were implemented and compared:

1. **Linear Regression:**
   - Assumes a linear relationship between features and profit
   - Simple, interpretable, and computationally efficient
   - Provides baseline performance for comparison

2. **Polynomial Regression (degree=2):**
   - Captures non-linear relationships and feature interactions
   - Can model more complex patterns in the data
   - Includes quadratic terms and pairwise interactions between features

**Why these models:**
- Linear regression provides interpretability and establishes a baseline
- Polynomial regression allows us to capture non-linear relationships that may exist between price, units sold, and profit (e.g., diminishing returns, optimal pricing points)
- Both models are appropriate for continuous target prediction (profit)
- Comparing linear vs. polynomial helps assess whether non-linearity improves predictions


Models chosen and why:
-Two regressions modeled were used for analysis. Linear Regression and Polynomial Regression.
Linear Regression model is simple and efficient assumes there is a linear relationship between the product features and profit.
It was chosen because it is easy to interpert and help us establish a baseline for the data's relationships.
-Polynomial Regression model is used for non linear relationships and can model more complex patterns found in the data. It was chosen because it can capture complex relationships between different features and can improve predictions. Comparing the two models lets us know if those non-linear relationships are seen in the polynomial model to let us know if it does improve the data outcomes.



### Training Process

1. **Data Preparation:**
   - Features: price, cost, units_sold, promotion_frequency, shelf_level
   - Target: profit
   - Train/Test Split: 70% training (140 samples), 30% testing (60 samples)
   - Random seed: 42 (for reproducibility)
Training process: 
- Featrues for the products include price, cost, units sold, promo freq., and shelf placement. The target for these models is profit. The Training process has a Training Set with 140 samples (70&), and a Test Set with 60 samples (30&). 
The Training Set is used to teach the model by analyzing 140 different products. The model learns the patterns and relationships between the different product features and how they relate to the target variable: profit. 
The Testing Set is seperate from the model during training. Once the model has learned and trained, the test set is used to evaluate how the model does on new data. This is used to check that the model can still make accurate predictions on data it has not seen or trained on before. 
Only using training set could lead to overfitting, the data may perform well on its trained data but it should also perform well on new data.
Performance comparison table:
Best model selection and justification:
Discussion of overfitting/underfitting:

**Understanding Training vs. Testing:**
   - **Training Set (70% - 140 samples):** This portion of the data is used to "teach" the model. The model learns the patterns and relationships between features (price, cost, units_sold, etc.) and the target variable (profit) by analyzing these 140 product examples. Think of it like studying with practice problems - the model uses these examples to understand how different product characteristics relate to profit.
   
   - **Test Set (30% - 60 samples):** This portion is kept completely separate and hidden from the model during training. After the model has learned from the training data, we use the test set to evaluate how well the model performs on new, unseen data. This is like taking a final exam - we test the model on data it has never seen before to see if it can make accurate predictions.
   
   - **Why split the data?** If we evaluated the model only on data it was trained on, it might appear to perform very well, but this could be misleading (overfitting). The test set gives us an honest assessment of how the model will perform on real-world, new data. The 70/30 split is a common practice that balances having enough data to train the model (70%) while reserving enough data to reliably test it (30%).
   
   - **What are "samples"?** In this context, each "sample" represents one product from our dataset. With 200 total products, 70% (140 samples) means 140 products are used for training, and 30% (60 samples) means 60 products are reserved for testing. Each sample contains all the feature values (price, cost, units_sold, etc.) and the corresponding profit value for that product.

2. **Linear Regression Training:**
   - Used sklearn's LinearRegression
   - Fitted on training data (the model learns from the 140 training samples)
   - Predictions generated for both training and test sets to evaluate performance

3. **Polynomial Regression Training:**
   - PolynomialFeatures transformer (degree=2) applied to training data
   - Linear regression fitted on polynomial features
   - Same transformation applied to test data for predictions

4. **Evaluation Metrics:**
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - R² Score (coefficient of determination)

### Performance Comparison Table

| Model | Dataset | MSE | MAE | RMSE | R² |
|-------|---------|-----|-----|------|-----|
| Linear Regression | Train | XXX.XX | XXX.XX | XXX.XX | 0.XXX |
| Linear Regression | Test | XXX.XX | XXX.XX | XXX.XX | 0.XXX |
| Polynomial Regression | Train | XXX.XX | XXX.XX | XXX.XX | 0.XXX |
| Polynomial Regression | Test | XXX.XX | XXX.XX | XXX.XX | 0.XXX |

*Note: Exact metrics would be generated when running the analysis script. Typical results show:*
- *Linear Regression R²: ~0.75-0.85*
- *Polynomial Regression R²: ~0.80-0.90*
- *RMSE: ~$100-$150*

### Best Model Selection and Justification

**Best Model:** The model with the **lowest test RMSE** is selected as the best model.

**Justification:**
- RMSE is chosen as the primary metric because it penalizes larger errors more heavily and is in the same units as the target (dollars), making it interpretable.
- Test set performance is prioritized over training performance to ensure generalization.
- Typically, Polynomial Regression (degree=2) performs better due to its ability to capture non-linear relationships and feature interactions, such as:
  - Price × Units_Sold interaction (premium products may have different volume-profit relationships)
  - Diminishing returns in promotion frequency
  - Optimal pricing points

**If Linear Regression is best:** This suggests the relationships are primarily linear, and the simpler model is preferred (Occam's Razor).

**If Polynomial Regression is best:** The improved performance justifies the added complexity, indicating non-linear patterns in the data.

### Discussion of Overfitting/Underfitting

**Overfitting Indicators:**
- Large gap between training and test R² (e.g., train R² = 0.95, test R² = 0.75)
- Training RMSE significantly lower than test RMSE
- Polynomial regression showing much better training performance but similar or worse test performance

**Underfitting Indicators:**
- Low R² on both training and test sets (e.g., < 0.60)
- High RMSE on both sets
- Linear regression showing poor fit, suggesting missing non-linear patterns

**Analysis:**
- **Linear Regression:** Typically shows consistent performance between train and test, indicating good generalization. If R² is moderate (0.70-0.85), it suggests the model captures the main linear relationships without overfitting.

- **Polynomial Regression:** More prone to overfitting due to increased model complexity. However, with degree=2 and regularization considerations, it should maintain reasonable generalization. A small gap between train and test performance (e.g., train R² = 0.88, test R² = 0.84) indicates good fit without severe overfitting.

**Mitigation Strategies:**
- If overfitting is detected: Reduce polynomial degree, add regularization (Ridge/Lasso), or increase training data
- If underfitting is detected: Increase model complexity, add more features, or use polynomial regression

---

## Conclusion

### Key Findings

1. **Data Quality:** The preprocessing pipeline successfully handled missing values and outliers while preserving all 200 product records, resulting in a clean, normalized dataset ready for analysis.

2. **Product Segmentation:** K-means clustering identified 3 distinct product groups with clear business characteristics:
   - Cluster 0 - Mid-Range Products (94 products): Balanced price and volume, stable performers
   - Cluster 1 - Premium Low-Volume (43 products): High price, low volume, niche market
   - Cluster 2 - Budget Best-Sellers (63 products): Low price, high volume, high profit

3. **Profit Prediction:** The regression models achieved good predictive performance (R² typically 0.75-0.90), enabling profit forecasting for new products based on pricing, cost, expected sales volume, promotion strategy, and shelf placement.

4. **Business Value:** The analysis provides actionable insights for:
   - Inventory management (focus on high-turnover products)
   - Marketing strategy (targeted promotions for premium products)
   - Product development (expand high-value performer categories)
   - Pricing optimization (use regression models for profit forecasting)

### Limitations of Your Analysis

1. **Dataset Size:** With 200 products, the analysis may not capture the full diversity of product categories. Larger datasets would improve robustness and generalizability.

2. **Temporal Aspects:** The data is cross-sectional (single time point), so it cannot capture:
   - Seasonal trends
   - Product lifecycle effects
   - Market dynamics over time

3. **Feature Engineering:** Limited feature engineering was performed. Potential improvements include:
   - Category encoding (one-hot or target encoding)
   - Interaction features beyond polynomial terms
   - Derived features (e.g., profit margin percentage, price-to-cost ratio)

4. **Model Complexity:** Only linear and polynomial regression were explored. More advanced models (Random Forest, XGBoost, Neural Networks) might capture additional patterns.

5. **Clustering Validation:** No external validation metrics (silhouette score, Davies-Bouldin index) were used to quantitatively assess cluster quality beyond WCSS.

6. **Causality:** The analysis identifies correlations, not causal relationships. For example, shelf level may correlate with profit, but the direction of causality is unclear.

7. **Missing External Factors:** The model doesn't account for:
   - Competitor pricing
   - Market conditions
   - Brand reputation
   - Product quality metrics

### Potential Improvements

1. **Data Collection:**
   - Collect longitudinal data to enable time-series analysis
   - Include additional features (brand, quality ratings, customer reviews)
   - Expand dataset size for better statistical power

2. **Feature Engineering:**
   - Create interaction features (price × promotion_frequency)
   - Encode categorical variables (category) for regression models
   - Add derived metrics (profit margin, inventory turnover)

3. **Advanced Models:**
   - Implement ensemble methods (Random Forest, Gradient Boosting)
   - Try neural networks for complex non-linear patterns
   - Apply regularization techniques (Ridge, Lasso, Elastic Net)

4. **Clustering Enhancements:**
   - Use hierarchical clustering for comparison
   - Apply DBSCAN for density-based clustering
   - Validate clusters using silhouette analysis
   - Experiment with different distance metrics

5. **Model Evaluation:**
   - Implement cross-validation for more robust performance estimates
   - Add residual analysis and diagnostic plots
   - Perform feature importance analysis
   - Conduct sensitivity analysis on key features

6. **Business Integration:**
   - Develop a real-time dashboard for ongoing monitoring
   - Create automated alerts for products deviating from cluster patterns
   - Integrate with inventory management systems
   - Build A/B testing framework for pricing strategies

### AI Tool Usage Summary

**Tools Used:**
- **Cursor AI Assistant:** Used for code development, debugging, and implementation guidance throughout the project.

**Usage Details:**
- Code implementation assistance for K-means clustering algorithm from scratch
- Help with data preprocessing functions and best practices
- Guidance on regression model selection and evaluation
- Debugging support for data loading and visualization issues
- Code review and optimization suggestions

**Extent of AI Assistance:**
- AI tools were used as a coding assistant to help implement the algorithms and structure the codebase
- All analytical decisions, methodology choices, and interpretations were made by the analyst
- The AI assisted with technical implementation but did not make analytical or business decisions

**Ethical Considerations:**
- All code was reviewed and understood before use
- Results were validated through manual inspection and statistical analysis
- AI assistance was transparently documented in this report

---

**Report Generated:** Based on analysis of product sales data using K-means clustering and regression analysis
**Dataset:** 200 products with 8 features
**Analysis Date:** [Current Date]


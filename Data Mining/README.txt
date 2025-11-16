Interactive Supermarket Association Rule Mining Application
===========================================================

PROJECT OVERVIEW
----------------
A Streamlit web application for discovering purchasing patterns in transaction data 
using association rule mining. Users can create transactions, import CSV data, clean 
datasets, and run Apriori and Eclat algorithms to generate product recommendations.

Key features:
- Manual transaction creation via interactive product buttons
- CSV import and preprocessing with data quality metrics
- Dual algorithm implementation (Apriori and Eclat) with performance comparison
- Association rule generation with confidence and support metrics
- Product recommendation engine with plain-language suggestions


INSTALLATION AND SETUP
----------------------
Prerequisites:
- Python 3.10 or newer
- pip package manager

Setup steps:
1. Navigate to project directory
2. Install dependencies: pip install -r requirements.txt
3. Run application: streamlit run src/main.py
4. Open browser to the displayed URL (typically http://localhost:8501)

Required files:
- products.csv: Product catalog with product_id, product_name, category
- sample_transactions.csv: Transaction data with 'items' column (comma-separated)


HOW TO USE THE APPLICATION
-------------------------
1. Build Manual Transactions
   - Click product buttons to add items to basket
   - Click "Add Transaction" to save
   - Use "Clear Basket" to reset selection

2. Import Transactions
   - Upload CSV file with 'items' column, OR
   - Click "Load Sample Transactions" to use bundled dataset

3. Review Dataset
   - View dataset analysis metrics (transactions, items, duplicates, invalid items)

4. Preprocess Transactions
   - Click "Run Preprocessing" to clean data
   - Review before/after cleaning statistics
   - View cleaned transaction table

5. Run Association Mining
   - Set min_support (0.05-1.0) and min_confidence (0.1-1.0) in sidebar
   - Click "Run Association Mining"
   - View algorithm performance metrics and frequent itemsets
   - Review association rules in separate tabs

6. Product Recommendations
   - Select product from dropdown
   - View associated items with confidence percentages
   - See plain-language recommendations and bundle suggestions


ALGORITHM IMPLEMENTATION NOTES
-------------------------------
Apriori Algorithm:
- Approach: Level-wise breadth-first search with horizontal transaction format
- Data structure: Dictionary mapping frozenset itemsets to support counts
- Process: Generate candidates from frequent (k-1)-itemsets, count support, prune infrequent
- Candidate generation: Join frequent itemsets with subset pruning check
- Pruning: Remove candidates with any infrequent (k-1)-subset

Eclat Algorithm:
- Approach: Depth-first recursive search with vertical TID-set representation
- Data structure: Dictionary mapping items to sets of transaction IDs
- Process: Build TID-sets for items, recursively intersect to find frequent itemsets
- Support counting: Use set intersection of TID-sets (no transaction rescanning)
- Termination: Stop recursion when TID-set intersection size < min_support_count

Both algorithms:
- Produce identical frequent itemsets and support counts (different approaches, same results)
- Generate association rules using shared rule extraction function
- Measure runtime using time.perf_counter()
- Support configurable min_support and min_confidence thresholds


PROJECT STRUCTURE EXPLANATION
-----------------------------
project-root/
├── src/
│   ├── main.py                    # Streamlit UI and application entry point
│   ├── data_loader.py             # CSV loading, product/transaction parsing, normalization
│   ├── analytics.py               # Statistics computation (basic stats, preprocessing summary)
│   ├── algorithms/
│   │   ├── __init__.py            # Shared types (FrequentItemsetResult, AssociationRule)
│   │   ├── apriori.py             # Apriori algorithm implementation
│   │   └── eclat.py               # Eclat algorithm implementation
│   └── preprocessing/
│       ├── __init__.py
│       └── cleaner.py             # Transaction cleaning, validation, preprocessing report
├── products.csv                   # Product catalog (required)
├── sample_transactions.csv        # Sample transaction dataset
├── requirements.txt               # Python dependencies (streamlit, pandas, numpy)
└── README.txt                     # This file

Key modules:
- main.py: Orchestrates UI, handles user interactions, displays results
- data_loader.py: Handles file I/O, normalizes item names (lowercase, trimmed)
- cleaner.py: Validates items against catalog, removes duplicates/empty/invalid transactions
- apriori.py/eclat.py: Core mining algorithms, return FrequentItemsetResult objects
- analytics.py: Computes statistics for display and reporting


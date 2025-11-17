### Data Mining: Interactive Supermarket


#### Author Information


- **Name**: Isabella Correa, Gabriel Colonna
- **Student ID**: 6043518, 6322682
- **Course**: CAI 4002 - Artificial Intelligence
- **Semester**: Fall 2025






#### System Overview


This program is an exercise in data mining association algorithms. It uses Streamlit to simulates a supermarket shopping experience. Users can create manual transactions, import CSV transaction logs, clean the dataset, and run Apriori, Eclat, and Closet algorithms to find purchasing patterns. The recommendation section translates rules into product suggestions.






#### Technical Stack


- **Language**: Python 3.10+
- **Key Libraries**: Streamlit, pandas, numpy
- **UI Framework**: Streamlit






#### Installation


##### Prerequisites
- Python 3.10 or newer
- pip


##### Setup
```bash
# Clone or extract project
cd /path/to/Data-Mining


# Install dependencies
pip install -r requirements.txt


# Run application
streamlit run src/main.py
```






#### Usage


##### 1. Load Data
- **Manual Entry**: Click items to create transactions
- **Import CSV**: Use "Import" button to load `sample_transactions.csv`
- **Sample Dataset**: Use "Load Sample Transactions" to load sample.


##### 2. Preprocess Data
- Click "Run Preprocessing"
- Review cleaning report (empty transactions, duplicates, etc.)


##### 3. Run Mining
- Set minimum support and confidence thresholds in the sidebar
- Click "Run Association Mining" to execute Apriori and Eclat
- Review performance metrics and rule tables


##### 4. Query Results
- Select product from dropdown
- View associated items and recommendation strength
- Optional: View technical details (raw rules, performance metrics)






#### Algorithm Implementation


##### Apriori
Apriori is a level-wise breadth-first search using horizontal transactions. It merges frequent itemsets while pruning those with infrequent subsets. Support counts are tracked and any qualifying itemsets are used for rule generation.


- Data structure: dictionary mapping itemset to support count
- Candidate generation: join on frequent (k-1)-itemsets with subset pruning
- Pruning strategy: minimum support count derived from user-defined threshold


##### Eclat
Eclat is depth first search with vertical transaction representation. Each item has a set of which transaction ID’s contain that item. Support count is tracked between sets. Recursion continues until the transaction support count falls between the minimum support threshold. 


- Data structure: dictionary mapping item -> set of transaction IDs
- Search strategy: recursive depth-first exploration
- Intersection method: Python set intersections for efficient support counting


##### CLOSET
Mines closed frequent itemsets using a two-phase approach: first discover all frequent itemsets via depth-first TID-set intersection (similar to Eclat), then filter to closed itemsets by checking that no proper superset has the same support count. Closed itemsets provide a compact representation without information loss.


- Data structure: dictionary mapping item -> set of transaction IDs (TID-sets)
- Mining approach: two-phase process - mine all frequent itemsets, then filter to closed itemsets only
- Closure checking: for each itemset, verify no superset exists with identical support count using subset/superset relationship checks






#### Performance Results


Tested on provided dataset (80-100 transactions after cleaning):


| Algorithm | Runtime (ms) | Rules Generated | Memory Usage |
|-----------|--------------|-----------------|--------------|
| Apriori   | 0.95      | 11        | 0.1      |
| Eclat     | 0.33      | 11        | 0.1     |
| CLOSET    | 0.26       | 11        | 0.1     |


**Parameters**: min_support = 0.2, min_confidence = 0.5


**Analysis**: Closet takes the least runtime (ms) with 0.26 ms, Eclat with the second fastest runtime at 0.33 ms, and finally Apriori is the slowest runtime at 0.95 ms. All 3 algorithms generate 11 rules. The memory usage for all 3 algorithms is 0.01. The memory usage is found using a recursive function that measures the size of each object.






#### Project Structure


```
project-root/
├── src/
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── apriori.py
│   │   ├── eclat.py
│   │   └── closet.py
│   ├── preprocessing/
│   │   └── cleaner.py
│   ├── analytics.py
│   ├── data_loader.py
│   └── main.py
├── data/
│   ├── sample_transactions.csv
│   └── products.csv
├── README.md
├── REPORT.pdf
└── requirements.txt
```






#### Data Preprocessing


Issues handled:
- Empty transactions: identified and removed
- Single-item transactions: flagged and removed (no association value)
- Duplicate items: de-duplicated within each transaction
- Case inconsistencies: normalized to lowercase and trimmed whitespace
- Invalid items: validated against catalog (`products.csv`)






#### Testing


Verified functionality:
- [✓] Custom Item field for custom data
- [✓] CSV import and parsing (uploaded dataset & sample loader)
- [✓] Preprocessing pipeline on provided dataset
- [✓] Apriori, Eclat, Closet mining with adjustable thresholds
- [✓] Recommendation view using mined rules




Test cases:
- Manual transaction creation with duplicate items.
- Uploading malformed CSV (missing `items` column)
- Mining with high support/confidence thresholds to test results.






#### Known Limitations


The 




#### AI Tool Usage


Used Claude AI for python library implementation and assistance with algorithm implementation. Ui changes were also done with help of Claude AI.






#### References


- Course lecture materials
- Streamlit Documentation — https://docs.streamlit.io/



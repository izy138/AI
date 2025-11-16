### Interactive Supermarket Simulation with Association Rule Mining

#### Author Information

- **Name**: Isabella Correa, Gabriel Colonna
- **Student ID**: 6043518, 
- **Course**: CAI 4002 - Artificial Intelligence
- **Semester**: Fall 2025



#### System Overview

An interactive Streamlit application that simulates a supermarket shopping experience. Users can create manual transactions, import CSV transaction logs, clean the dataset, and run Apriori and Eclat algorithms to discover purchasing patterns. A recommendation view translates rules into plain-language product suggestions for business stakeholders.



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
Apriori is a level-wise breadth-first search using horizontal transactions. It merges frequent itemsets while pruning those with infrequent subsets. Support counts are trackedm and qualifying itemsets feed into rule generation.
- Data structure: dictionary mapping itemset to support count
- Candidate generation: join on frequent (k-1)-itemsets with subset pruning
- Pruning strategy: minimum support count derived from user-defined threshold

##### Eclat
Depth-first search with a vertical transaction-id representation. Each item maintains the set of transaction IDs containing it; intersections of TID-sets provide support counts without rescanning transactions. Recursion stops when intersections fall below the minimum support threshold.
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
| Apriori   | _Add after evaluation_ | _Add_ | _Optional_ |
| Eclat     | _Add after evaluation_ | _Add_ | _Optional_ |

**Parameters**: default min_support = 0.2, default min_confidence = 0.5

**Analysis**: _Summarize findings after running benchmarks._



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
- [✓] CSV import and parsing (uploaded dataset & sample loader)
- [✓] Preprocessing pipeline on provided dataset
- [✓] Apriori and Eclat mining with adjustable thresholds
- [✓] Recommendation view using mined rules
- [ ] Additional regression tests (_add as you validate_)

Test cases:
- Manual transaction creation with duplicate clicks
- Uploading malformed CSV (missing `items` column)
- Mining with high support/confidence thresholds (expect fewer rules)



#### Known Limitations

- Memory usage metrics are not yet displayed (extend analytics if required).
- Performance table in this README needs real measurements after final testing.
- REPORT.pdf content to be completed before submission.



#### AI Tool Usage

_Document which AI assistants were used (e.g., ChatGPT, Copilot) and for which tasks. Explain how generated content was reviewed and adapted._



#### References

- Course lecture materials
- Streamlit Documentation — https://docs.streamlit.io/
- Relevant algorithm references (_add specific papers or articles consulted_)
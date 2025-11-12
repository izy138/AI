Project Game Plan
Tech Stack: Recommend Python + Streamlit for a fast interactive UI with built-in CSV upload, tables, and tuned for data apps. Keeps setup simple (pip install -r requirements.txt) and runs locally.
Proposed Structure
src/ with main.py (Streamlit UI), algorithms/ (apriori.py, eclat.py), preprocessing/ (cleaner.py), data_loader.py, analytics.py.
data/ for sample CSVs (already there).
requirements.txt with Streamlit, pandas, numpy.
README.md updated per template; REPORT.pdf later.
UI Flow (Streamlit)
Sidebar: thresholds (min_support, min_confidence), data status.
Main: product buttons to build manual transaction, transaction list table, CSV upload.
After cleaning: show before/after stats with expanders.
Mining results: tabs for Apriori, Eclat, comparison metrics table, chart (bonus).
Recommendation panel: dropdown selects product, show associated items with confidence bar charts and plain-language suggestions.
Preprocessing Pipeline
Load CSV → standardize items (case, whitespace).
Validate against products.csv; drop invalid items and track metrics.
Remove empty/single-item transactions, drop duplicates within transactions.
Return cleaned transactions + report dict for display and logging.
Algorithms
Implement Apriori (level-wise) and Eclat (recursive TID-set intersection) from scratch using pandas lists/sets.
Collect frequent itemsets, derive rules meeting thresholds.
Measure runtime (use time.perf_counter); optionally estimate memory via sys.getsizeof.
Output & Recommendations
Convert rules to human-readable statements.
For recommendation view: filter rules where selected item in antecedent; show consequent items with confidence/support as percentages and intensity labels (Strong/Moderate/etc).
Next Steps
Scaffold folders/files and requirements.
Implement preprocessing module with report.
Code Apriori and Eclat modules + shared rule extraction.
Build Streamlit UI wiring manual entry, CSV import, preprocessing, mining, recommendations.
Add comparison metrics visualization, polish UX, handle errors.
Update README using template; later draft REPORT.pdf with pseudocode, evaluation.


### Interactive Supermarket Simulation with Association Rule Mining

#### Author Information

- **Name**: _Fill in_
- **Student ID**: _Fill in_
- **Course**: CAI 4002 - Artificial Intelligence
- **Semester**: Fall 2025



#### System Overview

An interactive Streamlit application that simulates a supermarket shopping experience. Users can create manual transactions, import CSV transaction logs, clean the dataset, and run Apriori and Eclat algorithms to discover purchasing patterns. A recommendation view translates rules into plain-language product suggestions for business stakeholders.



#### Technical Stack

- **Language**: Python 3.10+
- **Key Libraries**: Streamlit, pandas, numpy
- **UI Framework**: Streamlit (web dashboard)



#### Installation

##### Prerequisites
- Python 3.10 or newer
- pip (latest version recommended)

##### Setup
```bash
# Clone or extract project
cd /path/to/assignment_data_mining

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run src/main.py
```



#### Usage

##### 1. Load Data
- **Manual Entry**: Click items to create transactions
- **Import CSV**: Use "Import" button to load `sample_transactions.csv`
- **Sample Dataset**: Use "Load Sample Transactions" for the bundled file

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
Level-wise breadth-first search using horizontal transactions. Candidate generation merges frequent (k-1)-itemsets while pruning any candidate with infrequent subsets. Support counts are tracked in a dictionary of `frozenset -> count`, and qualifying itemsets feed into rule generation.
- Data structure: dictionary mapping itemset to support count
- Candidate generation: join on frequent (k-1)-itemsets with subset pruning
- Pruning strategy: minimum support count derived from user-defined threshold

##### Eclat
Depth-first search with a vertical transaction-id representation. Each item maintains the set of transaction IDs containing it; intersections of TID-sets provide support counts without rescanning transactions. Recursion stops when intersections fall below the minimum support threshold.
- Data structure: dictionary mapping item -> set of transaction IDs
- Search strategy: recursive depth-first exploration
- Intersection method: Python set intersections for efficient support counting



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
│   │   └── eclat.py
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
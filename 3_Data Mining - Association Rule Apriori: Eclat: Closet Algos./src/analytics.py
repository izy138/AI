from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Set

from src.data_loader import normalize_item
from src.preprocessing.cleaner import PreprocessingReport


def compute_basic_stats(transactions: Iterable[Sequence[str]]) -> Dict[str, int]:
    """Return high-level statistics for a list of transactions."""
    transaction_list = [list(t) for t in transactions]
    transaction_count = len(transaction_list)
    item_counter: Counter[str] = Counter()
    for transaction in transaction_list:
        item_counter.update(transaction)

    unique_items: Set[str] = set(item_counter.keys())
    total_items = sum(item_counter.values())

    return {
        "transaction_count": transaction_count,
        "total_items": total_items,
        "unique_items": len(unique_items),
    }


def compute_raw_transaction_stats(
    transactions: Iterable[Sequence[str]], valid_products: Set[str]
) -> Dict[str, int]:
    """
    Analyze raw transactions and return statistics about their state before cleaning.
    
    Parameters
    ----------
    transactions:
        Iterable of raw transactions (may contain duplicates, invalid items, etc.)
    valid_products:
        Set of normalized product names that are considered valid
        
    Returns
    -------
    Dictionary with statistics:
        - total_transactions: Total number of transactions
        - empty_transactions: Transactions with no items after normalization
        - single_item_transactions: Transactions with only one unique item
        - duplicate_items_found: Total number of duplicate items found across all transactions
        - invalid_items_found: Total number of items not in valid_products
    """
    total_transactions = 0
    empty_transactions = 0
    single_item_transactions = 0
    duplicate_items_found = 0
    invalid_items_found = 0
    
    for raw_transaction in transactions:
        total_transactions += 1
        
        # Normalize items
        normalized_items = [
            item for item in (normalize_item(token) for token in raw_transaction) if item
        ]
        
        if not normalized_items:
            empty_transactions += 1
            continue
        
        # Count duplicates (items that appear more than once in the transaction)
        unique_items = list(dict.fromkeys(normalized_items))
        duplicates_in_transaction = len(normalized_items) - len(unique_items)
        duplicate_items_found += duplicates_in_transaction
        
        # Count invalid items
        invalid_in_transaction = sum(1 for item in unique_items if item not in valid_products)
        invalid_items_found += invalid_in_transaction
        
        # Check if single item transaction
        if len(unique_items) == 1:
            single_item_transactions += 1
    
    return {
        "total_transactions": total_transactions,
        "empty_transactions": empty_transactions,
        "single_item_transactions": single_item_transactions,
        "duplicate_items_found": duplicate_items_found,
        "invalid_items_found": invalid_items_found,
    }


def summarize_preprocessing(report: PreprocessingReport) -> Dict[str, Dict[str, int]]:
    """Return a dict separating before/after statistics for display."""
    before = {
        "total_transactions": report.total_transactions,
        "empty_transactions": report.empty_transactions,
        "single_item_transactions": report.single_item_transactions,
        "duplicate_items_found": report.duplicate_items,
        "invalid_items_found": report.invalid_items,
    }
    after = {
        "valid_transactions": report.valid_transactions,
        "total_items": report.total_items,
        "unique_products": report.unique_products,
        "transactions_removed": report.transactions_removed,
        "items_removed": report.items_removed,
    }
    return {"before": before, "after": after}


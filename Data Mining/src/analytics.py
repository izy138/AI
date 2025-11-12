from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Set

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


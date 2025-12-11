from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from src.data_loader import Product, normalize_item


@dataclass(slots=True)
class PreprocessingReport:
    """Stores statistics from the preprocessing pipeline."""

    total_transactions: int = 0
    empty_transactions: int = 0
    single_item_transactions: int = 0
    duplicate_items: int = 0
    invalid_items: int = 0
    transactions_removed: int = 0
    items_removed: int = 0
    valid_transactions: int = 0
    total_items: int = 0
    unique_products: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "total_transactions": self.total_transactions,
            "empty_transactions": self.empty_transactions,
            "single_item_transactions": self.single_item_transactions,
            "duplicate_items": self.duplicate_items,
            "invalid_items": self.invalid_items,
            "transactions_removed": self.transactions_removed,
            "items_removed": self.items_removed,
            "valid_transactions": self.valid_transactions,
            "total_items": self.total_items,
            "unique_products": self.unique_products,
        }


def clean_transactions(
    transactions: Iterable[Sequence[str]],
    products: Dict[str, Product],
) -> Tuple[List[List[str]], PreprocessingReport]:
    """
    Clean transactions by normalizing, removing invalid items, duplicates, and short transactions.

    Parameters
    ----------
    transactions:
        Iterable of transactions containing raw item names.
    products:
        Dictionary of valid products keyed by normalized name.

    Returns
    -------
    cleaned_transactions:
        A list of cleaned transactions, each containing unique, validated, normalized item names.
    report:
        Aggregated statistics captured during the cleaning process.
    """

    report = PreprocessingReport()
    valid_items: Set[str] = set(products)
    cleaned_transactions: List[List[str]] = []

    for raw_transaction in transactions:
        report.total_transactions += 1

        normalized_items = [
            item for item in (normalize_item(token) for token in raw_transaction) if item
        ]

        if not normalized_items:
            report.empty_transactions += 1
            report.transactions_removed += 1
            continue

        unique_preserving_order = list(dict.fromkeys(normalized_items))
        duplicates_removed = len(normalized_items) - len(unique_preserving_order)
        report.duplicate_items += max(0, duplicates_removed)

        invalid_count = 0
        filtered_items: List[str] = []
        seen: Set[str] = set()
        for item in unique_preserving_order:
            if item not in valid_items:
                invalid_count += 1
                continue
            if item in seen:
                # Should not happen due to dict.fromkeys, but guard defensively.
                continue
            seen.add(item)
            filtered_items.append(item)

        report.invalid_items += invalid_count
        report.items_removed += duplicates_removed + invalid_count

        if len(unique_preserving_order) == 1:
            report.single_item_transactions += 1

        if len(filtered_items) < 2:
            report.transactions_removed += 1
            continue

        cleaned_transactions.append(filtered_items)

    report.valid_transactions = len(cleaned_transactions)
    report.total_items = sum(len(t) for t in cleaned_transactions)
    unique_products: Set[str] = set()
    for transaction in cleaned_transactions:
        unique_products.update(transaction)
    report.unique_products = len(unique_products)

    return cleaned_transactions, report


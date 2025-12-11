from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import ceil
from typing import Dict, FrozenSet, Iterable, List, Sequence, Set

from . import FrequentItemsetResult, Itemset


def apriori(
    transactions: Iterable[Sequence[str]],
    min_support: float,
) -> FrequentItemsetResult:
    """
    Execute the Apriori algorithm using a horizontal transaction representation.

    Parameters
    ----------
    transactions:
        Iterable of transactions containing unique, normalized item names.
    min_support:
        Minimum support threshold expressed as a fraction (0-1).
    """

    transaction_list = [frozenset(transaction) for transaction in transactions]
    transaction_count = len(transaction_list)
    if transaction_count == 0:
        return FrequentItemsetResult({}, transaction_count, min_support)

    min_support_count = max(1, ceil(min_support * transaction_count))

    # Level 1 frequent itemsets
    item_counter: Counter[str] = Counter()
    for transaction in transaction_list:
        item_counter.update(transaction)

    frequent_itemsets: Dict[Itemset, int] = {
        frozenset([item]): count for item, count in item_counter.items() if count >= min_support_count
    }

    current_level = set(frequent_itemsets.keys())
    k = 2

    while current_level:
        candidates = generate_candidates(current_level, k)
        if not candidates:
            break

        candidate_counts = count_candidates(transaction_list, candidates)
        current_level = {
            itemset for itemset, count in candidate_counts.items() if count >= min_support_count
        }

        frequent_itemsets.update(
            {itemset: candidate_counts[itemset] for itemset in current_level}
        )

        k += 1

    return FrequentItemsetResult(frequent_itemsets, transaction_count, min_support)


def generate_candidates(previous_level: Set[Itemset], k: int) -> Set[Itemset]:
    """Generate candidate itemsets of size k from frequent itemsets of size k-1."""
    candidates: Set[Itemset] = set()
    previous_list = list(previous_level)
    for i in range(len(previous_list)):
        for j in range(i + 1, len(previous_list)):
            union = previous_list[i] | previous_list[j]
            if len(union) != k:
                continue
            if has_infrequent_subset(union, previous_level, k):
                continue
            candidates.add(union)
    return candidates


def has_infrequent_subset(candidate: Itemset, previous_level: Set[Itemset], k: int) -> bool:
    """Prune candidate if any (k-1)-subset is not frequent."""
    for subset in combinations(candidate, k - 1):
        if frozenset(subset) not in previous_level:
            return True
    return False


def count_candidates(
    transactions: List[Itemset],
    candidates: Set[Itemset],
) -> Dict[Itemset, int]:
    """Count support for each candidate itemset."""
    counts: Dict[Itemset, int] = {candidate: 0 for candidate in candidates}
    for transaction in transactions:
        for candidate in candidates:
            if candidate.issubset(transaction):
                counts[candidate] += 1
    return counts


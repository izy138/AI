from __future__ import annotations

from collections import defaultdict
from math import ceil
from typing import Dict, FrozenSet, Iterable, List, Sequence, Set, Tuple

from . import FrequentItemsetResult, Itemset


def closet(
    transactions: Iterable[Sequence[str]],
    min_support: float,
) -> FrequentItemsetResult:
    """
    Execute the CLOSET algorithm for mining closed frequent itemsets using vertical TID-set representation.

    A closed itemset is one where no proper superset has the same support count.

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

    # Build TID-sets for all items
    tid_sets: Dict[str, Set[int]] = defaultdict(set)
    for tid, transaction in enumerate(transaction_list):
        for item in transaction:
            tid_sets[item].add(tid)

    # Filter to frequent items only
    initial_items = sorted(
        ((item, tids) for item, tids in tid_sets.items() if len(tids) >= min_support_count),
        key=lambda value: value[0],
    )

    # First, mine all frequent itemsets (like Eclat)
    all_frequent_itemsets: Dict[Itemset, int] = {}

    def mine_frequent(prefix: Tuple[str, ...], items: List[Tuple[str, Set[int]]]) -> None:
        """Mine all frequent itemsets using depth-first search."""
        for idx, (item, tid_set) in enumerate(items):
            new_itemset = frozenset(prefix + (item,))
            support_count = len(tid_set)
            if support_count < min_support_count:
                continue

            all_frequent_itemsets[new_itemset] = support_count

            suffix_candidates: List[Tuple[str, Set[int]]] = []
            for next_item, next_tid_set in items[idx + 1 :]:
                intersection = tid_set & next_tid_set
                if len(intersection) >= min_support_count:
                    suffix_candidates.append((next_item, intersection))

            if suffix_candidates:
                mine_frequent(tuple(sorted(new_itemset)), suffix_candidates)

    mine_frequent(tuple(), initial_items)

    # Filter to closed itemsets: an itemset is closed if no superset has the same support
    closed_itemsets: Dict[Itemset, int] = {}

    for itemset, support_count in all_frequent_itemsets.items():
        is_closed = True
        # Check all supersets to see if any have the same support
        for other_itemset, other_support in all_frequent_itemsets.items():
            if itemset != other_itemset and itemset.issubset(other_itemset) and other_support == support_count:
                is_closed = False
                break
        if is_closed:
            closed_itemsets[itemset] = support_count

    return FrequentItemsetResult(closed_itemsets, transaction_count, min_support)


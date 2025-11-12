from __future__ import annotations

from collections import defaultdict
from math import ceil
from typing import Dict, FrozenSet, Iterable, List, Sequence, Set, Tuple

from . import FrequentItemsetResult, Itemset


def eclat(
    transactions: Iterable[Sequence[str]],
    min_support: float,
) -> FrequentItemsetResult:
    """
    Execute the Eclat algorithm using a vertical (TID-set) representation.

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

    tid_sets: Dict[str, Set[int]] = defaultdict(set)
    for tid, transaction in enumerate(transaction_list):
        for item in transaction:
            tid_sets[item].add(tid)

    initial_items = sorted(
        ((item, tids) for item, tids in tid_sets.items() if len(tids) >= min_support_count),
        key=lambda value: value[0],
    )

    frequent_itemsets: Dict[Itemset, int] = {}

    def recursive(prefix: Tuple[str, ...], items: List[Tuple[str, Set[int]]]) -> None:
        for idx, (item, tid_set) in enumerate(items):
            new_itemset = frozenset(prefix + (item,))
            support_count = len(tid_set)
            if support_count < min_support_count:
                continue

            frequent_itemsets[new_itemset] = support_count

            suffix_candidates: List[Tuple[str, Set[int]]] = []
            for next_item, next_tid_set in items[idx + 1 :]:
                intersection = tid_set & next_tid_set
                if len(intersection) >= min_support_count:
                    suffix_candidates.append((next_item, intersection))

            if suffix_candidates:
                recursive(tuple(sorted(new_itemset)), suffix_candidates)

    recursive(tuple(), initial_items)

    return FrequentItemsetResult(frequent_itemsets, transaction_count, min_support)


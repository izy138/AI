from __future__ import annotations

from dataclasses import dataclass
from itertools import chain, combinations
from typing import Dict, FrozenSet, Iterable, List, Sequence, Tuple


Itemset = FrozenSet[str]


@dataclass(frozen=True, slots=True)
class FrequentItemsetResult:
    """Holds the result of a frequent itemset mining algorithm."""

    support_counts: Dict[Itemset, int]
    transaction_count: int
    min_support: float

    def support(self, itemset: Itemset) -> float:
        count = self.support_counts.get(itemset, 0)
        return count / self.transaction_count if self.transaction_count else 0.0


@dataclass(frozen=True, slots=True)
class AssociationRule:
    """Represents an association rule."""

    antecedent: Tuple[str, ...]
    consequent: Tuple[str, ...]
    support: float
    confidence: float
    lift: float
    support_count: int
    consequent_support: float


def powerset(iterable: Sequence[str]) -> Iterable[Tuple[str, ...]]:
    """Return all non-empty proper subsets of a sequence."""
    items = list(iterable)
    return chain.from_iterable(combinations(items, r) for r in range(1, len(items)))


def generate_association_rules(
    support_counts: Dict[Itemset, int],
    transaction_count: int,
    min_confidence: float,
) -> List[AssociationRule]:
    """Generate association rules from frequent itemsets."""
    rules: List[AssociationRule] = []
    if transaction_count == 0:
        return rules

    support_lookup = {itemset: count / transaction_count for itemset, count in support_counts.items()}

    for itemset, itemset_count in support_counts.items():
        if len(itemset) < 2:
            continue

        itemset_support = itemset_count / transaction_count
        for antecedent_tuple in powerset(sorted(itemset)):
            antecedent = frozenset(antecedent_tuple)
            consequent = itemset.difference(antecedent)
            if not consequent:
                continue

            antecedent_count = support_counts.get(antecedent)
            consequent_count = support_counts.get(consequent)
            if not antecedent_count or not consequent_count:
                continue

            confidence = itemset_support / (antecedent_count / transaction_count)
            if confidence < min_confidence:
                continue

            consequent_support = consequent_count / transaction_count
            lift = confidence / consequent_support if consequent_support else 0.0

            rules.append(
                AssociationRule(
                    antecedent=tuple(sorted(antecedent)),
                    consequent=tuple(sorted(consequent)),
                    support=itemset_support,
                    confidence=confidence,
                    lift=lift,
                    support_count=itemset_count,
                    consequent_support=consequent_support,
                )
            )

    rules.sort(key=lambda rule: (-rule.confidence, -rule.support, rule.antecedent, rule.consequent))
    return rules


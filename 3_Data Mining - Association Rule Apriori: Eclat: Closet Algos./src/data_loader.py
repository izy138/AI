from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True, slots=True)
class Product:
    """Represents an entry from the product catalog."""

    product_id: str
    name: str
    category: str

    @property
    def normalized_name(self) -> str:
        return normalize_item(self.name)


class DataLoaderError(Exception):
    """Raised when loading product or transaction data fails."""


def normalize_item(raw: str | None) -> str:
    """Normalize a product name for consistent matching."""
    return (raw or "").strip().lower()


def load_products(csv_path: Path | str) -> Dict[str, Product]:
    """Load products keyed by their normalized name."""
    path = Path(csv_path)
    if not path.exists():
        raise DataLoaderError(f"Product file not found: {path}")

    products: Dict[str, Product] = {}
    try:
        with path.open(newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            required_columns = {"product_id", "product_name"}
            if not required_columns.issubset(reader.fieldnames or set()):
                raise DataLoaderError(
                    "Product file must contain 'product_id' and 'product_name' columns."
                )
            for row in reader:
                product_id = str(row["product_id"]).strip()
                product_name = str(row["product_name"]).strip()
                category = str(row.get("category", "")).strip()
                normalized = normalize_item(product_name)
                if not normalized:
                    continue
                products[normalized] = Product(
                    product_id=product_id, name=product_name, category=category
                )
    except csv.Error as exc:
        raise DataLoaderError(f"Failed to parse product file {path}: {exc}") from exc
    return products


def load_transactions(csv_path: Path | str) -> List[List[str]]:
    """Load raw transactions from a CSV file into lists of items."""
    path = Path(csv_path)
    if not path.exists():
        raise DataLoaderError(f"Transaction file not found: {path}")

    transactions: List[List[str]] = []
    try:
        with path.open(newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames or "items" not in reader.fieldnames:
                raise DataLoaderError("Transaction file must contain an 'items' column.")

            for row in reader:
                raw_items = row.get("items", "") or ""
                if isinstance(raw_items, str):
                    parts = raw_items.split(",")
                else:
                    parts = list(raw_items)
                transactions.append([part for part in parts])
    except csv.Error as exc:
        raise DataLoaderError(f"Failed to parse transactions file {path}: {exc}") from exc
    return transactions


def normalize_transactions(transactions: Iterable[Sequence[str]]) -> List[List[str]]:
    """Return transactions with normalized item names."""
    normalized_transactions: List[List[str]] = []
    for transaction in transactions:
        normalized_transactions.append([normalize_item(item) for item in transaction])
    return normalized_transactions


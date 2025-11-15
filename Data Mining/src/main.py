from __future__ import annotations

import sys
import time
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.algorithms import AssociationRule, FrequentItemsetResult, generate_association_rules  # noqa: E402
from src.algorithms.apriori import apriori  # noqa: E402
from src.algorithms.eclat import eclat  # noqa: E402
from src.analytics import compute_basic_stats, summarize_preprocessing  # noqa: E402
from src.data_loader import DataLoaderError, Product, load_products, load_transactions  # noqa: E402
from src.preprocessing.cleaner import clean_transactions  # noqa: E402


PAGE_TITLE = "Interactive Supermarket Association Explorer"
DEFAULT_MIN_SUPPORT = 0.2
DEFAULT_MIN_CONFIDENCE = 0.5
PRODUCT_BUTTON_COLUMNS = 5


def initialize_state() -> None:
    """Ensure Streamlit session state contains required keys."""
    defaults = {
        "manual_transactions": [],
        "current_selection": [],
        "uploaded_transactions": [],
        "cleaned_transactions": None,
        "preprocessing_report": None,
        "mining_results": {},
        "rules_lookup": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_data_paths() -> Tuple[Path, Path]:
    """Return paths to the bundled data files."""
    products_path = ROOT_DIR / "products.csv"
    transactions_path = ROOT_DIR / "sample_transactions.csv"
    return products_path, transactions_path


def add_manual_item(item: str) -> None:
    selection: List[str] = st.session_state["current_selection"]
    if item not in selection:
        selection.append(item)


def remove_manual_item(item: str) -> None:
    selection: List[str] = st.session_state["current_selection"]
    if item in selection:
        selection.remove(item)


def add_manual_transaction() -> None:
    selection: List[str] = st.session_state["current_selection"]
    if not selection:
        st.warning("Add at least one item before saving the transaction.")
        return
    st.session_state["manual_transactions"].append(selection.copy())
    st.session_state["current_selection"] = []
    st.success("Manual transaction added.")


def clear_manual_transactions() -> None:
    st.session_state["manual_transactions"] = []
    st.session_state["current_selection"] = []


def parse_uploaded_transactions(dataframe: pd.DataFrame) -> List[List[str]]:
    """Convert uploaded CSV data into list of raw transactions."""
    if "items" not in dataframe.columns:
        raise DataLoaderError("Uploaded CSV must include an 'items' column.")
    transactions: List[List[str]] = []
    for _, row in dataframe.iterrows():
        raw_items = str(row.get("items", "") or "")
        transactions.append([part for part in raw_items.split(",")])
    return transactions


def display_product_buttons(products: List[Product]) -> None:
    st.caption("Click items to build a shopping basket. Items won't repeat within the same basket.")
    columns = st.columns(PRODUCT_BUTTON_COLUMNS)
    for index, product in enumerate(products):
        column = columns[index % PRODUCT_BUTTON_COLUMNS]
        label = product.name.title()
        if column.button(label, key=f"product_btn_{product.product_id}"):
            add_manual_item(product.normalized_name)


def display_current_selection(products: Dict[str, Product]) -> None:
    selection: List[str] = st.session_state["current_selection"]
    if not selection:
        st.info("No items selected yet.")
        return

    st.write("**Current Basket:**")
    chips = st.columns(len(selection))
    for idx, item in enumerate(selection):
        product_name = products.get(item, Product(item, item, "")).name.title()
        chip = chips[idx]
        chip.button(
            f"❌ {product_name}",
            key=f"remove_{item}",
            on_click=remove_manual_item,
            kwargs={"item": item},
        )


def format_transactions_display(transactions: List[Sequence[str]], products: Dict[str, Product]) -> pd.DataFrame:
    """Create a display-friendly dataframe for transactions."""
    rows = []
    for transaction in transactions:
        readable = [products.get(item, Product(item, item, "")).name.title() for item in transaction]
        rows.append({"Items": ", ".join(readable)})
    df = pd.DataFrame(rows)
    df.index = pd.Index(range(1, len(df) + 1), name="Transaction ID")
    return df


def run_preprocessing(products: Dict[str, Product]) -> None:
    original_transactions = (
        st.session_state.get("manual_transactions", []) + st.session_state.get("uploaded_transactions", [])
    )
    if not original_transactions:
        st.session_state["preprocessing_report"] = None
        st.session_state["cleaned_transactions"] = None
        st.session_state["preprocessing_error"] = "Add or import transactions before running preprocessing."
        return
        return

    cleaned, report = clean_transactions(original_transactions, products)
    st.session_state["cleaned_transactions"] = cleaned
    st.session_state["preprocessing_report"] = report
    st.session_state["mining_results"] = {}
    st.session_state["rules_lookup"] = {}

    summary = summarize_preprocessing(report)
    st.success("Preprocessing complete.")
    with st.expander("Preprocessing Summary", expanded=True):
        st.subheader("Before Cleaning")
        st.write(summary["before"])
        st.subheader("After Cleaning")
        st.write(summary["after"])

    st.dataframe(format_transactions_display(cleaned, products), use_container_width=True)


def itemsets_to_dataframe(result: FrequentItemsetResult) -> pd.DataFrame:
    records = []
    for itemset, count in result.support_counts.items():
        records.append(
            {
                "Items": ", ".join(sorted(itemset)),
                "Size": len(itemset),
                "Support (%)": round((count / result.transaction_count) * 100, 2),
                "Support Count": count,
            }
        )
    return pd.DataFrame(records).sort_values(by=["Size", "Support (%)"], ascending=[True, False])


def rules_to_dataframe(rules: List[AssociationRule]) -> pd.DataFrame:
    records = []
    for rule in rules:
        records.append(
            {
                "Antecedent": ", ".join(rule.antecedent),
                "Consequent": ", ".join(rule.consequent),
                "Support (%)": round(rule.support * 100, 2),
                "Confidence (%)": round(rule.confidence * 100, 2),
                "Lift": round(rule.lift, 2),
            }
        )
    return pd.DataFrame(records)


def run_mining(min_support: float, min_confidence: float) -> None:
    cleaned_transactions: List[List[str]] | None = st.session_state["cleaned_transactions"]
    if not cleaned_transactions:
        st.warning("Run preprocessing before mining association rules.")
        return

    mining_results: Dict[str, Dict[str, float | int]] = {}
    rules_lookup: Dict[str, List[AssociationRule]] = {}

    # Apriori
    start = time.perf_counter()
    apriori_result = apriori(cleaned_transactions, min_support=min_support)
    apriori_rules = generate_association_rules(
        apriori_result.support_counts, apriori_result.transaction_count, min_confidence
    )
    apriori_runtime = (time.perf_counter() - start) * 1000

    mining_results["Apriori"] = {
        "Runtime (ms)": round(apriori_runtime, 2),
        "Frequent Itemsets": len(apriori_result.support_counts),
        "Rules Generated": len(apriori_rules),
    }
    rules_lookup["Apriori"] = apriori_rules

    # Eclat
    start = time.perf_counter()
    eclat_result = eclat(cleaned_transactions, min_support=min_support)
    eclat_rules = generate_association_rules(
        eclat_result.support_counts, eclat_result.transaction_count, min_confidence
    )
    eclat_runtime = (time.perf_counter() - start) * 1000

    mining_results["Eclat"] = {
        "runtime_ms": round(eclat_runtime, 2),
        "frequent_itemsets": len(eclat_result.support_counts),
        "rules_generated": len(eclat_rules),
    }
    rules_lookup["Eclat"] = eclat_rules

    st.session_state["mining_results"] = {
        "Apriori": apriori_result,
        "Eclat": eclat_result,
        "metrics": mining_results,
    }
    st.session_state["rules_lookup"] = rules_lookup

    st.success("Association rule mining complete.")


def combine_rules(rules_lookup: Dict[str, List[AssociationRule]]) -> List[AssociationRule]:
    """Merge rules from different algorithms, keeping the best confidence for duplicates."""
    merged: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], AssociationRule] = {}
    for rule_list in rules_lookup.values():
        for rule in rule_list:
            key = (rule.antecedent, rule.consequent)
            existing = merged.get(key)
            if not existing or rule.confidence > existing.confidence:
                merged[key] = rule
    return sorted(merged.values(), key=lambda r: (-r.confidence, -r.support))


def strength_label(confidence: float) -> str:
    if confidence >= 0.75:
        return "Strong"
    if confidence >= 0.5:
        return "Moderate"
    return "Developing"


def recommend_layout(selected_item: str, rules: List[AssociationRule]) -> Tuple[str, str]:
    top_consequents = [rule.consequent for rule in rules][:2]
    flattened = sorted({item for consequent in top_consequents for item in consequent if item != selected_item})
    layout = ", ".join(item.title() for item in flattened) if flattened else "related staples"
    bundle = [selected_item] + flattened
    bundle_text = " + ".join(item.title() for item in bundle)
    recommendation = f"Consider placing {layout} near {selected_item.title()} to encourage cross-selling."
    bundle_suggestion = f"Potential bundle: {bundle_text}"
    return recommendation, bundle_suggestion


def render_recommendations(products: Dict[str, Product]) -> None:
    rules_lookup: Dict[str, List[AssociationRule]] = st.session_state.get("rules_lookup", {})
    if not rules_lookup:
        st.info("Run association mining to generate recommendations.")
        return

    merged_rules = combine_rules(rules_lookup)
    all_items = sorted({item for rule in merged_rules for item in rule.antecedent})
    if not all_items:
        st.warning("No actionable rules found. Adjust thresholds or add more data.")
        return

    selected_item_title = st.selectbox(
        "Query a product",
        options=[products[item].name.title() if item in products else item.title() for item in all_items],
    )
    normalized_lookup = {products[item].name.title() if item in products else item.title(): item for item in all_items}
    selected_item = normalized_lookup[selected_item_title]

    matching_rules = [rule for rule in merged_rules if selected_item in rule.antecedent]
    if not matching_rules:
        st.info("No recommendations available for the selected product yet.")
        return

    st.subheader(f"Customers who bought {selected_item_title} also bought:")
    for rule in matching_rules:
        consequent_names = [products.get(item, Product(item, item, "")).name.title() for item in rule.consequent]
        display_name = ", ".join(consequent_names)
        confidence_pct = round(rule.confidence * 100, 1)
        support_pct = round(rule.support * 100, 1)
        label = strength_label(rule.confidence)
        st.write(f"- {display_name}: {confidence_pct}% of the time ({label}) — Support {support_pct}%")
        st.progress(min(1.0, rule.confidence))

    recommendation, bundle = recommend_layout(selected_item, matching_rules)
    st.info(recommendation)
    st.success(bundle)


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    initialize_state()

    products_path, sample_transactions_path = get_data_paths()
    products = load_products(products_path)
    product_list = sorted(products.values(), key=lambda product: product.name)

    st.title(PAGE_TITLE)
    st.caption("Create shopping baskets, clean transaction data, and discover purchasing patterns.")

    with st.sidebar:
        st.header("Algorithm Settings")
        min_support = st.slider(
            "Minimum Support",
            min_value=0.05,
            max_value=1.0,
            value=DEFAULT_MIN_SUPPORT,
            step=0.05,
        )
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.1,
            max_value=1.0,
            value=DEFAULT_MIN_CONFIDENCE,
            step=0.05,
        )
        st.divider()
        if st.button("Run Association Mining", use_container_width=True):
            run_mining(min_support=min_support, min_confidence=min_confidence)

    st.subheader("1. Build Manual Transactions")
    product_subset = product_list[:15]
    display_product_buttons(product_subset)
    display_current_selection(products)

    col_add, col_clear = st.columns([1, 1])
    col_add.button("Add Transaction", on_click=add_manual_transaction, use_container_width=True)
    col_clear.button("Clear Basket", on_click=lambda: st.session_state.update({"current_selection": []}), use_container_width=True)

    if st.session_state["manual_transactions"]:
        st.write("### Saved Manual Transactions")
        st.dataframe(
            format_transactions_display(st.session_state["manual_transactions"], products),
            use_container_width=True,
        )

    st.divider()
    st.subheader("2. Import Transactions from CSV")
    uploaded_file = st.file_uploader("Upload a CSV file with an 'items' column", type=["csv"])
    if uploaded_file:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            st.session_state["uploaded_transactions"] = parse_uploaded_transactions(df)
            st.success(f"Loaded {len(st.session_state['uploaded_transactions'])} transactions from {uploaded_file.name}.")
        except Exception as exc:  # broad to show message
            st.error(f"Failed to load uploaded data: {exc}")

    if st.button("Load Sample Transactions", use_container_width=True):
        sample_transactions = load_transactions(sample_transactions_path)
        st.session_state["uploaded_transactions"] = sample_transactions
        st.success(f"Sample dataset loaded with {len(sample_transactions)} transactions.")

    if st.session_state["uploaded_transactions"]:
        st.dataframe(
            format_transactions_display(st.session_state["uploaded_transactions"], products),
            use_container_width=True,
        )

    st.divider()
    st.subheader("3. Review Dataset")
    combined_transactions = (
        st.session_state["manual_transactions"] + st.session_state["uploaded_transactions"]
    )
    if combined_transactions:
        stats = compute_basic_stats(combined_transactions)
        st.metric("Transactions", stats["transaction_count"])
        st.metric("Unique Items", stats["unique_items"])
        st.metric("Total Items", stats["total_items"])
    else:
        st.info("No transactions yet. Add manual data or import a CSV to continue.")

    st.divider()
    st.subheader("4. Preprocess Transactions")
    # Enable preprocessing only when there are transactions (manual, uploaded, or sample)
    has_transactions = bool(
        st.session_state.get("manual_transactions") or st.session_state.get("uploaded_transactions")
    )
    if not has_transactions:
        st.info("Add a manual transaction, load sample transactions, or upload a CSV to enable preprocessing.")

    st.button(
        "Run Preprocessing",
        on_click=lambda: run_preprocessing(products),
        use_container_width=True,
        disabled=not has_transactions,
    )

    st.divider()
    st.subheader("5. Association Mining Results")
    mining_store = st.session_state.get("mining_results", {})
    if mining_store and "metrics" in mining_store:
        metrics_df = pd.DataFrame.from_dict(mining_store["metrics"], orient="index")
        st.dataframe(metrics_df, use_container_width=True)

        apriori_result: FrequentItemsetResult = mining_store.get("Apriori")
        eclat_result: FrequentItemsetResult = mining_store.get("Eclat")
        if apriori_result and eclat_result:
            tab_apriori, tab_eclat = st.tabs(["Apriori Itemsets", "Eclat Itemsets"])
            with tab_apriori:
                st.dataframe(itemsets_to_dataframe(apriori_result), use_container_width=True)
                st.write("Association Rules")
                st.dataframe(
                    rules_to_dataframe(st.session_state["rules_lookup"].get("Apriori", [])),
                    use_container_width=True,
                )
            with tab_eclat:
                st.dataframe(itemsets_to_dataframe(eclat_result), use_container_width=True)
                st.write("Association Rules")
                st.dataframe(
                    rules_to_dataframe(st.session_state["rules_lookup"].get("Eclat", [])),
                    use_container_width=True,
                )
    else:
        st.info("Run the mining algorithms from the sidebar to view detailed results.")

    st.divider()
    st.subheader("6. Product Recommendations")
    render_recommendations(products)


if __name__ == "__main__":
    main()


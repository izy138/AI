from __future__ import annotations

import sys
import time
from dataclasses import asdict, fields
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
from src.algorithms.closet import closet  # noqa: E402
from src.analytics import compute_basic_stats, compute_raw_transaction_stats, summarize_preprocessing  # noqa: E402
from src.data_loader import DataLoaderError, Product, load_products, load_transactions, normalize_item  # noqa: E402
from src.preprocessing.cleaner import clean_transactions  # noqa: E402


PAGE_TITLE = "Data Mining: Interactive Supermarket"
DEFAULT_MIN_SUPPORT = 0.2
DEFAULT_MIN_CONFIDENCE = 0.5
PRODUCT_BUTTON_COLUMNS = 5


def inject_button_color() -> None:
    """Inject CSS to set Streamlit buttons to the app's blue (`#0d6efd`)."""
    css = """
    <style>
    .stButton>button {
        background-color: #0d6efd !important;
        background-image: none !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(13,110,253,0.15) !important;
        border-radius: 8px !important;
    }
    .stButton>button:hover {
        filter: brightness(0.96) !important;
    }
    .stButton>button:focus {
        outline: none !important;
        box-shadow: 0 0 0 4px rgba(13,110,253,0.12) !important;
    }
    </style>
    """
    try:
        st.markdown(css, unsafe_allow_html=True)
    except Exception:
        # If Streamlit API changes, fail silently — visual tweak only.
        pass


def render_decorative_header(number: int, title: str) -> None:
        """Render a header with a decorative horizontal line above it.

        Uses inline HTML/CSS so it works without relying on Streamlit internals.
        """
        html = f"""
        <div style="margin-top:18px; margin-bottom:6px;">
            <div style="height:6px; background:linear-gradient(90deg,#0d6efd,#0ea5e9); border-radius:6px; margin-bottom:10px; box-shadow:0 6px 18px rgba(13,110,253,0.12);"></div>
            <div style="display:flex; align-items:baseline; gap:12px;">
                <div style="font-weight:700; color:#0d6efd; font-size:20px;">{number}.</div>
                <div style="font-size:20px; font-weight:700; color:#e6eef1; letter-spacing:0.2px;">{title}</div>
            </div>
        </div>
        """
        try:
                st.markdown(html, unsafe_allow_html=True)
        except Exception:
                # Fallback to plain subheader if markdown injection fails
                st.subheader(f"{number}. {title}")


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
    normalized = normalize_item(item)
    if normalized and normalized not in selection:
        selection.append(normalized)


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


def add_custom_items(custom_input: str) -> None:
    """Add custom items from text input to the current selection."""
    if not custom_input or not custom_input.strip():
        return
    
    # Split by comma and normalize each item
    items = [normalize_item(item.strip()) for item in custom_input.split(",")]
    selection: List[str] = st.session_state["current_selection"]
    
    added_count = 0
    for item in items:
        if item and item not in selection:
            selection.append(item)
            added_count += 1
    
    if added_count > 0:
        st.success(f"Added {added_count} item(s) to basket.")
    elif items:
        st.info("All items are already in the basket.")


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


def _get_size_mb(obj: object) -> float:
    """Recursively calculate the size of an object in MB."""
    def _get_size(obj: object, seen: set) -> int:
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        size = sys.getsizeof(obj)
        
        if isinstance(obj, dict):
            size += sum(_get_size(k, seen) + _get_size(v, seen) for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            size += sum(_get_size(item, seen) for item in obj)
        elif isinstance(obj, (set, frozenset)):
            size += sum(_get_size(item, seen) for item in obj)
        elif isinstance(obj, str):
            # Strings are already counted in sys.getsizeof
            pass
        elif hasattr(type(obj), '__dataclass_fields__'):
            # Handle dataclasses (including those with slots=True) - check this first
            for field in fields(obj):
                field_value = getattr(obj, field.name)
                size += _get_size(field_value, seen)
        elif hasattr(obj, '__dict__'):
            # Regular class with __dict__
            size += _get_size(obj.__dict__, seen)
        elif hasattr(obj, '__slots__'):
            # Class with __slots__ (non-dataclass)
            for slot in obj.__slots__:
                if hasattr(obj, slot):
                    size += _get_size(getattr(obj, slot), seen)
        return size
    return _get_size(obj, set()) / 1024 / 1024


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
    # Calculate memory footprint of results
    apriori_memory_mb = _get_size_mb(apriori_result) + _get_size_mb(apriori_rules)

    mining_results["Apriori"] = {
        "Runtime (ms)": round(apriori_runtime, 2),
        "Memory (MB)": round(apriori_memory_mb, 2),
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
    eclat_memory_mb = _get_size_mb(eclat_result) + _get_size_mb(eclat_rules)

    mining_results["Eclat"] = {
        "Runtime (ms)": round(eclat_runtime, 2),
        "Memory (MB)": round(eclat_memory_mb, 2),
        "Frequent Itemsets": len(eclat_result.support_counts),
        "Rules Generated": len(eclat_rules),
    }
    rules_lookup["Eclat"] = eclat_rules

    # CLOSET
    start = time.perf_counter()
    closet_result = closet(cleaned_transactions, min_support=min_support)
    closet_rules = generate_association_rules(
        closet_result.support_counts, closet_result.transaction_count, min_confidence
    )
    closet_runtime = (time.perf_counter() - start) * 1000
    closet_memory_mb = _get_size_mb(closet_result) + _get_size_mb(closet_rules)

    mining_results["CLOSET"] = {
        "Runtime (ms)": round(closet_runtime, 2),
        "Memory (MB)": round(closet_memory_mb, 2),
        "Frequent Itemsets": len(closet_result.support_counts),
        "Rules Generated": len(closet_rules),
    }
    rules_lookup["CLOSET"] = closet_rules

    st.session_state["mining_results"] = {
        "Apriori": apriori_result,
        "Eclat": eclat_result,
        "CLOSET": closet_result,
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
    inject_button_color()

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
    # Decorative header and product list belong in the main content area (not the sidebar)
    render_decorative_header(1, "Build Manual Transactions")
    # Show only products that are listed in `products.csv` when building manual transactions.
    # Use the full product list (loaded from the CSV) rather than any items inferred
    # from uploaded or previous transaction data.
    product_subset = product_list
    display_product_buttons(product_subset)
    
    # Custom item input field
    st.write("**Or enter custom items:**")
    custom_input = st.text_input(
        "Enter item(s) separated by commas (e.g., 'Custom Item, Another Item')",
        key="custom_item_input",
        placeholder="e.g., Custom Product, Special Item",
        label_visibility="collapsed"
    )
    if st.button("Add Custom Items", key="add_custom_items"):
        add_custom_items(custom_input)
    
    display_current_selection(products)

    col_add, col_clear = st.columns([1, 1])
    col_add.button("Add Transaction", on_click=add_manual_transaction, use_container_width=True)
    col_clear.button("Clear Basket", on_click=lambda: st.session_state.update({"current_selection": []}), use_container_width=True)

    if st.session_state["manual_transactions"]:
            # Filter saved manual transactions so only items present in `products.csv` are shown.
            filtered_manual = []
            for tx in st.session_state["manual_transactions"]:
                filtered_items = [item for item in tx if item in products]
                if filtered_items:
                    filtered_manual.append(filtered_items)

            if filtered_manual:
                st.write("### Saved Manual Transactions")
                st.dataframe(
                    format_transactions_display(filtered_manual, products),
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
        valid_products_set = set(products.keys())
        raw_stats = compute_raw_transaction_stats(combined_transactions, valid_products_set)
        
        st.write("**Dataset Analysis:**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", raw_stats["total_transactions"])
        col2.metric("Total Items", stats["total_items"])
        col3.metric("Unique Items", stats["unique_items"])
        
        col4, col5, col6, col7 = st.columns(4)
        col4.metric("Empty Transactions", raw_stats["empty_transactions"])
        col5.metric("Single Item Transactions", raw_stats["single_item_transactions"])
        col6.metric("Duplicate Items Found", raw_stats["duplicate_items_found"])
        col7.metric("Invalid Items Found", raw_stats["invalid_items_found"])
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

    # Display preprocessing results if available
    if st.session_state.get("preprocessing_report") is not None:
        cleaned_transactions = st.session_state.get("cleaned_transactions")
        if cleaned_transactions:
            report = st.session_state["preprocessing_report"]
            summary = summarize_preprocessing(report)
            st.success("Preprocessing complete.")
            with st.expander("Preprocessing Summary", expanded=True):
                st.subheader("Before Cleaning")
                st.write(summary["before"])
                st.subheader("After Cleaning")
                st.write(summary["after"])

            st.dataframe(format_transactions_display(cleaned_transactions, products), use_container_width=True)

    st.divider()
    st.subheader("5. Association Mining Results")
    mining_store = st.session_state.get("mining_results", {})
    if mining_store and "metrics" in mining_store:
        metrics_df = pd.DataFrame.from_dict(mining_store["metrics"], orient="index")
        st.dataframe(metrics_df, use_container_width=True)

        apriori_result: FrequentItemsetResult = mining_store.get("Apriori")
        eclat_result: FrequentItemsetResult = mining_store.get("Eclat")
        closet_result: FrequentItemsetResult = mining_store.get("CLOSET")
        if apriori_result and eclat_result and closet_result:
            tab_apriori, tab_eclat, tab_closet = st.tabs(["Apriori Itemsets", "Eclat Itemsets", "CLOSET Itemsets"])
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
            with tab_closet:
                st.dataframe(itemsets_to_dataframe(closet_result), use_container_width=True)
                st.write("Association Rules")
                st.dataframe(
                    rules_to_dataframe(st.session_state["rules_lookup"].get("CLOSET", [])),
                    use_container_width=True,
                )
    else:
        st.info("Run the mining algorithms from the sidebar to view detailed results.")

    st.divider()
    st.subheader("6. Product Recommendations")
    render_recommendations(products)


if __name__ == "__main__":
    main()


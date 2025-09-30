"""
Validation checks
"""

import pandas as pd
import numpy as np



def check_row_col_balance(matrices):
    """
    Checks row and column balances of SUT system.

    Parameters
    ----------
    matrices : list of pandas.DataFrame
        A list of DataFrames in the order: [Vn, Un, Unfd, Unva, Unm]

    Returns
    -------
    dict
        Dictionary showing mismatches (if any) between supply and use per product and per industry.
    Also prints the max absolute product and industry balance, their indices, and the sign.
    """
    Vn, Un, Unfd, Unva, Unm = matrices

    # --- Total supply (by product) - domestic only (i.e. ensure no imports column)
    total_supply = Vn.sum(axis=1)

    # --- Total use (by product) - domestic only
    total_use = Un.sum(axis=1) + Unfd.sum(axis=1) 

    # --- Total output (by industry)
    total_output = Vn.sum(axis=0)

    # --- Total input (by industry)
    total_input = Un.sum(axis=0) + Unva.sum(axis=0)

        # --- Check balances
    product_balance = total_supply - total_use
    industry_balance = total_output - total_input

    abs_product_balance = np.abs(product_balance)
    abs_industry_balance = np.abs(industry_balance)

    product_max_index = product_balance.index[abs_product_balance.argmax()] if isinstance(product_balance, pd.Series) else abs_product_balance.argmax()
    industry_max_index = industry_balance.index[abs_industry_balance.argmax()] if isinstance(industry_balance, pd.Series) else abs_industry_balance.argmax()

    product_max_val = product_balance[product_max_index] if isinstance(product_balance, pd.Series) else product_balance[product_max_index]
    industry_max_val = industry_balance[industry_max_index] if isinstance(industry_balance, pd.Series) else industry_balance[industry_max_index]

    prod_balance_relation = "supply > use" if product_max_val > 0 else "supply < use" if product_max_val < 0 else "supply = use"

    print("Max absolute product balance:", abs_product_balance.max(), "at", product_max_index, 
          f"({'positive' if product_max_val > 0 else 'negative' if product_max_val < 0 else 'zero'})",
          f"({prod_balance_relation})")
    print("Max absolute industry balance:", abs_industry_balance.max(), "at", industry_max_index, 
          f"({'positive' if industry_max_val > 0 else 'negative' if industry_max_val < 0 else 'zero'})")

    return {
        "product_balance": product_balance,
        "industry_balance": industry_balance,
        "product_balance_max_abs": abs_product_balance.max(),
        "product_balance_max_index": product_max_index,
        "product_balance_max_sign": "positive" if product_max_val > 0 else "negative" if product_max_val < 0 else "zero",
        "product_balance_relation": prod_balance_relation,
        "industry_balance_max_abs": abs_industry_balance.max(),
        "industry_balance_max_index": industry_max_index,
        "industry_balance_max_sign": "positive" if industry_max_val > 0 else "negative" if industry_max_val < 0 else "zero",
    }


def compare_disagg_to_agg(
    disagg_matrix, agg_matrix, row_agg_matrix, col_agg_matrix, top_n=10,matrix_name=None,
):
    """
    Compares a disaggregated matrix to an aggregated matrix after aggregating the disaggregated matrix.
    Reports the top N absolute differences and the underlying disaggregated values.
    """

    # --- Shape and index checks ---
    if agg_matrix.shape != (row_agg_matrix.shape[0], col_agg_matrix.shape[0]):
        raise ValueError("agg_matrix shape does not match aggregation matrix dimensions")

    if not all(idx in row_agg_matrix.index for idx in agg_matrix.index):
        raise ValueError("Some agg_matrix rows are not in row_agg_matrix index")
    if not all(col in col_agg_matrix.index for col in agg_matrix.columns):
        raise ValueError("Some agg_matrix columns are not in col_agg_matrix columns")

    # --- Aggregate disaggregated matrix ---
    aggregated_disagg = row_agg_matrix @ disagg_matrix @ col_agg_matrix.T
    aggregated_disagg_df = pd.DataFrame(
        aggregated_disagg, index=agg_matrix.index, columns=agg_matrix.columns
    )

    # --- Compute differences ---
    diff = aggregated_disagg_df - agg_matrix
    abs_diff = diff.abs()
    abs_diff_flat = abs_diff.stack()
    top_diff = abs_diff_flat.nlargest(top_n)

    # --- Collect records ---
    records = []
    for (agg_row, agg_col), abs_val in top_diff.items():
        expected_agg = agg_matrix.loc[agg_row, agg_col]
        resultant_agg = aggregated_disagg_df.loc[agg_row, agg_col]
        difference = resultant_agg - expected_agg

        # Get contributing disaggregated rows/columns
        try:
            disagg_rows = row_agg_matrix.columns[row_agg_matrix.loc[agg_row] != 0]
        except KeyError:
            disagg_rows = []
        try:
            disagg_cols = col_agg_matrix.columns[col_agg_matrix.loc[agg_col] != 0]
        except KeyError:
            disagg_cols = []

        contributing_vals = []
        for drow in disagg_rows:
            for dcol in disagg_cols:
                try:
                    val = disagg_matrix.loc[drow, dcol]
                except KeyError:
                    val = np.nan
                contributing_vals.append({
                    "agg_row": agg_row,
                    "agg_col": agg_col,
                    "disagg_row": drow,
                    "disagg_col": dcol,
                    "disagg_value": val
                })

        records.append({
            "agg_row": agg_row,
            "agg_col": agg_col,
            "difference": difference,
            "expected_agg": expected_agg,
            "resultant_agg": resultant_agg,
            "abs_difference": abs_val,
            "disagg_contributors": contributing_vals
        })
        
    top_agg_diff_df = pd.DataFrame.from_records(
        [
            {
                "agg_row": rec["agg_row"],
                "agg_col": rec["agg_col"],
                "expected_agg": rec["expected_agg"],
                "resultant_agg": rec["resultant_agg"],
                "difference": rec["difference"],
                "abs_difference": rec["abs_difference"],
            }
            for rec in records
        ]
    ).sort_values(by="abs_difference", ascending=False)

    # --- Build a flat DataFrame: one row per disagg contributor ---
    flat_rows = []
    for rec in records:
        for contrib in rec["disagg_contributors"]:
            flat_rows.append({
                "agg_row": rec["agg_row"],
                "agg_col": rec["agg_col"],
                "expected agg value": rec["expected_agg"],
                "resultant agg value": rec["resultant_agg"],
                "difference": rec["difference"],
                "abs_difference": rec["abs_difference"],
                "disagg_row": contrib["disagg_row"],
                "disagg_col": contrib["disagg_col"],
                "disagg_value": contrib["disagg_value"]
            })

    top_diff_df = pd.DataFrame(flat_rows)

    # --- Print diagnostics ---
    # print("Max absolute Agg "{table_name}" difference:", abs_diff.values.max())
    name_str = f" ({matrix_name})" if matrix_name else ""
    print(f"Max absolute difference of {name_str}: {abs_diff.values.max():.4g}")
    print("Top differences (absolute):")
    print(top_agg_diff_df.head(2))

    return diff, aggregated_disagg_df, top_diff_df,top_agg_diff_df





def compare_disagg_to_agg_vec(
    disagg_vec,
    agg_vec,
    row_agg_matrix,
    *,
    outfile=None,
    rel_floor=1e-12,
    sort_by_abs=True
):
    """
    Aggregate a disaggregated vector, compare to aggregate vector,
    and return/write a report with row/col codes and errors.

    Parameters
    ----------
    disagg_vec : pandas.Series or np.ndarray
    agg_vec : pandas.Series or np.ndarray
    row_agg_matrix : pandas.DataFrame or np.ndarray
    outfile : str, optional
        If provided, writes the results to CSV/TSV.
    rel_floor : float
        Floor to avoid division by zero in relative error.
    sort_by_abs : bool
        If True, sort results by absolute difference (descending).

    Returns
    -------
    diff : Series or ndarray
    aggregated_disagg : Series or ndarray
    report_df : pandas.DataFrame
        Columns: row, col, expected_agg, resultant_agg,
                 difference, abs_difference, rel_error
    """
    # Normalize to arrays
    if isinstance(disagg_vec, pd.Series):
        d_vals = disagg_vec.values.astype(float)
    else:
        d_vals = np.asarray(disagg_vec, dtype=float)

    if isinstance(agg_vec, pd.Series):
        a_vals = agg_vec.values.astype(float)
        a_index = agg_vec.index
    else:
        a_vals = np.asarray(agg_vec, dtype=float)
        a_index = pd.RangeIndex(len(a_vals))

    if isinstance(row_agg_matrix, pd.DataFrame):
        M = row_agg_matrix.values.astype(float)
    else:
        M = np.asarray(row_agg_matrix, dtype=float)

    # Aggregate
    aggregated_disagg = M @ d_vals
    diff = aggregated_disagg - a_vals
    abs_diff = np.abs(diff)
    rel_error = diff / np.maximum(np.abs(a_vals), rel_floor)

    # Build tidy dataframe: row and col are both agg indices
    report_df = pd.DataFrame({
        "Sector": a_index,
        "expected_agg": a_vals,
        "resultant_agg": aggregated_disagg,
        "difference": diff,
        "abs_difference": abs_diff,
        "rel_error": rel_error,
    })

    if sort_by_abs:
        report_df = report_df.sort_values("abs_difference", ascending=False).reset_index(drop=True)

    # Console summary
    print("Max absolute Agg Table difference:", abs_diff.max())

    # Write if requested
    if outfile:
        ext = outfile.lower().rsplit(".", 1)[-1]
        sep = "\t" if ext == "tsv" else ","
        report_df.to_csv(outfile, sep=sep, index=False)

    # Wrap back into Series if input was Series
    if isinstance(agg_vec, pd.Series):
        diff = pd.Series(diff, index=a_index, name="difference")
        aggregated_disagg = pd.Series(aggregated_disagg, index=a_index, name="resultant_agg")

    return diff, aggregated_disagg, report_df


def compare_tables(table1, table2, name1="Table1", name2="Table2"):
    """
    Compares two tables (DataFrames or numpy arrays) and prints summary statistics of their differences.

    Parameters
    ----------
    table1 : pandas.DataFrame or numpy.ndarray
        First table to compare.
    table2 : pandas.DataFrame or numpy.ndarray
        Second table to compare.
    name1 : str
        Name for the first table (for printing).
    name2 : str
        Name for the second table (for printing).

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        The difference table (table1 - table2).
    """
    # Convert to DataFrame if possible for better output
    is_df = isinstance(table1, pd.DataFrame) and isinstance(table2, pd.DataFrame)
    if isinstance(table1, np.ndarray):
        table1 = pd.DataFrame(table1)
        is_df = False
    if isinstance(table2, np.ndarray):
        table2 = pd.DataFrame(table2)
        is_df = False
    diff = table1 - table2
    abs_diff = np.abs(diff.values)
    max_abs = abs_diff.max()
    mean_abs = abs_diff.mean()
    min_diff = diff.values.min()
    max_diff = diff.values.max()
    min_idx = np.unravel_index(diff.values.argmin(), diff.shape)
    max_idx = np.unravel_index(diff.values.argmax(), diff.shape)
    print(f"Comparing {name1} and {name2}:")
    print(f"  Max absolute difference: {max_abs}")
    print(f"  Mean absolute difference: {mean_abs}")
    print(f"  Min difference: {min_diff} ({'positive' if min_diff > 0 else 'negative' if min_diff < 0 else 'zero'})")
    print(f"  Max difference: {max_diff} ({'positive' if max_diff > 0 else 'negative' if max_diff < 0 else 'zero'})")
    if is_df:
        row_label_max = diff.index[max_idx[0]]
        col_label_max = diff.columns[max_idx[1]]
        row_label_min = diff.index[min_idx[0]]
        col_label_min = diff.columns[min_idx[1]]
        print(f"  Max diff at row: {row_label_max}, col: {col_label_max}")
        print(f"  Min diff at row: {row_label_min}, col: {col_label_min}")
    else:
        print(f"  Max diff at indices: {max_idx}")
        print(f"  Min diff at indices: {min_idx}")
    return diff


def check_and_align_agg_indices(agg_matrix, row_agg_matrix, col_agg_matrix):
    """
    Check and align the indices and columns of agg_matrix with the row and col aggregation matrices.
    """

    # --- Check and align rows ---
    expected_rows = list(row_agg_matrix.index)
    actual_rows = list(agg_matrix.index)
    
    missing_rows = set(expected_rows) - set(actual_rows)
    extra_rows = set(actual_rows) - set(expected_rows)
    unordered_rows = actual_rows != expected_rows

    # --- Check and align columns ---
    expected_cols = list(col_agg_matrix.index)
    actual_cols = list(agg_matrix.columns)

    missing_cols = set(expected_cols) - set(actual_cols)
    extra_cols = set(actual_cols) - set(expected_cols)
    unordered_cols = actual_cols != expected_cols

    # --- Report findings ---
    print("=== ROW CHECK ===")
    print(f"Missing rows in agg_matrix: {missing_rows}")
    print(f"Extra rows in agg_matrix: {extra_rows}")
    print(f"Rows are ordered correctly? {not unordered_rows}")

    print("\n=== COLUMN CHECK ===")
    print(f"Missing columns in agg_matrix: {missing_cols}")
    print(f"Extra columns in agg_matrix: {extra_cols}")
    print(f"Columns are ordered correctly? {not unordered_cols}")

    # --- Optionally fix order ---
    try:
        aligned_agg = agg_matrix.loc[expected_rows, expected_cols]
        print("agg_matrix successfully aligned to aggregation matrix structure.")
    except KeyError as e:
        raise ValueError("agg_matrix is missing required rows or columns and cannot be aligned.") from e

    return aligned_agg


def check_import_balance(df: pd.DataFrame, threshold=1e3, name="Um_out"):
    """
    Pandas DataFrame variant. Returns (max_abs, row_label, row_sums_series).
    """
    s = df.sum(axis=1, numeric_only=True, min_count=1)
    abs_s = s.abs()
    max_abs = float(abs_s.max(skipna=True))
    row_label = abs_s.idxmax()

    print(f"[check] max |row-sum({name})| = {max_abs:.6g} at {row_label}")
    if max_abs > threshold:
        raise ValueError(f"max |row-sum({name})| = {max_abs:.6g} exceeds threshold {threshold}")

    return max_abs, row_label, s

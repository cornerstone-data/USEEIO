# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 12:55:00 2025

@author: richa
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 30 13:36:45 2025

@author: richa
"""
import pandas as pd

# --- Aggregate detailed tables to summary tables using MasterCrosswalk ---
def get_aggregation_matrix(crosswalk, detail_col, summary_col):
    matrix = pd.crosstab(crosswalk[summary_col], crosswalk[detail_col])
    return (matrix > 0).astype(int)

def build_aggregation_matrices(
    crosswalk, V, Vagg, detail_col, summary_col,
    va=None, va_agg=None, fd=None, fd_agg=None
):
    """
    Constructs row and column aggregation matrices for aligning detailed matrix V
    with an aggregated matrix Vagg, using a crosswalk.
    Optionally also builds aggregation matrices for value added (va) and final demand (fd).

    Returns:
        (Vagg_rows, Vagg_cols, VAagg_rows, VAagg_cols, FDagg_rows, FDagg_cols)
    """

    # Row aggregation matrix for V
    agg_rows = get_aggregation_matrix(crosswalk, detail_col, summary_col)
    Vagg_rows = agg_rows.loc[Vagg.index, V.index]

    # Column aggregation matrix for V
    agg_cols = get_aggregation_matrix(crosswalk, detail_col, summary_col)
    Vagg_cols = agg_cols.loc[Vagg.columns, V.columns]

    # Value Added aggregation matrices
    VAagg_rows = VAagg_cols = None
    if va is not None and va_agg is not None:
        va_rows = get_aggregation_matrix(crosswalk, detail_col, summary_col)
        VAagg_rows = va_rows.loc[va_agg.index, va.index]
        va_cols = get_aggregation_matrix(crosswalk, detail_col, summary_col)
        VAagg_cols = va_cols.loc[va_agg.columns, va.columns]

    # Final Demand aggregation matrices
    FDagg_rows = FDagg_cols = None
    if fd is not None and fd_agg is not None:
        fd_rows = get_aggregation_matrix(crosswalk, detail_col, summary_col)
        FDagg_rows = fd_rows.loc[fd_agg.index, fd.index]
        fd_cols = get_aggregation_matrix(crosswalk, detail_col, summary_col)
        FDagg_cols = fd_cols.loc[fd_agg.columns, fd.columns]

    return Vagg_rows, Vagg_cols, VAagg_rows, VAagg_cols, FDagg_rows, FDagg_cols
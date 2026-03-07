#!/usr/bin/env python
from __future__ import annotations

import os
import pandas as pd


def _fallback_infer_df_stype(df: pd.DataFrame) -> dict[str, str]:
    stypes: dict[str, str] = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            stypes[col] = "categorical"
        elif pd.api.types.is_numeric_dtype(series):
            stypes[col] = "numerical"
        elif pd.api.types.is_datetime64_any_dtype(series):
            stypes[col] = "timestamp"
        else:
            stypes[col] = "categorical"
    return stypes


def infer_df_stype(df: pd.DataFrame) -> dict[str, object]:
    """Infer semantic types with torch_frame when available.

    Falls back to a lightweight pandas-based inference when torch_frame is
    unavailable or fails to import in the current environment.
    """
    if os.getenv("GRAPHREDUCE_USE_TORCH_FRAME", "0") != "1":
        return _fallback_infer_df_stype(df)

    from torch_frame.utils import infer_df_stype as torch_infer_df_stype

    return torch_infer_df_stype(df)

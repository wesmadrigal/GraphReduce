#!/usr/bin/env python

import pytz
from datetime import datetime
import pandas as pd
import dask.dataframe as dd
from pyspark.sql import functions as F
import pyspark
from torch_frame import stype


stype_map = {
        'numerical': [
            'min',
            'max',
            'median',
            'mean',
            'sum',
            ],
        'categorical': [
            'nunique',
            'count',
            'mode',
            ],
        'text_embedded': [
            'length'
            ],
        'text_tokenized': [
            'length'
            ],
        'multicategorical': [
            'length'
            ],
        'sequence_numerical': [
            'sum',
            'min',
            'max',
            'median',
            ],
        'timestamp': [
            'min',
            'max',
            'delta'
            ],
        'image_embedded': [],
        'embedding': []
}


def clean_datetime_pandas(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Count the number of rows before removing invalid dates
    total_before = len(df)

    # Remove rows where timestamp is NaT (indicating parsing failure)
    df = df.dropna(subset=[col])

    # Count the number of rows after removing invalid dates
    total_after = len(df)

    # Calculate the percentage of rows removed
    percentage_removed = ((total_before - total_after) / total_before) * 100

    # Print the percentage of comments removed
    print(
        f"Percentage of rows removed due to invalid dates: "
        f"{percentage_removed:.2f}%"
    )
    return df


def clean_datetime_dask(df: dd.DataFrame, col: str) -> dd.DataFrame:
    df[col] = dd.to_datetime(df[col])
    total_before = len(df)
    df = df.dropna(subset=[col])
    total_after = len(df)
    percentage_removed = ((total_before - total_after) / total_before) * 100
    return df


def clean_datetime_spark(df, col: str) -> pyspark.sql.DataFrame:
    pass



def convert_to_utc(dt):
  """Converts a datetime object to UTC.

  Args:
    dt: The datetime object to convert.

  Returns:
    The datetime object converted to UTC.
  """
  if dt.tzinfo is None:  # Naive datetime
    # Assuming the original timezone is the local system time
    local_tz = pytz.timezone('US/Pacific')  # Replace with the actual timezone if known
    dt = local_tz.localize(dt)
  return dt.astimezone(pytz.UTC)

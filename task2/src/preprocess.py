import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def parse_year(df: pd.DataFrame, col: str = "Year") -> pd.DataFrame:
    # Extract the 4-digit year (e.g., from "2020 (India)") and convert it to float
    df[col] = df[col].str.extract(r"(\d{4})").astype(float)
    return df


def parse_duration(df: pd.DataFrame, col: str = "Duration") -> pd.DataFrame:
    # Extract numerical duration (e.g., from "120 min") and convert to float
    df[col] = df[col].str.extract(r"(\d+)").astype(float)
    return df


def clean_votes_helper(x) -> float:
    # If the input is already a number, return it
    if isinstance(x, (float, int)):
        return x

    # Normalize the string: lowercase, remove commas and dollar signs
    x = x.lower().replace(",", "").replace("$", "")

    # Convert "1.2m" → 1,200,000
    if "m" in x:
        return float(x.replace("m", "")) * 1e6

    # Otherwise, try converting directly to float
    return float(x)


def clean_votes(df: pd.DataFrame, col: str = "Votes") -> pd.DataFrame:
    if col == "Votes":
        df[col] = df[col].apply(clean_votes_helper)

    df[col] = (
        df[col]
        .astype(str)
        .str.replace(r"[+,]", "", regex=True)
        .replace("nan", np.nan)
        .astype(float)
    )
    return df


def handle_missing(df: pd.DataFrame, thresh: float = 0.5) -> pd.DataFrame:
    # Drop columns with > thresh missing fraction
    df = df.loc[:, df.isnull().mean() < thresh]

    # Numeric → median, Categorical → 'Unknown'
    for c in df.select_dtypes(include=[np.number]).columns:
        df = df.fillna({c: df[c].median()})

    # For categorical (object) columns, fill missing with the string "Unknown"
    for c in df.select_dtypes(include=[object]).columns:
        df = df.fillna({c: "Unknown"})

    return df


def preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    # Step-by-step data cleaning pipeline:

    # 1. Extract and convert year from string
    df = parse_year(df)

    # 2. Extract and convert duration from string
    df = parse_duration(df)

    # 3. Clean and normalize the vote counts
    df = clean_votes(df)

    # 4. Handle missing data: drop sparse columns, fill other missing values
    df = handle_missing(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preprocess and split raw movie data")
    parser.add_argument("--input", type=Path, required=True, help="raw CSV file path")
    parser.add_argument(
        "--output", type=Path, required=True, help="where to save the cleaned csv"
    )
    args = parser.parse_args()

    # 1) Load
    df = pd.read_csv(args.input)

    # 2) Clean & preprocess
    df_clean = preprocess_raw(df)

    df_clean.to_csv(args.output, index=False)

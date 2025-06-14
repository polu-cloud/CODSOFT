import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path


def encode_genre(df: pd.DataFrame, col: str = "Genre") -> pd.DataFrame:
    # Split multi-label genres into dummy columns
    vect = CountVectorizer(token_pattern="[^,]+")
    mat = vect.fit_transform(df[col].fillna(""))
    genre_df = pd.DataFrame(
        mat.toarray(),
        columns=[f"genre_{g.strip()}" for g in vect.get_feature_names_out()],
    )
    return pd.concat([df.reset_index(drop=True), genre_df], axis=1)


def encode_top_categories(df: pd.DataFrame, col: str, top_n: int = 30) -> pd.DataFrame:
    top = df[col].value_counts().nlargest(top_n).index
    cat_col = f"{col}_cat"
    df[cat_col] = df[col].where(df[col].isin(top), other="Other")

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    arr = enc.fit_transform(df[[cat_col]])
    cols = [f"{col}_{c}" for c in enc.categories_[0]]
    cat_df = pd.DataFrame(arr, columns=cols)

    df = pd.concat([df.reset_index(drop=True), cat_df], axis=1)
    return df.drop(columns=[col, cat_col])  # drop original + _cat columns


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Multi-label encode genres
    df = encode_genre(df)

    # Step 2: Encode top 30 Directors
    df = encode_top_categories(df, "Director", top_n=30)

    # Step 3: Encode top 30 Actor 1 values
    df = encode_top_categories(df, "Actor 1", top_n=30)

    # Step 4: Drop columns no longer needed or hard to encode meaningfully
    drop = ["Name", "Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]

    return df.drop(columns=[c for c in drop if c in df.columns])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Build features")
    parser.add_argument("--input", type=Path, required=True, help="clean CSV")
    parser.add_argument(
        "--output", type=Path, required=True, help="where to save feature CSV"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df_feat = build_features(df)

    assert not all(
        pd.api.types.is_numeric_dtype(df_feat[col]) for col in df_feat.columns
    ), "Non-numeric columns found!"

    df_feat.to_csv(args.output, index=False)

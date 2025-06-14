import pandas as pd
from pathlib import Path
import argparse


def load_raw_data(data_dir: Path) -> pd.DataFrame:
    """
    Load raw movie data from CSV files in the specified directory.

    Args:
        data_dir (Path): Path to the `data/raw` directory.

    Returns:
        pd.DataFrame: Concatenated raw data.
    """

    # Although for now I have only one dataset
    # I am assuming, there are several datasets. This ensures robustness
    csv_files = list(data_dir.glob("*.csv"))

    # utf-8 cannot be used in this dataset. So, I used 'cp1252' encoding instead.
    # It is a common encoding in `windows`
    df_list = [pd.read_csv(f, encoding="cp1252") for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Load raw movie CSV(s)")
    parser.add_argument(
        "--input", type=Path, required=True, help="raw CSV file or folder"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="where to dump combined CSV"
    )
    args = parser.parse_args()

    df = load_raw_data(args.input) if args.input.is_dir() else pd.read_csv(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

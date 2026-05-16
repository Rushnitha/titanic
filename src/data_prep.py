"""
data_prep.py
------------
Step 1 – Fetch the raw Titanic CSV from the internet and save it.
Step 2 – Clean and basic-preprocess the raw data; save to processed/.

Run standalone:
    python src/data_prep.py
"""

import os
import pandas as pd
from src.utils import get_logger, ensure_dir, data_path

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

RAW_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)

# Columns that carry no predictive signal or are too sparse to use
DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin"]

TARGET_COL = "Survived"

RAW_CSV  = data_path("raw",  "titanic.csv")
PROC_CSV = data_path("processed", "titanic_clean.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – Fetch
# ──────────────────────────────────────────────────────────────────────────────

def fetch_raw_data(url: str = RAW_URL, dest: str = RAW_CSV) -> pd.DataFrame:
    """
    Download the Titanic CSV and save it to data/raw/.
    Falls back to the already-present local file if the network is unavailable.

    Parameters
    ----------
    url  : str – remote CSV address
    dest : str – local file path to write

    Returns
    -------
    pd.DataFrame – the raw, unmodified dataset
    """
    ensure_dir(data_path("raw"))

    # Try network first; silently fall back to local copy
    try:
        logger.info("Fetching raw data from %s", url)
        df = pd.read_csv(url)
        df.to_csv(dest, index=False)
        logger.info("Raw data saved  →  %s   shape=%s", dest, df.shape)
    except Exception as exc:
        if os.path.exists(dest):
            logger.warning(
                "Network fetch failed (%s). Loading existing local file: %s",
                exc, dest
            )
            df = pd.read_csv(dest)
        else:
            raise RuntimeError(
                f"Cannot fetch data from URL ({exc}) "
                f"and no local file found at {dest}."
            ) from exc

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – Clean
# ──────────────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame, dest: str = PROC_CSV) -> pd.DataFrame:
    """
    Apply lightweight cleaning steps and save the result.

    Cleaning decisions
    ------------------
    1. Drop columns with high cardinality or >75 % missingness.
    2. Convert 'Sex' to a binary integer (male=1, female=0) for readability;
       OneHotEncoder in features.py handles the remaining categoricals.
    3. Log missing-value counts so the analyst can see what needs imputation.

    Parameters
    ----------
    df   : pd.DataFrame – raw data (output of fetch_raw_data)
    dest : str          – where to write the cleaned CSV

    Returns
    -------
    pd.DataFrame – cleaned dataframe (still contains NaNs; imputation is in
                   features.py to prevent train/test leakage)
    """
    logger.info("── Cleaning raw data ──")
    original_shape = df.shape

    # 1. Drop uninformative columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    logger.info("Dropped columns: %s  →  shape now %s", DROP_COLS, df.shape)

    # 2. Summarise missing values (informational only – no filling here)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.info(
            "Missing values per column:\n%s",
            missing.to_string()
        )
    else:
        logger.info("No missing values detected after dropping columns.")

    # 3. Convert Sex to int for a friendlier raw CSV
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

    # 4. Ensure correct dtypes
    df["Pclass"] = df["Pclass"].astype(str)       # treat as nominal category

    # 5. Save
    ensure_dir(data_path("processed"))
    df.to_csv(dest, index=False)
    logger.info(
        "Cleaned data saved  →  %s   original=%s  →  final=%s",
        dest, original_shape, df.shape
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Convenience loader
# ──────────────────────────────────────────────────────────────────────────────

def load_clean_data(path: str = PROC_CSV) -> pd.DataFrame:
    """Load the processed CSV produced by clean_data()."""
    logger.info("Loading cleaned data from %s", path)
    return pd.read_csv(path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw = fetch_raw_data()
    clean_data(raw)
    logger.info("data_prep complete.")
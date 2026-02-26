import os
import csv
import warnings
import numpy as np
import requests
import gdown
from pathlib import Path


# Google Drive file IDs for each climate zone
DRIVE_FILES = {
    "Aw": {
        "input_knn.csv": "1G2FwZDTx7uegIIZc9Ifo6pFQYKLl0sKU",
        "minutal_new.csv": "1J7tMorzG0ckfVEtW6FEm_XHiWRQfQqjf",
    },
    "BSh": {
        "input_knn.csv": "1iuJft77lXQbyrehWDh8NH-IY_3TU_V4U",
        "minutal_new.csv": "1A4XDnAeB_W1iDc-EXSvE8DwJMrhVh0tL",
    },
    "BSk": {
        "input_knn.csv": "1zi5Z-q1glwHSKyWj7Q8Ap_QRpcblLox5",
        "minutal_new.csv": "1oz_UkpyWlAmdYFFLHVUKBRj-TZtTL6Vi",
    },
    "BWh": {
        "input_knn.csv": "16MvQajHiMHIkgd4QYIc6tkZmXL4WOe8x",
        "minutal_new.csv": "1z168OCOfpEOfWg0bDhDMCvrGvEfnPFYd",
    },
    "Cfa": {
        "input_knn.csv": "1GrKFyxJ01bUvTX_fkIarxdAbuQ99UcR3",
        "minutal_new.csv": "1i4tAzzh6O5ILsUa0qHZW1D-QTQjijEGp",
    },
    "Cfb": {
        "input_knn.csv": "1cZ6-fP8YotQ5HzKg9GzByaY81bIauSQn",
        "minutal_new.csv": "1wkTCWoqm0z7DZ81wRuDn-X-nBRrFBoi9",
    },
    "Csa": {
        "input_knn.csv": "1kyCjDAWFTgZLg1nMI5m7wLqfCtVzni7Q",
        "minutal_new.csv": "1h9hhPlc1IR5AMVefCHCcsZ01zz7uTMhx",
    },
    "Csb": {
        "input_knn.csv": "1qNKFeEOsVgDG7w0G3D5CuXKyKdnuw9c1",
        "minutal_new.csv": "1sdHeyn2KqDr9Fon7pmTlwxlj3M6qzzVx",
    },
    "Others": {
        "input_knn.csv": "1O7u5TsKBs-Ay49jkJiL4NOQwa-PYMrP4",
        "minutal_new.csv": "1ymn988zFjFJkKtiktFcjPjWyt3BU3E1O",
    },
}


# ----------------------------
# Helper functions
# ----------------------------

def _load_csv_to_numpy(file_path, expected_min_cols=2):
    """Load a CSV and validate its shape."""
    try:
        arr = np.genfromtxt(file_path, delimiter=",", dtype=float)
    except Exception as e:
        raise ValueError(f"Could not parse {file_path}: {e}")

    if arr.size == 0 or len(arr.shape) < 2 or arr.shape[1] < expected_min_cols:
        raise ValueError(
            f"File {file_path} has invalid format — shape {arr.shape}. "
            f"Expected at least {expected_min_cols} columns."
        )

    return arr


def download_large_file_from_drive(file_id, destination):
    """Download large files from Google Drive using gdown."""
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading from Drive: {url}")
    gdown.download(url, str(destination), quiet=False)


# ----------------------------
# Main data manager function
# ----------------------------

def get_data_file(mapped_zone: str, filename: str, expected_min_cols=2) -> Path:
    """
    Fetch and cache CSV data for a given mapped climate zone.
    Downloads from Google Drive if not already available or invalid.
    """
    mapped_zone = mapped_zone.strip()
    local_dir = Path.home() / ".ethos_tised" / "data" / mapped_zone
    local_file = local_dir / filename

    local_dir.mkdir(parents=True, exist_ok=True)

    # If cached version exists, validate it
    if local_file.exists():
        try:
            arr = _load_csv_to_numpy(local_file, expected_min_cols=expected_min_cols)
            print(f"Using cached file: {local_file}")
            #print(f"{mapped_zone} {filename} shape: {arr.shape}")
            return local_file
        except Exception as e:
            print(f"Cached file validation failed: {e}")
            print("Will attempt to re-download.")

    # Find Drive file ID
    file_id = (
        DRIVE_FILES.get(mapped_zone, {}).get(filename)
        or DRIVE_FILES.get("Others", {}).get(filename)
    )
    if not file_id:
        raise RuntimeError(f"No Drive file ID found for {mapped_zone}/{filename}")

    # Download file
    download_large_file_from_drive(file_id, local_file)

    # Validate downloaded file
    arr = _load_csv_to_numpy(local_file, expected_min_cols=expected_min_cols)
    print(f"Saved file to cache: {local_file} ({arr.shape[0]} rows, {arr.shape[1]} cols)")
    return local_file
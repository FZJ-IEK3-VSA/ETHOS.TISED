"""Gap-filling for hourly Global Horizontal Irradiance (GHI) input series.

This module implements the missing-data handling used by
:meth:`ethos_tised.model.SolarModel.handle_missing_hourly_data` before the
minute-resolution downscaling pipeline runs. Because the model always
operates on one full calendar year at hourly resolution (8760 or 8784
samples), the rules below are expressed directly in units of hourly samples
rather than as a general-purpose time-series utility.

Rules, applied in this order:
    1. Night-time samples          -> 0
       (samples where the solar elevation angle is <= 0)
    2. Gaps of >= `long_gap_hours` -> filled from the same clock hour on the
       previous day (falls through to rule 4 if the previous day is itself
       missing at that hour)
    3. Gaps of a single sample     -> linear interpolation between
       (for a single missing hour, the previous and next hours must be present; 
       otherwise it falls through to rule 4)
    4. Anything left               -> KNN imputation on a (day x hour-of-day)
       matrix, so that neighbours are chosen based on similarity of daily
       irradiance shape rather than raw point-wise distance
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib
from sklearn.impute import KNNImputer

__all__ = ["impute_hourly_ghi"]


def impute_hourly_ghi(
    ghi: np.ndarray,
    lat: float,
    lon: float,
    start: pd.Timestamp,
    *,
    altitude: float = 0.0,
    long_gap_hours: int = 8,
    short_gap_samples: int = 1,
    n_neighbors: int = 3,
    verbose: bool = True,
) -> tuple[np.ndarray, pd.Series]:
    """Fill missing values (NaNs) in a one-year hourly GHI series.

    Parameters
    ----------
    ghi : np.ndarray
        1D array of length 8760 (ordinary year) or 8784 (leap year).
        Missing values must be represented as NaN.
    lat : float
        Site latitude in decimal degrees.
    lon : float
        Site longitude in decimal degrees.
    start : pd.Timestamp
        Tz-aware Timestamp for the first hour of the year (as returned
        by ``SolarModel._get_timezone_and_range``).
    altitude : float
        Site altitude in meters, used for the solar-position call.
    long_gap_hours : int
        Consecutive-missing-sample threshold (in hours) at or above which
        the previous-day fill rule applies.
    short_gap_samples : int
        Consecutive-missing-sample threshold at or below which linear
        interpolation is used. Defaults to 1, i.e. a single isolated
        missing hour -- see module docstring.
    n_neighbors : int
        k used by the KNN imputer for the remaining gaps.
    verbose : bool
        If True, print a one-line summary of how many samples were filled
        by each rule.

    Returns
    -------
    (filled_ghi, method) : the imputed 1D array (same length/dtype as
        `ghi`), and a same-length pandas Series of labels
        ('observed', 'night', 'previous_day', 'interpolation', 'knn')
        recording how each sample was produced, useful for QA/logging.

    """
    ghi = np.asarray(ghi, dtype=float)
    n = len(ghi)
    if n not in (8760, 8784):
        raise ValueError(
            "impute_hourly_ghi expects one full calendar year of hourly "
            f"data (8760 or 8784 samples); got {n}."
        )

    index = pd.date_range(start=start, periods=n, freq="h")
    series = pd.Series(ghi, index=index, copy=True)
    method = pd.Series(
        np.where(series.isna(), "unfilled", "observed"), index=index, dtype=object
    )

    # ---- Rule 1: night samples -> 0 ------------------------------------
    solar_position = pvlib.solarposition.get_solarposition(
        index.tz_convert("UTC"),
        latitude=lat,
        longitude=lon,
        altitude=altitude,
        pressure=1013.25,
    )
    is_night = (solar_position["elevation"].values <= 0)

    night_missing = series.isna().to_numpy() & is_night
    series[night_missing] = 0.0
    method[night_missing] = "night"

    # ---- Rules 2 & 3: classify remaining (daytime) missing runs --------
    missing = series.isna()
    if missing.any():
        run_id = (missing != missing.shift()).cumsum()
        for _, run in series[missing].groupby(run_id[missing]):
            run_index = run.index
            run_len = len(run_index)

            if run_len >= long_gap_hours:
                for t in run_index:
                    prev_t = t - pd.Timedelta(days=1)
                    if prev_t in series.index and pd.notna(series.loc[prev_t]):
                        series.loc[t] = series.loc[prev_t]
                        method.loc[t] = "previous_day"
                    # else: left as NaN, picked up by the KNN pass below

            elif run_len <= short_gap_samples:
                lo = index.get_indexer([run_index.min()])[0] - 1
                hi = index.get_indexer([run_index.max()])[0] + 1
                if lo < 0 or hi >= n:
                    continue  # can't interpolate at the very edge of the year
                window = series.iloc[lo : hi + 1]
                filled = window.interpolate(method="time")
                series.loc[run_index] = filled.loc[run_index]
                method.loc[run_index] = "interpolation"
            # else: leave for the KNN pass below

    # ---- Rule 4: KNN on a (day x hour-of-day) matrix --------------------
    still_missing = series.isna()
    if still_missing.any():
        frame = series.to_frame("val")
        frame["date"] = frame.index.date
        frame["hour"] = frame.index.hour
        matrix = frame.pivot_table(
            index="date", columns="hour", values="val", aggfunc="first"
        )

        imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        filled_matrix = pd.DataFrame(
            imputer.fit_transform(matrix), index=matrix.index, columns=matrix.columns
        )

        for t in index[still_missing.to_numpy()]:
            d, h = t.date(), t.hour
            series.loc[t] = filled_matrix.loc[d, h]
            method.loc[t] = "knn"

    if verbose:
        counts = method.value_counts()
        summary = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"Imputation summary ({n} samples): {summary}")

    return series.to_numpy(), method
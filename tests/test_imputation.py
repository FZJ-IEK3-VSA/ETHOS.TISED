import pathlib

import numpy as np
import pandas as pd
import pytest

from ethos_tised.imputation import impute_hourly_ghi


@pytest.fixture
def berlin_ghi():
    current_dir = pathlib.Path(__file__).parent
    root_dir = current_dir.parent
    path = root_dir.joinpath("Examples", "hourly_data", "Berlin.csv")
    return np.genfromtxt(path, delimiter=",")


@pytest.fixture
def berlin_start():
    return pd.Timestamp("2018-01-01 00:00:00", tz="Europe/Berlin")


def test_no_missing_values_are_untouched(berlin_ghi, berlin_start):
    filled, method = impute_hourly_ghi(
        berlin_ghi, lat=52.455778, lon=13.523917, start=berlin_start, altitude=34.0
    )
    np.testing.assert_allclose(filled, berlin_ghi)
    assert (method == "observed").all()


def test_single_point_gap_uses_interpolation(berlin_ghi, berlin_start):
    arr = berlin_ghi.copy()
    # pick a daytime hour in summer so it isn't classified as night
    idx = pd.Timestamp("2018-06-15 12:00:00", tz="Europe/Berlin")
    pos = (idx - berlin_start).total_seconds() // 3600
    pos = int(pos)
    arr[pos] = np.nan

    filled, method = impute_hourly_ghi(
        arr, lat=52.455778, lon=13.523917, start=berlin_start, altitude=34.0
    )
    assert not np.isnan(filled).any()
    assert method.iloc[pos] == "interpolation"


def test_long_daytime_gap_uses_previous_day(berlin_ghi, berlin_start):
    arr = berlin_ghi.copy()
    day_start_idx = 150 * 24  # end of May, long summer days
    arr[day_start_idx + 6 : day_start_idx + 15] = np.nan  # 9 daytime hours

    filled, method = impute_hourly_ghi(
        arr, lat=52.455778, lon=13.523917, start=berlin_start, altitude=34.0
    )
    assert not np.isnan(filled).any()
    filled_slice = method.iloc[day_start_idx + 6 : day_start_idx + 15]
    assert (filled_slice == "previous_day").all()
    # values should match the previous day's same hours exactly
    prev_day_slice = arr[day_start_idx - 24 + 6 : day_start_idx - 24 + 15]
    np.testing.assert_allclose(filled[day_start_idx + 6 : day_start_idx + 15], prev_day_slice)


def test_night_gap_filled_with_zero(berlin_ghi, berlin_start):
    arr = berlin_ghi.copy()
    # January 1st, 02:00 local time is night in Berlin
    arr[2] = np.nan

    filled, method = impute_hourly_ghi(
        arr, lat=52.455778, lon=13.523917, start=berlin_start, altitude=34.0
    )
    assert method.iloc[2] == "night"
    assert filled[2] == 0.0


def test_medium_gap_uses_knn_and_leaves_no_nans(berlin_ghi, berlin_start):
    arr = berlin_ghi.copy()
    day_start_idx = 150 * 24
    arr[day_start_idx + 8 : day_start_idx + 12] = np.nan  # 4-hour daytime gap

    filled, method = impute_hourly_ghi(
        arr, lat=52.455778, lon=13.523917, start=berlin_start, altitude=34.0
    )
    assert not np.isnan(filled).any()
    assert (method.iloc[day_start_idx + 8 : day_start_idx + 12] == "knn").all()


def test_rejects_wrong_length_input(berlin_start):
    with pytest.raises(ValueError):
        impute_hourly_ghi(
            np.zeros(100), lat=52.455778, lon=13.523917, start=berlin_start
        )

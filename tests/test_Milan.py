import pathlib

import numpy as np
import pandas as pd

from ethos_tised import SolarModel


def test_solar_models():
    current_dir = pathlib.Path(__file__).parent
    root_dir = current_dir.parent
    hourly_irrad_data_path = root_dir.joinpath("Examples", "hourly_data", "Milan.csv")
    hourly_irrad_m = np.genfromtxt(
        hourly_irrad_data_path,
        delimiter=",",
    )

    synthetic_data_frame = SolarModel(
        Lat=45.5028249, Lon=9.1561092, date=2017, data=hourly_irrad_m
    )

    path_to_syntehtic_data = root_dir.joinpath(
        "tests", "test_data", "synthetic_data_Milan.csv"
    )
    synthetic_data_assert = pd.read_csv(path_to_syntehtic_data)
    pd.testing.assert_series_equal(
        synthetic_data_frame.loc[:, "synthetic_ghi"],
        synthetic_data_assert.loc[:, "synthetic_ghi"],
        check_dtype=False,
        atol=1e-3,
    )


if __name__ == "__main__":
    test_solar_models()
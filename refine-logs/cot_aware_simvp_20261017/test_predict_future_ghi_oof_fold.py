from datetime import datetime, timedelta

from predict_future_ghi_oof_fold import future_geometry_paths, observed_paths


def main():
    paths = [(datetime(2026, 4, 1) + timedelta(minutes=15*i)).strftime("2026/202604/20260401/%Y%m%d%H%M.npy") for i in range(24)]
    row = {"data_relpaths": "|".join(paths)}
    hist = observed_paths(row)
    future = future_geometry_paths(row)
    assert len(hist) == 8 and len(future) == 16
    assert hist == paths[:8] and future == paths[8:]
    # The model-input path selector only exposes observed AGRI paths.
    assert not set(hist).intersection(future)
    print("OOF_FORECAST_CAUSAL_PATH_WITNESS_PASS")


if __name__ == "__main__":
    main()

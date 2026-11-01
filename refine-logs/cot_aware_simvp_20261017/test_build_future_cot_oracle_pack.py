import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_future_cot_oracle_pack import materialize_split


class OraclePackBuilderTest(unittest.TestCase):
    def test_only_future_slots_are_replaced_at_exact_context_times(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            forecast = root / "forecast"
            actual = root / "actual"
            output = root / "oracle"
            (forecast / "train").mkdir(parents=True)
            actual.mkdir()
            rows = [
                {"split": "train", "station": "sili", "lead_minutes": "15",
                 "target_time_utc": "2024-04-01 00:15:00"},
                {"split": "train", "station": "sili", "lead_minutes": "30",
                 "target_time_utc": "2024-04-01 00:30:00"},
            ]
            with (forecast / "train/keys.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            original = np.full((2, 3, 1, 3, 3), -7, dtype=np.float32)
            original[:, 0] = 11
            original[:, 1] = 12
            original[:, 2] = 13
            np.save(forecast / "train/cot_log1p.npy", original)
            for name in ("agri_raw.npy", "geometry.npy", "ghi.npy", "clear.npy"):
                (forecast / "train" / name).write_bytes(name.encode())
            refs = np.zeros((2, 2, 1, 16, 16), dtype=np.float32)
            grid = np.arange(256, dtype=np.float32).reshape(16, 16)
            refs[0, 0, 0] = grid + 1000
            refs[1, 0, 0] = grid + 2000
            index = {("sili", "2024-04-01 00:15:00"): 0,
                     ("sili", "2024-04-01 00:30:00"): 1}
            index = {k: (v, 0) for k, v in index.items()}
            materialize_split(forecast, output, "train", index, refs, chunk_rows=1)
            got = np.load(output / "train/cot_log1p.npy", allow_pickle=False)
            # +15 min: only the current forecast slot is replaced; two history slots remain.
            self.assertTrue(np.all(got[0, 0] == 11))
            self.assertTrue(np.all(got[0, 1] == 12))
            self.assertTrue(np.array_equal(got[0, 2], refs[0, 0, :, 7:10, 7:10]))
            # +30 min: init-time history remains unchanged; +15/+30 use their exact target times.
            self.assertTrue(np.all(got[1, 0] == 11))
            self.assertTrue(np.array_equal(got[1, 1], refs[0, 0, :, 7:10, 7:10]))
            self.assertTrue(np.array_equal(got[1, 2], refs[1, 0, :, 7:10, 7:10]))
            self.assertEqual((output / "train/agri_raw.npy").read_bytes(),
                             (forecast / "train/agri_raw.npy").read_bytes())


if __name__ == "__main__":
    unittest.main()

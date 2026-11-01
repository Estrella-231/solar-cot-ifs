import unittest
from datetime import datetime, timedelta
from hubei_ghi15_contract import aggregate_window


class IntervalTests(unittest.TestCase):
    def setUp(self):
        self.target = datetime(2026, 4, 1, 10, 15)
        self.start = self.target - timedelta(minutes=15)
        self.records = [(self.start + timedelta(minutes=i), float(i))
                        for i in range(1, 16)]

    def test_end_included_start_excluded(self):
        rows = self.records + [(self.start, 999999),
                               (self.target + timedelta(microseconds=1), 999999)]
        self.assertEqual(aggregate_window(rows, self.target)["ghi"], 8)

    def test_seconds_offsets(self):
        for seconds in (3, 14, 49, 52):
            rows = [(self.start + timedelta(minutes=i, seconds=seconds), float(i))
                    for i in range(15)]
            self.assertEqual(aggregate_window(rows, self.target)["ghi"], 7)

    def test_zero_is_valid(self):
        rows = [(t, 0) for t, _ in self.records]
        self.assertEqual(aggregate_window(rows, self.target)["ghi"], 0)

    def test_missing_minute(self):
        self.assertFalse(aggregate_window(self.records[:-1], self.target)["valid"])

    def test_minute_bins_have_equal_weight(self):
        rows = self.records + [(self.start + timedelta(seconds=30), 3)]
        self.assertAlmostEqual(aggregate_window(rows, self.target)["ghi"], 8 + 1/15)

    def test_midnight_window(self):
        target = datetime(2026, 4, 2)
        rows = [(target-timedelta(minutes=15-i), i) for i in range(1, 16)]
        self.assertEqual(aggregate_window(rows, target)["ghi"], 8)

    def test_duplicate_identical_counted_once(self):
        result = aggregate_window(self.records + [self.records[0]], self.target)
        self.assertEqual(result["ghi"], 8)
        self.assertEqual(result["unique_records"], 15)

    def test_conflicting_duplicate(self):
        result = aggregate_window(self.records + [(self.records[0][0], 100)], self.target)
        self.assertEqual(result["reason"], "conflicting_duplicate")

    def test_invalid_value(self):
        for value in (-1, float("nan"), float("inf"), None):
            rows = self.records[:-1] + [(self.target, value)]
            self.assertFalse(aggregate_window(rows, self.target)["valid"])


if __name__ == "__main__":
    unittest.main()

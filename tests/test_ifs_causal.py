import sys
import unittest
from pathlib import Path
from datetime import datetime, timedelta, timezone
sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from ifs_causal import accumulated_to_flux, interval_mean, select_cycle

T = datetime(2025, 7, 1, tzinfo=timezone.utc)


class CausalTests(unittest.TestCase):
    def test_energy_conservation(self):
        native = accumulated_to_flux([(T, 0), (T+timedelta(hours=1), 360000)], 'J m**-2')
        quarter = [interval_mean(native, T+timedelta(minutes=15*k), T+timedelta(minutes=15*(k+1))) for k in range(4)]
        self.assertEqual(quarter, [100.0]*4)
        self.assertEqual(sum(quarter)*900, 360000)

    def test_cross_hour_weighting(self):
        native = [(T, T+timedelta(hours=1), 100), (T+timedelta(hours=1), T+timedelta(hours=2), 200)]
        self.assertAlmostEqual(interval_mean(native, T+timedelta(minutes=55), T+timedelta(minutes=70)), 500/3)

    def test_missing_release_and_future_cycle(self):
        base = dict(product_contract='fixed', product_id='a', cycle=T-timedelta(hours=12),
                    release=T-timedelta(hours=5), sha256='hash', intervals=[(T, T+timedelta(hours=1), 100)])
        future = dict(base, product_id='b', release=T+timedelta(minutes=1))
        unknown = dict(base, product_id='c', release=None)
        chosen, rejects = select_cycle([future, unknown, base], T, [(T, T+timedelta(minutes=15))])
        self.assertEqual(chosen[0]['product_id'], 'a')
        self.assertEqual(len(rejects), 2)

    def test_gap_reset_and_wrong_units(self):
        with self.assertRaises(ValueError):
            interval_mean([(T, T+timedelta(minutes=10), 100)], T, T+timedelta(minutes=15))
        with self.assertRaises(ValueError):
            accumulated_to_flux([(T, 100), (T+timedelta(hours=1), 0)], 'J m-2')
        with self.assertRaises(ValueError):
            accumulated_to_flux([(T, 0), (T+timedelta(hours=1), 100)], 'W m-2')

    def test_duplicate_hash_conflict(self):
        a = dict(product_contract='x', product_id='a', cycle=T, release=T, sha256='one', intervals=[])
        with self.assertRaises(ValueError):
            select_cycle([a, dict(a, sha256='two')], T, [(T, T+timedelta(minutes=15))])


if __name__ == '__main__':
    unittest.main()

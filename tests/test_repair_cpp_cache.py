import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    import netCDF4 as nc
except ImportError:
    nc = None
from load_repaired_cpp import load_repaired_sample, cot_target


@unittest.skipIf(nc is None, 'netCDF4 integration tests run on server')
class RepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.original = self.root/'sili/2024/202404/20240402/202404020000.npz'
        self.original.parent.mkdir(parents=True)
        lat = np.arange(20,dtype=np.float32)[::-1]
        lon = np.arange(20,dtype=np.float32)
        self.lat, self.lon = np.meshgrid(lat[1:17],lon[1:17],indexing='ij')
        agri = np.ones((20,16,16),np.float32)
        agri[18] = 30
        np.savez(self.original,agri=agri,cpp=np.zeros((4,16,16)),
            cpp_crop_bounds=[2,18,2,18],timestamp_utc='2024-04-02T00:00:00Z',
            station_id='sili',grid_lat=self.lat,grid_lon=self.lon,ghi_5min=[111,222,333])
        self.before = self.original.read_bytes()
        self.source = self.root/'cpp/2024/20240402/FY4B_AGRI_20240402000000.nc'
        self.source.parent.mkdir(parents=True)
        with nc.Dataset(self.source,'w') as ds:
            ds.createDimension('y',20); ds.createDimension('x',20)
            ds.createVariable('LAT','f4',('y',))[:] = lat
            ds.createVariable('LON','f4',('x',))[:] = lon
            for name in ('COT','CER','CTH','CLP'):
                var=ds.createVariable(name,'f4',('y','x'),fill_value=np.nan)
                var.setncattr('Range','0 - 100')
                var[:] = np.arange(400).reshape(20,20)%101

    def tearDown(self):
        self.temp.cleanup()

    def run_one(self):
        from repair_cpp_cache import repair_one
        return repair_one((str(self.original),str(self.original.relative_to(self.root)),
            str(self.root/'cpp'),str(self.root/'out'),'test-contract'))

    def test_corrected_values_and_original_unchanged(self):
        r=self.run_one()
        self.assertEqual(r['status'],'ok')
        self.assertEqual(self.before,self.original.read_bytes())
        z=load_repaired_sample(self.original,r['sidecar_path'])
        np.testing.assert_array_equal(z['cpp'][0],(np.arange(400).reshape(20,20)%101)[1:17,1:17])
        np.testing.assert_array_equal(z['ghi_5min'],[111,222,333])
        self.assertNotIn('cpp_crop_bounds',z)
        target,mask=cot_target(z)
        self.assertTrue(mask.all())
        self.assertEqual(float(target.max()),1.0)

    def test_missing_source_never_becomes_clear_sky(self):
        self.source.unlink()
        r=self.run_one()
        self.assertEqual(r['status'],'missing_source')
        z=load_repaired_sample(self.original,r['sidecar_path'])
        self.assertTrue(np.isnan(z['cpp']).all())
        self.assertFalse(z['cot_reference_mask'].any())

    def test_mismatched_original_rejected(self):
        r=self.run_one()
        self.original.write_bytes(self.before+b'x')
        with self.assertRaises(ValueError):
            load_repaired_sample(self.original,r['sidecar_path'])

    def test_resume_ledger_discards_only_partial_tail(self):
        from repair_cpp_cache import read_ledger
        ledger=self.root/'ledger.jsonl'
        complete=b'{"relative_path":"a","status":"error"}\n{"relative_path":"a","status":"ok"}\n'
        ledger.write_bytes(complete+b'{"relative_path":')
        self.assertEqual(read_ledger(ledger)['a']['status'],'ok')
        self.assertEqual(ledger.read_bytes(),complete)


if __name__=='__main__':
    unittest.main()

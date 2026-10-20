import unittest
from datetime import datetime,timedelta
from hubei_forecast_time_keys import canonical_bjt_stamps

class CanonicalTimeTests(unittest.TestCase):
    def row(self,start):
        ts=[start+timedelta(minutes=15*i) for i in range(24)]
        return dict(data_paths='|'.join('/data/'+(t-timedelta(hours=8)).strftime('%Y%m%d%H%M')+'.npy' for t in ts),aux_paths='|'.join('/aux/'+t.strftime('%Y%m%d%H%M')+'_1_aux.npz' for t in ts))
    def test_utc_midnight_is_previous_day_not_split_day(self):
        r=self.row(datetime(2025,8,22,6));v=canonical_bjt_stamps(r)
        self.assertEqual(v[7],'202508220745');self.assertEqual(v[8],'202508220800');self.assertEqual(v[23],'202508221145')
    def test_reject_matching_digits_without_timezone_conversion(self):
        r=self.row(datetime(2025,8,22,8));r['aux_paths']=r['data_paths']
        with self.assertRaises(ValueError):canonical_bjt_stamps(r)
    def test_reject_omitted_frame(self):
        r=self.row(datetime(2025,8,22,8));r['aux_paths']='|'.join(r['aux_paths'].split('|')[:-1])
        with self.assertRaises(ValueError):canonical_bjt_stamps(r)

if __name__=='__main__':unittest.main()

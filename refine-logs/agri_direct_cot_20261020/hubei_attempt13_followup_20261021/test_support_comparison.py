"""Synthetic unit checks for support accounting; not experimental evidence."""
import unittest
import numpy as np
from audit_cpp_geometry_support import compare_masks

class SupportComparisonTests(unittest.TestCase):
    def test_missing_added_and_removed_support(self):
        old=np.array([[True,True,True,False]])
        new=np.array([[True,False,False,True]])
        cos=np.array([[.5,.1,.5,.5]])
        day=np.array([[1.,1.,0.,1.]])
        r=compare_masks(old,new,cos,day)
        self.assertEqual((r['same_valid'],r['old_only'],r['corrected_only']),(1,2,1))
        self.assertEqual(r['old_valid_unambiguously_outside_corrected_geometry'],2)
        self.assertEqual(r['corrected_valid_outside_corrected_geometry'],0)

    def test_identical_valid_day_support(self):
        m=np.ones((2,3),bool);r=compare_masks(m,m,np.full((2,3),.5),np.ones((2,3)))
        self.assertEqual(r['same_valid'],6)
        self.assertEqual(r['old_only']+r['corrected_only']+r['old_valid_outside_corrected_geometry'],0)

    def test_non_boolean_mask_is_rejected(self):
        with self.assertRaises(AssertionError):
            compare_masks(np.array([[2]],np.uint8),np.ones((1,1),bool),np.ones((1,1)),np.ones((1,1)))

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(AssertionError):
            compare_masks(np.ones((1,1),bool),np.ones((1,2),bool),np.ones((1,1)),np.ones((1,1)))

if __name__=='__main__':unittest.main()

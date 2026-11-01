import run_direct_cot_lead_hcv as m

assert [m.arm_uses_lead(x) for x in m.ARMS] == [False,False,True,True]
assert m.Head(False).readout[0].in_features == 36*9
assert m.Head(True).readout[0].in_features == 36*9+8
assert m.ARMS == ('none_no_lead','pxc_no_lead','none_lead','pxc_lead')
print('DIRECT_COT_LEAD_FACTORIAL_CONTRACT_PASS')

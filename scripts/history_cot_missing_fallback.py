"""Missing-input policy; a valid clear-sky COT=0 is NOT a missing value."""
import numpy as np

def apply_history_missing_fallback(base, cot_head, history_valid):
    base=np.asarray(base);cot_head=np.asarray(cot_head);valid=np.asarray(history_valid)
    if base.shape!=cot_head.shape or valid.dtype!=np.bool_ or valid.ndim!=5 or valid.shape[1:]!=(8,1,3,3):
        raise ValueError('Expected matching [...,N] predictions and bool [N,8,1,3,3] QA')
    if base.shape[-1]!=len(valid) or not np.isfinite(base).all() or not np.isfinite(cot_head).all():
        raise ValueError('Prediction rows or finite values invalid')
    any_valid=valid.any(axis=(1,2,3,4))
    return np.where(any_valid,cot_head,base)

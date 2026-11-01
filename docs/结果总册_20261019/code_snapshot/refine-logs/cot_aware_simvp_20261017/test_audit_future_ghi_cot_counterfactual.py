import numpy as np

from audit_future_ghi_cot_counterfactual import model_cot, shuffled_future


def main():
    n = 6
    cot = np.arange(n * 3 * 9, dtype=np.float32).reshape(n, 3, 1, 3, 3)
    mask = np.tile(np.asarray([[1, 1, 0]], dtype=np.float32), (n, 1))
    keys = [{"lead_minutes": 15, "target_time_utc": f"2026-07-{i+1:02d} 04:00:00"} for i in range(n)]
    cosz = np.full(n, 0.7, dtype=np.float32)
    shuffled, info = shuffled_future(cot, mask, keys, cosz, 101)
    assert shuffled.shape == cot.shape
    assert info["rows_with_different_donor"] == n
    assert np.array_equal(shuffled[:, :2], cot[:, :2])
    assert not np.array_equal(shuffled[:, 2], cot[:, 2])
    # The donor fields are permuted whole 3x3 maps, preserving the group's field multiset.
    assert sorted(x.tobytes() for x in shuffled[:, 2]) == sorted(x.tobytes() for x in cot[:, 2])
    native = model_cot(cot, mask, "AC", "native")
    half = model_cot(cot, mask, "AC50", "native")
    zero = model_cot(cot, mask, "AC", "future_zero")
    assert np.array_equal(native[:, :2], cot[:, :2])
    assert np.allclose(half[:, 2], 0.5 * cot[:, 2])
    assert np.allclose(zero[:, 2], 0.0)
    # Match the production loader's already expanded [N,3,1,1,1] mask.
    loader_mask = mask[:, :, None, None, None]
    for scenario in ("native", "future_zero"):
        got = model_cot(cot, loader_mask, "AC", scenario)
        expected = native if scenario == "native" else zero
        assert got.shape == cot.shape and np.array_equal(got, expected)
    try:
        model_cot(cot, np.zeros((n, 3, 1, 1, 1, 1), dtype=np.float32), "AC", "native")
    except ValueError:
        pass
    else:
        raise AssertionError("malformed mask shape must be rejected")
    print("FUTURE_COT_COUNTERFACTUAL_WITNESS_PASS")


if __name__ == "__main__":
    main()

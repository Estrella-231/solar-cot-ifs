import torch

from cot_cpp_only_auxiliary import cpp_only_cot_target


def test_cpp_only_target_never_fills_missing_cpp_from_r():
    cpp = torch.tensor([[[[0.0, 2.0, float("nan"), 50.0]]]])
    cpp_mask = torch.tensor([[[[True, True, False, True]]]])
    r_valid = torch.ones_like(cpp_mask)
    target, valid, cpp_used, fallback_used = cpp_only_cot_target(cpp, cpp_mask, r_valid)
    assert valid.tolist() == [[[[True, True, False, True]]]]
    assert cpp_used.equal(valid)
    assert not fallback_used.any()
    assert torch.allclose(target[valid], torch.log1p(torch.tensor([0.0, 2.0, 50.0])))
    assert torch.isfinite(target).all()


def test_cpp_only_respects_r_input_validity():
    cpp = torch.tensor([[[[1.0, 3.0]]]])
    cpp_mask = torch.ones_like(cpp, dtype=torch.bool)
    r_valid = torch.tensor([[[[True, False]]]])
    target, valid, cpp_used, fallback_used = cpp_only_cot_target(cpp, cpp_mask, r_valid)
    assert valid.tolist() == [[[[True, False]]]]
    assert cpp_used.equal(valid)
    assert not fallback_used.any()
    assert target[0, 0, 0, 1].item() == 0.0


if __name__ == "__main__":
    test_cpp_only_target_never_fills_missing_cpp_from_r()
    test_cpp_only_respects_r_input_validity()
    print("CPP_ONLY_TARGET_WITNESS_PASS")

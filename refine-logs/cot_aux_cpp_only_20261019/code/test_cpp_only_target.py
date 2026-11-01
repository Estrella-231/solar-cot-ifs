import torch

from cot_auxiliary import cpp_only_cot_target


def main():
    cpp = torch.tensor([[[[[1.0, float("nan"), -1.0, 2.0]]]]])
    cpp_mask = torch.tensor([[[[[True, True, True, False]]]]])
    r_valid = torch.tensor([[[[[True, True, True, True]]]]])
    target, valid, from_cpp, from_r = cpp_only_cot_target(cpp, cpp_mask, r_valid)
    assert valid.tolist() == [[[[[True, False, False, False]]]]]
    assert torch.allclose(target[valid], torch.log1p(torch.tensor([1.0])))
    assert torch.isfinite(target).all()
    assert torch.equal(from_cpp, valid)
    assert not from_r.any()
    print("CPP_ONLY_TARGET_CPU_WITNESS_PASS", {"valid": int(valid.sum()), "fallback": int(from_r.sum())})


if __name__ == "__main__":
    main()

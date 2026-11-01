"""Small deterministic CPU witness for provisional T07 A/AC/AP shapes."""
import json
import torch

from head_candidate_v0 import LocalGHIHead, parameter_counts


def run():
    torch.set_num_threads(1)
    torch.manual_seed(20261013)
    x = torch.randn(8, 13, 3, 3, requires_grad=True)
    geom = torch.randn(8, 2, 3, 3, requires_grad=True)  # provisional until T03
    cot = torch.rand(8, 1, 3, 3, requires_grad=True)
    models = {}
    for v in ("A", "AC", "AP"):
        torch.manual_seed(42)
        models[v] = LocalGHIHead(v, geometry_channels=2)
    counts = parameter_counts(2)
    outputs = {v: m(x, geom, cot if v == "AC" else None) for v, m in models.items()}
    assert all(y.shape == (8,) and torch.isfinite(y).all() and (y >= 0).all()
               for y in outputs.values())

    # A has no COT input pathway; the common trunk begins from identical weights.
    torch.testing.assert_close(models["A"].agri_encoder[0].weight,
                               models["AC"].agri_encoder[0].weight, rtol=0, atol=0)
    a0 = models["A"](x, geom, torch.zeros_like(cot))
    a1 = models["A"](x, geom, cot * 100)
    torch.testing.assert_close(a0, a1, rtol=0, atol=0)

    # AC must respond to COT and pass a finite, nonzero gradient to the field.
    ac0 = models["AC"](x, geom, torch.zeros_like(cot)).sum()
    ac1 = models["AC"](x, geom, cot).sum()
    assert not torch.equal(ac0.detach(), ac1.detach())
    grad_cot = torch.autograd.grad(ac1, cot, retain_graph=True)[0]
    assert torch.isfinite(grad_cot).all() and grad_cot.abs().sum() > 0

    # AP uses AGRI-derived spatial features and matches AC capacity within 5%.
    ap = models["AP"](x, geom).sum()
    grad_x, grad_geom = torch.autograd.grad(ap, (x, geom))
    assert torch.isfinite(grad_x).all() and grad_x.abs().sum() > 0
    assert torch.isfinite(grad_geom).all() and grad_geom.abs().sum() > 0
    assert abs(counts["AC"] - counts["AP"]) / counts["AC"] <= 0.05

    return {
        "state": "PASS_PROVISIONAL_CPU_SHAPE_GRADIENT_CAPACITY",
        "test_used": False,
        "input_shapes": {"agri": [8, 13, 3, 3], "geometry": [8, 2, 3, 3],
                         "cot": [8, 1, 3, 3]},
        "geometry_channels": "provisional; T03 has not frozen it",
        "parameters": counts,
        "AC_AP_parameter_delta_fraction": abs(counts["AC"] - counts["AP"]) / counts["AC"],
        "spatial_pooling": "none; fixed row-major 3x3 flatten at readout",
        "tests": ["finite nonnegative kt outputs", "A ignores COT", "common trunk init matches",
                  "AC output responds to COT", "nonzero COT gradient", "AP AGRI/geometry gradients",
                  "AC/AP parameter delta <=5 percent"],
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

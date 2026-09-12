import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import torch
from train_cot_repaired import COTUNet, loss_fn


class COTTrainingContract(unittest.TestCase):
    def test_invalid_pixels_do_not_change_loss_or_gradients(self):
        pred = torch.tensor([[[[1.0, 4.0]]]], requires_grad=True)
        reference = torch.tensor([[[[0.0, 1000.0]]]])
        mask = torch.tensor([[[[True, False]]]])
        loss = loss_fn(pred, reference, mask)
        self.assertAlmostEqual(float(loss), 0.095, places=6)
        loss.backward()
        self.assertAlmostEqual(float(pred.grad.flatten()[0]), 0.1, places=6)
        self.assertEqual(float(pred.grad.flatten()[1]), 0.0)

    def test_nonnegative_output_and_shared_latent_hook(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = COTUNet().eval()
        x = torch.randn(2, 16, 16, 16)
        features = model.forward_features(x)
        output = model(x)
        self.assertEqual(tuple(features.shape), (2, 16, 16, 16))
        self.assertEqual(tuple(output.shape), (2, 1, 16, 16))
        self.assertTrue(bool((output >= 0).all()))
        torch.testing.assert_close(output, torch.nn.functional.softplus(model.head(features)))


if __name__ == '__main__':
    unittest.main()

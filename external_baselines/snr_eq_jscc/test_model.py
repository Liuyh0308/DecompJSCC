import unittest

import torch

from external_baselines.snr_eq_jscc.model import SNREQJSCC, SNREQJSCCConfig


class SNREQJSCCTests(unittest.TestCase):
    def test_complete_model_forward_and_condition_gradients(self):
        model = SNREQJSCC(
            SNREQJSCCConfig(
                transmit_channels=12,
                image_size=32,
                model_size="small",
                hidden_dim=4,
                penalty_weight=10.0,
                pass_channel=False,
            )
        )
        images = torch.rand(2, 3, 32, 32)
        output = model(images, torch.tensor([1.0, 13.0]), rate=12, return_aux=True)
        self.assertEqual(output["reconstructed"].shape, images.shape)
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertEqual(output["snr_eq_aux"]["query_scale"].shape, (2,))
        output["loss"].backward()
        condition_gradients = []
        for attention in model.snr_eq_attentions:
            for module in (attention.query_scale, attention.query_bias, attention.snr_embedding):
                condition_gradients.extend(
                    parameter.grad for parameter in module.parameters() if parameter.grad is not None
                )
        self.assertTrue(condition_gradients)
        self.assertGreater(sum(float(item.abs().sum()) for item in condition_gradients), 0.0)

    def test_fixed_rate_contract(self):
        model = SNREQJSCC(
            SNREQJSCCConfig(
                transmit_channels=12,
                image_size=32,
                model_size="small",
                pass_channel=False,
            )
        )
        with self.assertRaisesRegex(ValueError, "fixed-rate"):
            model(torch.rand(1, 3, 32, 32), 7.0, rate=8)


if __name__ == "__main__":
    unittest.main()

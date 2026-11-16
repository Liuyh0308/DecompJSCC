import unittest

import torch

from external_baselines.sa_ra_jscc.model import SARAJSCC, SARAJSCCConfig


class SARAJSCCTests(unittest.TestCase):
    def test_complete_joint_model_forward_budget_and_gradients(self):
        model = SARAJSCC(
            SARAJSCCConfig(image_size=32, model_size="small", pass_channel=False)
        )
        images = torch.rand(2, 3, 32, 32)
        output = model(images, torch.tensor([1.0, 13.0]), rate=32, return_aux=True)
        self.assertEqual(output["reconstructed"].shape, images.shape)
        self.assertAlmostEqual(output["cbr"], 32 / 1536)
        self.assertTrue(
            torch.equal(
                output["sa_ra_aux"]["channel_mask"].sum(dim=1),
                torch.tensor([32.0, 32.0]),
            )
        )
        output["loss"].backward()
        gradient = model.latent_conditioner.rate_adapter.mlp[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_rate_must_fit_latent_dimension(self):
        model = SARAJSCC(
            SARAJSCCConfig(image_size=32, model_size="small", pass_channel=False)
        )
        with self.assertRaisesRegex(ValueError, "latent channel"):
            model(torch.rand(1, 3, 32, 32), 7.0, rate=321)


if __name__ == "__main__":
    unittest.main()

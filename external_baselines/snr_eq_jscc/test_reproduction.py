import unittest

import torch

from external_baselines.snr_eq_jscc.reproduction import (
    SNREQConfig,
    SNRConditionedWindowAttention,
    correlation_penalty,
    snr_eq_loss,
)


class PenaltyTests(unittest.TestCase):
    def test_penalty_only_rejects_positive_correlation(self):
        snr = torch.tensor([-10.0, 0.0, 10.0, 20.0])
        self.assertAlmostEqual(float(correlation_penalty(snr, -snr)), 0.0)
        self.assertAlmostEqual(float(correlation_penalty(snr, snr)), 1.0, places=6)

    def test_loss_matches_paper_equation(self):
        mse = torch.tensor(0.25)
        snr = torch.tensor([-1.0, 1.0])
        total, parts = snr_eq_loss(mse, snr, snr, -snr, penalty_weight=10.0)
        self.assertAlmostEqual(float(parts["scale_penalty"]), 1.0)
        self.assertAlmostEqual(float(parts["bias_penalty"]), 0.0)
        self.assertAlmostEqual(float(total), 10.25)


class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.module = SNRConditionedWindowAttention(
            dim=32,
            window_size=(2, 2),
            num_heads=4,
            config=SNREQConfig(hidden_dim=8),
        )

    def test_shape_attention_probability_and_nonnegative_scale(self):
        features = torch.randn(6, 4, 32)
        output, aux = self.module(
            features,
            torch.tensor([1.0, 13.0]),
            batch_size=2,
            return_aux=True,
        )
        self.assertEqual(output.shape, features.shape)
        self.assertEqual(aux["attention"].shape, (6, 4, 4, 4))
        self.assertTrue(bool(torch.all(aux["query_scale"] >= 0)))
        row_sums = aux["attention"].sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6))
        self.assertTrue(torch.equal(aux["snr_per_window"], torch.tensor([1.0] * 3 + [13.0] * 3)))

    def test_shifted_window_mask_and_gradients(self):
        features = torch.randn(2, 4, 32, requires_grad=True)
        mask = torch.zeros(2, 4, 4)
        mask[:, 0, 3] = -100.0
        output, aux = self.module(features, torch.tensor([1.0, 13.0]), mask=mask, return_aux=True)
        output.square().mean().backward()
        self.assertEqual(output.shape, features.shape)
        self.assertIsNotNone(features.grad)
        self.assertGreater(float(features.grad.abs().sum()), 0.0)
        self.assertIsNotNone(self.module.query_scale.layers[0].weight.grad)
        self.assertIsNotNone(self.module.query_bias.layers[0].weight.grad)
        self.assertIsNotNone(self.module.snr_embedding.layers[0].weight.grad)
        self.assertEqual(aux["attention"].shape, (2, 4, 4, 4))

    def test_requires_explicit_batch_mapping(self):
        features = torch.randn(6, 4, 32)
        with self.assertRaisesRegex(ValueError, "provide scalar SNR"):
            self.module(features, torch.tensor([1.0, 13.0]))


if __name__ == "__main__":
    unittest.main()

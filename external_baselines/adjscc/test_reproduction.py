import unittest

import torch

from external_baselines.adjscc.reproduction import (
    ADJSCC,
    ADJSCCConfig,
    AdaptiveFeatureScaling,
    adaptive_awgn_channel,
)


class AttentionTests(unittest.TestCase):
    def test_attention_gate_shape_and_range(self):
        module = AdaptiveFeatureScaling(32)
        features = torch.randn(2, 32, 8, 8)
        scaled, gate = module(features, torch.tensor([1.0, 13.0]))
        self.assertEqual(scaled.shape, features.shape)
        self.assertEqual(gate.shape, (2, 32))
        self.assertTrue(bool(torch.all((gate >= 0) & (gate <= 1))))


class ChannelTests(unittest.TestCase):
    def test_channel_is_seed_deterministic(self):
        features = torch.randn(2, 12, 4, 4)
        first = adaptive_awgn_channel(features, torch.tensor([1.0, 13.0]), seed=42)
        second = adaptive_awgn_channel(features, torch.tensor([1.0, 13.0]), seed=42)
        third = adaptive_awgn_channel(features, torch.tensor([1.0, 13.0]), seed=43)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, third))

    def test_channel_rejects_odd_feature_count(self):
        with self.assertRaisesRegex(ValueError, "must be even"):
            adaptive_awgn_channel(torch.randn(1, 3, 3), 7.0, seed=42)


class ModelTests(unittest.TestCase):
    def test_reproduction_has_end_to_end_gradients(self):
        model = ADJSCC(ADJSCCConfig(transmit_channels=4)).train()
        image = torch.rand(1, 3, 16, 16)
        output = model(image, 7.0, noise_seed=42)
        loss = torch.mean((output["reconstruction"] - image).square())
        loss.backward()
        gradients = [
            model.encoder.layers[0].conv.weight.grad,
            model.encoder.attention[0].dense2.weight.grad,
            model.decoder.layers[4].conv.weight.grad,
        ]
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(bool(torch.all(torch.isfinite(gradient))))
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_paper_rate_model_shapes_and_budget(self):
        model = ADJSCC(ADJSCCConfig(transmit_channels=16)).eval()
        image = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            output = model(image, 7.0, noise_seed=42, return_aux=True)
        self.assertEqual(output["latent"].shape, (1, 16, 8, 8))
        self.assertEqual(output["reconstruction"].shape, image.shape)
        self.assertAlmostEqual(output["cbr"], 1 / 6)
        self.assertEqual(len(output["encoder_gates"]), 4)
        self.assertEqual(len(output["decoder_gates"]), 4)

    def test_local_one_eighth_rate_mapping(self):
        model = ADJSCC(ADJSCCConfig(transmit_channels=12)).eval()
        image = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            output = model(image, 7.0, noise_seed=42)
        self.assertAlmostEqual(output["cbr"], 1 / 8)


if __name__ == "__main__":
    unittest.main()

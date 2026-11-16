import unittest

import torch

from external_baselines.sa_ra_jscc.reproduction import (
    SARAConfig,
    SARALatentConditioner,
    ScalarConditionPreprocessor,
    hard_topk_channel_mask,
)


class PreprocessorTests(unittest.TestCase):
    def test_three_layer_condition_vector_shape_and_range(self):
        module = ScalarConditionPreprocessor(16)
        output = module(torch.tensor([1.0, 13.0]), batch_size=2)
        self.assertEqual(output.shape, (2, 16))
        self.assertTrue(bool(torch.all((output >= 0) & (output <= 1))))


class MaskTests(unittest.TestCase):
    def test_topk_mask_has_exact_budget(self):
        scores = torch.tensor([[0.1, 0.9, 0.3, 0.2], [4.0, 1.0, 3.0, 2.0]])
        mask = hard_topk_channel_mask(scores, 2)
        self.assertTrue(torch.equal(mask.sum(dim=1), torch.tensor([2.0, 2.0])))
        self.assertTrue(torch.equal(mask[0], torch.tensor([0.0, 1.0, 1.0, 0.0])))
        self.assertTrue(torch.equal(mask[1], torch.tensor([1.0, 0.0, 1.0, 0.0])))

    def test_same_scores_produce_nested_masks(self):
        scores = torch.randn(3, 16, generator=torch.Generator().manual_seed(42))
        small = hard_topk_channel_mask(scores, 4)
        large = hard_topk_channel_mask(scores, 9)
        self.assertTrue(bool(torch.all(small <= large)))


class ConditionerTests(unittest.TestCase):
    def test_snr_then_rate_residual_and_mask(self):
        module = SARALatentConditioner(SARAConfig(feature_dim=16, mlp_hidden_dim=8))
        features = torch.randn(2, 5, 16)
        scores = torch.randn(2, 16)
        output, aux = module(
            features,
            torch.tensor([1.0, 13.0]),
            torch.tensor([1 / 32, 1 / 8]),
            channel_scores=scores,
            active_channels=6,
        )
        self.assertEqual(output.shape, features.shape)
        self.assertEqual(aux["snr_residual"].shape, features.shape)
        self.assertEqual(aux["rate_residual"].shape, features.shape)
        self.assertTrue(torch.equal(aux["channel_mask"].sum(dim=1), torch.tensor([6.0, 6.0])))
        inactive = aux["channel_mask"][:, None, :] == 0
        self.assertTrue(bool(torch.all(output.masked_select(inactive.expand_as(output)) == 0)))

    def test_conditioning_modules_receive_gradients(self):
        module = SARALatentConditioner(SARAConfig(feature_dim=16, mlp_hidden_dim=8))
        features = torch.randn(2, 5, 16, requires_grad=True)
        output, _ = module(features, torch.tensor([1.0, 13.0]), torch.tensor([1 / 32, 1 / 8]))
        output.square().mean().backward()
        gradients = [
            module.snr_adapter.preprocessor.layers[0].weight.grad,
            module.snr_adapter.mlp[0].weight.grad,
            module.rate_adapter.preprocessor.layers[0].weight.grad,
            module.rate_adapter.mlp[0].weight.grad,
        ]
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_internal_rate_residual_scores_keep_exact_budget(self):
        module = SARALatentConditioner(SARAConfig(feature_dim=16, mlp_hidden_dim=8))
        output, aux = module(torch.randn(2, 3, 16), 7.0, 1 / 8, active_channels=4)
        self.assertEqual(output.shape, (2, 3, 16))
        self.assertEqual(aux["channel_scores"].shape, (2, 16))
        self.assertTrue(torch.equal(aux["channel_mask"].sum(dim=1), torch.tensor([4.0, 4.0])))

    def test_straight_through_mask_preserves_score_gradients(self):
        module = SARALatentConditioner(SARAConfig(feature_dim=16, mlp_hidden_dim=8))
        features = torch.randn(2, 3, 16, requires_grad=True)
        output, aux = module(features, 7.0, 1 / 8, active_channels=4)
        output.square().mean().backward()
        score_layer = module.rate_adapter.mlp[-1]
        self.assertIsNotNone(score_layer.weight.grad)
        self.assertGreater(float(score_layer.weight.grad.abs().sum()), 0.0)
        self.assertTrue(
            torch.allclose(aux["channel_mask_surrogate"], aux["channel_mask"], atol=1e-7)
        )


if __name__ == "__main__":
    unittest.main()

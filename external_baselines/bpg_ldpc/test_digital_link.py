import unittest

import torch

from external_baselines.bpg_ldpc.digital_link import (
    add_complex_awgn,
    append_crc24a,
    bits_to_bytes,
    bytes_to_bits,
    check_crc24a,
    crc24a,
    estimate_link_budget,
    qam_demodulate_logits,
    qam_modulate,
    run_5g_ldpc_awgn,
)
from external_baselines.bpg_ldpc.mcs import candidate_mcs


class BitPackingTests(unittest.TestCase):
    def test_bytes_round_trip(self):
        payload = bytes([0, 1, 127, 128, 254, 255])
        bits = bytes_to_bits(payload)
        self.assertEqual(bits.numel(), 8 * len(payload))
        self.assertEqual(bits_to_bytes(bits), payload)

    def test_rejects_non_bit_values(self):
        with self.assertRaisesRegex(ValueError, "only 0 and 1"):
            qam_modulate(torch.tensor([0, 1, 2, 0]), 2)

    def test_crc24a_round_trip_and_error_detection(self):
        payload = bytes_to_bits(b"DecompJSCC")
        protected = append_crc24a(payload)
        self.assertEqual(protected.numel(), payload.numel() + 24)
        self.assertTrue(check_crc24a(protected))
        corrupted = protected.clone()
        corrupted[7] ^= 1
        self.assertFalse(check_crc24a(corrupted))

    def test_crc24a_standard_check_vector(self):
        remainder = crc24a(bytes_to_bits(b"123456789"))
        value = 0
        for bit in remainder.tolist():
            value = (value << 1) | bit
        self.assertEqual(value, 0xCDE703)


class QamTests(unittest.TestCase):
    def test_budget_includes_crc_padding_and_modulation(self):
        budget = estimate_link_budget(
            173,
            information_block_bits=100,
            coded_block_bits=200,
            bits_per_symbol=6,
            crc_bits=24,
        )
        self.assertEqual(budget.num_code_blocks, 3)
        self.assertEqual(budget.information_padding_bits, 55)
        self.assertEqual(budget.total_crc_bits, 72)
        self.assertEqual(budget.coded_bits, 600)
        self.assertEqual(budget.modulation_padding_bits, 0)
        self.assertEqual(budget.channel_uses, 100)

    def test_noiseless_demapper_recovers_all_constellation_labels(self):
        for bits_per_symbol in (2, 4, 6, 8):
            order = 2**bits_per_symbol
            labels = torch.arange(order, dtype=torch.int64)
            shifts = torch.arange(bits_per_symbol - 1, -1, -1, dtype=torch.int64)
            bits = ((labels[:, None] >> shifts) & 1).reshape(-1).to(torch.uint8)
            symbols = qam_modulate(bits, bits_per_symbol)
            logits = qam_demodulate_logits(symbols, bits_per_symbol, 1e-4)
            recovered = (logits > 0).to(torch.uint8)
            self.assertTrue(torch.equal(recovered, bits))
            self.assertAlmostEqual(float(symbols.abs().square().mean()), 1.0, places=6)

    def test_awgn_is_seed_deterministic(self):
        symbols = torch.ones(32, dtype=torch.complex64)
        first, first_variance = add_complex_awgn(symbols, 7.0, seed=20260827)
        second, second_variance = add_complex_awgn(symbols, 7.0, seed=20260827)
        third, _ = add_complex_awgn(symbols, 7.0, seed=20260828)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, third))
        self.assertEqual(first_variance, second_variance)


class LdpcTests(unittest.TestCase):
    def test_all_calibration_candidates_have_exact_rate_and_qam_alignment(self):
        for mcs in candidate_mcs():
            self.assertEqual(mcs.information_block_bits, 8352)
            self.assertEqual(mcs.coded_block_bits % mcs.bits_per_symbol, 0)
            self.assertAlmostEqual(
                mcs.information_block_bits / mcs.coded_block_bits,
                mcs.code_rate,
            )

    def test_5g_ldpc_high_snr_round_trip_and_budget(self):
        try:
            import sionna  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("Sionna is not installed")

        payload = torch.randint(0, 2, (173,), generator=torch.Generator().manual_seed(42))
        result = run_5g_ldpc_awgn(
            payload,
            information_block_bits=100,
            coded_block_bits=200,
            bits_per_symbol=2,
            snr_db=20.0,
            seed=20260827,
            decoder_iterations=20,
        )
        self.assertTrue(torch.equal(result.decoded_bits.cpu(), payload.to(torch.uint8)))
        self.assertTrue(result.stats.decode_success)
        self.assertEqual(result.stats.bit_errors, 0)
        self.assertEqual(result.stats.num_code_blocks, 2)
        self.assertEqual(result.stats.information_padding_bits, 27)
        self.assertEqual(result.stats.coded_bits, 400)
        self.assertEqual(result.stats.modulation_padding_bits, 0)
        self.assertEqual(result.stats.channel_uses, 200)
        self.assertAlmostEqual(result.stats.nominal_code_rate, 0.5)
        self.assertAlmostEqual(result.stats.effective_payload_rate, 173 / 400)
        self.assertEqual(result.stats.crc_bits, 0)

    def test_5g_ldpc_crc24_transport_blocks(self):
        try:
            import sionna  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("Sionna is not installed")

        payload = torch.randint(0, 2, (173,), generator=torch.Generator().manual_seed(7))
        result = run_5g_ldpc_awgn(
            payload,
            information_block_bits=100,
            coded_block_bits=200,
            bits_per_symbol=2,
            snr_db=20.0,
            seed=20260827,
            decoder_iterations=20,
            crc_bits=24,
        )
        self.assertTrue(torch.equal(result.decoded_bits.cpu(), payload.to(torch.uint8)))
        self.assertTrue(result.stats.crc_pass)
        self.assertEqual(result.stats.crc_failed_blocks, 0)
        self.assertEqual(result.stats.num_code_blocks, 3)
        self.assertEqual(result.stats.total_crc_bits, 72)
        self.assertEqual(result.stats.information_padding_bits, 55)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from external_baselines.bpg_ldpc.codec import BPGCodec, search_bpg_qp
from external_baselines.bpg_ldpc.mcs import MCS, candidate_mcs


class MCSTests(unittest.TestCase):
    def test_frozen_candidate_grid(self):
        candidates = candidate_mcs()
        self.assertEqual(len(candidates), 16)
        self.assertEqual({item.bits_per_symbol for item in candidates}, {2, 4, 6, 8})
        self.assertEqual(candidates[0].coded_block_bits, 25056)


class CodecConfigurationTests(unittest.TestCase):
    def test_public_grid_defaults_use_current_paper_configuration(self):
        from external_baselines.bpg_ldpc import eval_bpg_ldpc_snr_rate_sweep as grid
        args = grid.build_parser().parse_args([])
        self.assertEqual((args.encoder_mode, args.decoder_iterations), (8, 5))
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(Path(args.mcs_calibration),
                         root / 'paper_data/updates/bpg_i5_unified_20261104/config/mcs.json')
        self.assertTrue(Path(args.mcs_calibration).is_file())
        self.assertEqual(Path(args.bpgenc), root / 'external_baselines/bpg_ldpc/bin/bpgenc')
        self.assertEqual(Path(args.bpgdec), root / 'external_baselines/bpg_ldpc/bin/bpgdec')
        metadata = BPGCodec(args.bpgenc, args.bpgdec, encoder_mode=args.encoder_mode).metadata()
        self.assertEqual(metadata['bpgenc_sha256'],
                         '1bbef095fde622c9fb5c2f908801866798147b59fbbde52b07c8a5d8fc856c5d')
        self.assertEqual(metadata['bpgdec_sha256'],
                         '098d7eb4caedc4959179bea2d3b01e2c0d7ab855112675eb407f42729e04b067')

    def test_summary_requires_explicit_result_inputs(self):
        from external_baselines.bpg_ldpc import summarize_bpg_comparison as summary
        parser = summary.build_parser()
        actions = {action.dest: action for action in parser._actions}
        self.assertTrue(actions['decomp_dir'].required)
        self.assertTrue(actions['bpg_dir'].required)
        args = parser.parse_args(['--decomp-dir', 'new/neural', '--bpg-dir', 'new/bpg'])
        self.assertEqual((args.decomp_dir, args.bpg_dir), ('new/neural', 'new/bpg'))

    def test_frozen_codec_metadata_is_explicit(self):
        codec = BPGCodec("/bin/true", "/bin/true")
        metadata = codec.metadata()
        self.assertEqual(metadata["chroma_format"], "420")
        self.assertEqual(metadata["color_space"], "ycbcr")
        self.assertEqual(metadata["bit_depth"], 8)
        self.assertTrue(metadata["complete_bpg_file_in_payload"])


class QPSearchTests(unittest.TestCase):
    def test_selects_lowest_quality_index_that_fits(self):
        mcs = MCS(2, 1, 2, information_block_bits=100)

        def encode(qp):
            return bytes(max(1, 100 - qp))

        result = search_bpg_qp(
            encode,
            max_channel_uses=600,
            mcs=mcs,
            minimum_qp=0,
            maximum_qp=51,
        )
        self.assertIsNotNone(result.qp)
        self.assertLessEqual(result.channel_uses, 600)
        previous = encode(result.qp - 1)
        from external_baselines.bpg_ldpc.digital_link import estimate_link_budget

        previous_budget = estimate_link_budget(
            len(previous) * 8,
            information_block_bits=mcs.information_block_bits,
            coded_block_bits=mcs.coded_block_bits,
            bits_per_symbol=mcs.bits_per_symbol,
            crc_bits=24,
        )
        self.assertGreater(previous_budget.channel_uses, 600)


class OutputProtectionTests(unittest.TestCase):
    def test_existing_outputs_are_rejected_before_compute(self):
        from external_baselines.bpg_ldpc import calibration
        from external_baselines.bpg_ldpc import eval_bpg_ldpc_snr_rate_sweep

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "frozen.json"
            output.write_text("frozen\n", encoding="utf-8")
            for module, extra_args in (
                (calibration, ["--snr-db", "1", "--run"]),
                (eval_bpg_ldpc_snr_rate_sweep, []),
            ):
                with self.subTest(module=module.__name__):
                    with patch("sys.argv", ["program", "--output", str(output), *extra_args]):
                        with self.assertRaises(FileExistsError):
                            module.main()
                    self.assertEqual(output.read_text(encoding="utf-8"), "frozen\n")

    def test_existing_reconstructions_are_rejected_before_compute(self):
        from external_baselines.bpg_ldpc import eval_bpg_ldpc_snr_rate_sweep

        for relative in ("original.png", "bpg_ldpc/rate32_snr4.png"):
            with self.subTest(path=relative), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                image = root / relative
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b"frozen image")
                arguments = [
                    "program", "--output", str(root / "new.json"),
                    "--save-image-id", "kodim13.png", "--rates", "32",
                    "--snr-db-values", "4", "--reconstruction-dir", str(root),
                ]
                with patch("sys.argv", arguments):
                    with self.assertRaisesRegex(FileExistsError, "Reconstruction"):
                        eval_bpg_ldpc_snr_rate_sweep.main()
                self.assertEqual(image.read_bytes(), b"frozen image")


if __name__ == "__main__":
    unittest.main()

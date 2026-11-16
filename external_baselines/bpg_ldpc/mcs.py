"""Independent random-bit MCS calibration for the BPG+5G LDPC baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

import torch

from external_baselines.bpg_ldpc.digital_link import run_5g_ldpc_awgn


@dataclass(frozen=True)
class MCS:
    bits_per_symbol: int
    code_rate_numerator: int
    code_rate_denominator: int
    # Largest shared 5G BG1 block for which all four target rates produce
    # lengths divisible by every configured QAM order in Sionna 2.0.1.
    information_block_bits: int = 8352

    @property
    def code_rate(self) -> float:
        return self.code_rate_numerator / self.code_rate_denominator

    @property
    def coded_block_bits(self) -> int:
        rate = Fraction(self.code_rate_numerator, self.code_rate_denominator)
        return self.information_block_bits * rate.denominator // rate.numerator

    @property
    def spectral_efficiency(self) -> float:
        return self.bits_per_symbol * self.code_rate

    @property
    def name(self) -> str:
        order = 2**self.bits_per_symbol
        return f"{order}QAM_R{self.code_rate_numerator}_{self.code_rate_denominator}"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "name": self.name,
            "code_rate": self.code_rate,
            "coded_block_bits": self.coded_block_bits,
            "spectral_efficiency": self.spectral_efficiency,
        }


def candidate_mcs() -> tuple[MCS, ...]:
    return tuple(
        MCS(bits_per_symbol, numerator, denominator)
        for bits_per_symbol in (2, 4, 6, 8)
        for numerator, denominator in ((1, 3), (1, 2), (2, 3), (3, 4))
    )


def calibrate_snr(
    snr_db: float,
    *,
    trials: int,
    target_bler: float,
    seed: int,
    decoder_iterations: int,
    device: str | torch.device,
) -> dict[str, Any]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0.0 <= target_bler <= 1.0:
        raise ValueError("target_bler must be within [0, 1]")
    rows = []
    for index, mcs in enumerate(candidate_mcs()):
        payload_per_block = mcs.information_block_bits - 24
        generator = torch.Generator(device="cpu").manual_seed(seed + index)
        payload = torch.randint(0, 2, (trials * payload_per_block,), generator=generator)
        result = run_5g_ldpc_awgn(
            payload,
            information_block_bits=mcs.information_block_bits,
            coded_block_bits=mcs.coded_block_bits,
            bits_per_symbol=mcs.bits_per_symbol,
            snr_db=snr_db,
            seed=seed + 1000 + index,
            decoder_iterations=decoder_iterations,
            crc_bits=24,
            device=device,
        )
        block_errors = result.stats.crc_failed_blocks
        rows.append(
            {
                **mcs.to_dict(),
                "trials": trials,
                "block_errors": block_errors,
                "bler": block_errors / trials,
                "bit_errors": result.stats.bit_errors,
                "ber": result.stats.bit_errors / result.stats.payload_bits,
            }
        )
    eligible = [row for row in rows if row["bler"] <= target_bler]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (row["spectral_efficiency"], -row["bits_per_symbol"]),
        )
        threshold_met = True
    else:
        selected = next(row for row in rows if row["name"] == "4QAM_R1_3")
        threshold_met = False
    return {
        "snr_db": float(snr_db),
        "target_bler": float(target_bler),
        "threshold_met": threshold_met,
        "selected_mcs": selected,
        "candidates": rows,
    }


def mcs_from_dict(payload: dict[str, Any]) -> MCS:
    return MCS(
        bits_per_symbol=int(payload["bits_per_symbol"]),
        code_rate_numerator=int(payload["code_rate_numerator"]),
        code_rate_denominator=int(payload["code_rate_denominator"]),
        information_block_bits=int(payload.get("information_block_bits", 8352)),
    )

"""Python subprocess wrapper and channel-budget search for official BPG."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
import tempfile
from typing import Callable

from PIL import Image

from external_baselines.bpg_ldpc.digital_link import estimate_link_budget
from external_baselines.bpg_ldpc.mcs import MCS


@dataclass(frozen=True)
class QPSearchResult:
    qp: int | None
    bitstream: bytes | None
    channel_uses: int | None
    trials: tuple[dict[str, int | bool], ...]


class BPGCodec:
    def __init__(
        self,
        bpgenc: str | Path,
        bpgdec: str | Path,
        *,
        encoder_mode: int = 9,
        chroma_format: str = "420",
        color_space: str = "ycbcr",
        bit_depth: int = 8,
    ):
        self.bpgenc = Path(bpgenc).resolve()
        self.bpgdec = Path(bpgdec).resolve()
        self.encoder_mode = int(encoder_mode)
        self.chroma_format = str(chroma_format)
        self.color_space = str(color_space)
        self.bit_depth = int(bit_depth)
        if not 1 <= self.encoder_mode <= 9:
            raise ValueError("encoder_mode must be within [1, 9]")
        if self.chroma_format not in {"420", "422", "444"}:
            raise ValueError("chroma_format must be 420, 422, or 444")
        if self.color_space not in {
            "ycbcr",
            "rgb",
            "ycgco",
            "ycbcr_bt709",
            "ycbcr_bt2020",
        }:
            raise ValueError("unsupported BPG color_space")
        if not 8 <= self.bit_depth <= 12:
            raise ValueError("bit_depth must be within [8, 12]")
        for path in (self.bpgenc, self.bpgdec):
            if not path.is_file():
                raise FileNotFoundError(path)
        self._cache: dict[tuple[str, int], bytes] = {}

    def encode(self, input_png: str | Path, qp: int) -> bytes:
        if not 0 <= int(qp) <= 51:
            raise ValueError("BPG qp must be within [0, 51]")
        source = Path(input_png).resolve()
        key = (str(source), int(qp))
        if key in self._cache:
            return self._cache[key]
        with tempfile.TemporaryDirectory(prefix="decompjscc_bpgenc_") as directory:
            output = Path(directory) / "image.bpg"
            subprocess.run(
                [
                    str(self.bpgenc),
                    "-q",
                    str(int(qp)),
                    "-m",
                    str(self.encoder_mode),
                    "-f",
                    self.chroma_format,
                    "-c",
                    self.color_space,
                    "-b",
                    str(self.bit_depth),
                    "-o",
                    str(output),
                    str(source),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            bitstream = output.read_bytes()
        self._cache[key] = bitstream
        return bitstream

    def decode(self, bitstream: bytes) -> Image.Image:
        with tempfile.TemporaryDirectory(prefix="decompjscc_bpgdec_") as directory:
            source = Path(directory) / "image.bpg"
            output = Path(directory) / "image.png"
            source.write_bytes(bitstream)
            subprocess.run(
                [str(self.bpgdec), "-o", str(output), str(source)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            with Image.open(output) as image:
                return image.convert("RGB").copy()

    def metadata(self) -> dict[str, str | int]:
        return {
            "version": "libbpg-0.9.8",
            "bpgenc": str(self.bpgenc),
            "bpgdec": str(self.bpgdec),
            "bpgenc_sha256": _sha256(self.bpgenc),
            "bpgdec_sha256": _sha256(self.bpgdec),
            "encoder_mode": self.encoder_mode,
            "chroma_format": self.chroma_format,
            "color_space": self.color_space,
            "bit_depth": self.bit_depth,
            "complete_bpg_file_in_payload": True,
        }


def search_bpg_qp(
    encode: Callable[[int], bytes],
    *,
    max_channel_uses: int,
    mcs: MCS,
    minimum_qp: int = 0,
    maximum_qp: int = 51,
) -> QPSearchResult:
    if max_channel_uses <= 0:
        raise ValueError("max_channel_uses must be positive")
    low = int(minimum_qp)
    high = int(maximum_qp)
    best: tuple[int, bytes, int] | None = None
    trials = []
    while low <= high:
        qp = (low + high) // 2
        bitstream = encode(qp)
        budget = estimate_link_budget(
            len(bitstream) * 8,
            information_block_bits=mcs.information_block_bits,
            coded_block_bits=mcs.coded_block_bits,
            bits_per_symbol=mcs.bits_per_symbol,
            crc_bits=24,
        )
        fits = budget.channel_uses <= max_channel_uses
        trials.append(
            {
                "qp": qp,
                "payload_bytes": len(bitstream),
                "channel_uses": budget.channel_uses,
                "fits": fits,
            }
        )
        if fits:
            best = (qp, bitstream, budget.channel_uses)
            high = qp - 1
        else:
            low = qp + 1
    if best is None:
        return QPSearchResult(None, None, None, tuple(trials))
    return QPSearchResult(best[0], best[1], best[2], tuple(trials))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

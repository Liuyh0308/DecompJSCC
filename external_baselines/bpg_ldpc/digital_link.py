"""Pure-Python orchestration for a 5G NR LDPC coded QAM/AWGN link."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from typing import Any

import torch


@dataclass(frozen=True)
class DigitalLinkStats:
    payload_bits: int
    transport_bits: int
    information_block_bits: int
    coded_block_bits: int
    num_code_blocks: int
    information_padding_bits: int
    coded_bits: int
    modulation_padding_bits: int
    bits_per_symbol: int
    channel_uses: int
    nominal_code_rate: float
    effective_payload_rate: float
    snr_db: float
    decoder_iterations: int
    bit_errors: int
    decode_success: bool
    crc_pass: bool
    crc_failed_blocks: int
    base_graph: str
    lifting_size: int
    crc_bits: int = 0
    total_crc_bits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DigitalLinkResult:
    decoded_bits: torch.Tensor
    stats: DigitalLinkStats


@dataclass(frozen=True)
class LinkBudget:
    payload_bits: int
    num_code_blocks: int
    information_padding_bits: int
    total_crc_bits: int
    coded_bits: int
    modulation_padding_bits: int
    channel_uses: int


def bytes_to_bits(payload: bytes, *, device: str | torch.device = "cpu") -> torch.Tensor:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not payload:
        return torch.empty(0, dtype=torch.uint8, device=device)
    values = torch.tensor(list(payload), dtype=torch.uint8, device=device)
    shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=device)
    return ((values[:, None] >> shifts) & 1).reshape(-1)


def bits_to_bytes(bits: torch.Tensor, *, valid_bits: int | None = None) -> bytes:
    flat = _validate_bits(bits)
    if valid_bits is None:
        valid_bits = int(flat.numel())
    if valid_bits < 0 or valid_bits > flat.numel():
        raise ValueError("valid_bits must be within the input bit length")
    if valid_bits % 8 != 0:
        raise ValueError("valid_bits must be divisible by 8 for byte reconstruction")
    flat = flat[:valid_bits].to(device="cpu", dtype=torch.uint8)
    if flat.numel() == 0:
        return b""
    groups = flat.reshape(-1, 8).to(torch.int64)
    weights = 2 ** torch.arange(7, -1, -1, dtype=torch.int64)
    values = torch.sum(groups * weights, dim=1).tolist()
    return bytes(values)


CRC24A_POLYNOMIAL = 0x1864CFB


def _crc24a_register(bits: torch.Tensor, *, append_zeros: bool) -> int:
    flat = _validate_bits(bits).to(device="cpu", dtype=torch.uint8)
    value = 0
    for bit in flat.tolist():
        value = (value << 1) | int(bit)
    if append_zeros:
        value <<= 24
    while value.bit_length() >= 25:
        value ^= CRC24A_POLYNOMIAL << (value.bit_length() - 25)
    return value


def crc24a(bits: torch.Tensor) -> torch.Tensor:
    """Return the 3GPP CRC24A remainder, MSB first."""
    register = _crc24a_register(bits, append_zeros=True)
    shifts = torch.arange(23, -1, -1, dtype=torch.int64)
    remainder = ((register >> shifts) & 1).to(torch.uint8)
    return remainder.to(bits.device)


def append_crc24a(bits: torch.Tensor) -> torch.Tensor:
    flat = _validate_bits(bits)
    return torch.cat([flat, crc24a(flat)])


def check_crc24a(bits_with_crc: torch.Tensor) -> bool:
    flat = _validate_bits(bits_with_crc)
    return flat.numel() >= 24 and _crc24a_register(flat, append_zeros=False) == 0


def qam_modulate(bits: torch.Tensor, bits_per_symbol: int) -> torch.Tensor:
    flat = _validate_bits(bits)
    _validate_bits_per_symbol(bits_per_symbol)
    if flat.numel() % bits_per_symbol != 0:
        raise ValueError("bit length must be divisible by bits_per_symbol")
    points, _ = _qam_constellation(bits_per_symbol, flat.device)
    groups = flat.reshape(-1, bits_per_symbol).to(torch.int64)
    weights = 2 ** torch.arange(
        bits_per_symbol - 1,
        -1,
        -1,
        dtype=torch.int64,
        device=flat.device,
    )
    indices = torch.sum(groups * weights, dim=1)
    return points[indices]


def qam_demodulate_logits(
    received: torch.Tensor,
    bits_per_symbol: int,
    noise_variance: float,
) -> torch.Tensor:
    _validate_bits_per_symbol(bits_per_symbol)
    if not torch.is_complex(received):
        raise TypeError("received symbols must be a complex tensor")
    if noise_variance <= 0:
        raise ValueError("noise_variance must be positive")
    points, labels = _qam_constellation(bits_per_symbol, received.device)
    metrics = -torch.abs(received.reshape(-1, 1) - points.reshape(1, -1)).square()
    metrics = metrics / float(noise_variance)
    logits = []
    for bit_index in range(bits_per_symbol):
        one = torch.logsumexp(metrics[:, labels[:, bit_index] == 1], dim=1)
        zero = torch.logsumexp(metrics[:, labels[:, bit_index] == 0], dim=1)
        logits.append(one - zero)
    return torch.stack(logits, dim=1).reshape(-1)


def add_complex_awgn(
    symbols: torch.Tensor,
    snr_db: float,
    *,
    seed: int,
) -> tuple[torch.Tensor, float]:
    if not torch.is_complex(symbols):
        raise TypeError("symbols must be a complex tensor")
    noise_variance = 10.0 ** (-float(snr_db) / 10.0)
    component_std = math.sqrt(noise_variance / 2.0)
    generator = torch.Generator(device=symbols.device)
    generator.manual_seed(int(seed))
    real = torch.randn(
        symbols.shape,
        dtype=symbols.real.dtype,
        device=symbols.device,
        generator=generator,
    )
    imag = torch.randn(
        symbols.shape,
        dtype=symbols.real.dtype,
        device=symbols.device,
        generator=generator,
    )
    noise = torch.complex(real, imag) * component_std
    return symbols + noise, noise_variance


def estimate_link_budget(
    payload_bits: int,
    *,
    information_block_bits: int,
    coded_block_bits: int,
    bits_per_symbol: int,
    crc_bits: int = 24,
) -> LinkBudget:
    if payload_bits <= 0:
        raise ValueError("payload_bits must be positive")
    if information_block_bits <= crc_bits or coded_block_bits <= information_block_bits:
        raise ValueError("invalid information/coded block sizes")
    if crc_bits not in {0, 24}:
        raise ValueError("crc_bits must be 0 or 24")
    _validate_bits_per_symbol(bits_per_symbol)
    payload_per_block = information_block_bits - crc_bits
    num_blocks = math.ceil(payload_bits / payload_per_block)
    information_padding = num_blocks * payload_per_block - payload_bits
    coded_bits = num_blocks * coded_block_bits
    modulation_padding = (-coded_bits) % bits_per_symbol
    return LinkBudget(
        payload_bits=int(payload_bits),
        num_code_blocks=int(num_blocks),
        information_padding_bits=int(information_padding),
        total_crc_bits=int(num_blocks * crc_bits),
        coded_bits=int(coded_bits),
        modulation_padding_bits=int(modulation_padding),
        channel_uses=int((coded_bits + modulation_padding) // bits_per_symbol),
    )


@lru_cache(maxsize=32)
def _build_ldpc_modules(
    information_block_bits: int,
    coded_block_bits: int,
    bits_per_symbol: int,
    decoder_iterations: int,
    device: str,
):
    try:
        from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Sionna is required; install external_baselines/bpg_ldpc/"
            "requirements-sionna.txt"
        ) from exc
    encoder = LDPC5GEncoder(
        information_block_bits,
        coded_block_bits,
        num_bits_per_symbol=bits_per_symbol,
        device=device,
    )
    decoder = LDPC5GDecoder(
        encoder,
        hard_out=True,
        return_infobits=True,
        num_iter=decoder_iterations,
        device=device,
    )
    return encoder, decoder


def run_5g_ldpc_awgn(
    payload_bits: torch.Tensor,
    *,
    information_block_bits: int,
    coded_block_bits: int,
    bits_per_symbol: int,
    snr_db: float,
    seed: int,
    decoder_iterations: int = 20,
    crc_bits: int = 0,
    device: str | torch.device = "cpu",
) -> DigitalLinkResult:
    """Transmit arbitrary payload bits over a rate-matched 5G LDPC link."""
    bits = _validate_bits(payload_bits).to(device=device, dtype=torch.uint8)
    if bits.numel() == 0:
        raise ValueError("payload_bits must not be empty")
    if information_block_bits <= 0 or coded_block_bits <= information_block_bits:
        raise ValueError("require 0 < information_block_bits < coded_block_bits")
    if crc_bits not in {0, 24}:
        raise ValueError("crc_bits must be 0 or 24")
    if information_block_bits <= crc_bits:
        raise ValueError("information_block_bits must exceed crc_bits")
    if decoder_iterations <= 0:
        raise ValueError("decoder_iterations must be positive")
    _validate_bits_per_symbol(bits_per_symbol)

    encoder, decoder = _build_ldpc_modules(
        information_block_bits,
        coded_block_bits,
        bits_per_symbol,
        decoder_iterations,
        str(device),
    )

    payload_per_block = information_block_bits - crc_bits
    num_blocks = math.ceil(bits.numel() / payload_per_block)
    information_padding = num_blocks * payload_per_block - bits.numel()
    padded_payload = torch.nn.functional.pad(bits, (0, information_padding))
    payload_words = padded_payload.reshape(num_blocks, payload_per_block)
    if crc_bits:
        information_words = torch.stack([append_crc24a(word) for word in payload_words])
    else:
        information_words = payload_words
    codewords = encoder(information_words.to(torch.float32))
    coded = codewords.reshape(-1).to(torch.uint8)

    modulation_padding = (-coded.numel()) % bits_per_symbol
    modulation_bits = torch.nn.functional.pad(coded, (0, modulation_padding))
    transmitted = qam_modulate(modulation_bits, bits_per_symbol)
    received, noise_variance = add_complex_awgn(transmitted, snr_db, seed=seed)
    logits = qam_demodulate_logits(received, bits_per_symbol, noise_variance)
    if modulation_padding:
        logits = logits[:-modulation_padding]
    decoded_words = decoder(logits.reshape(num_blocks, coded_block_bits)).round().to(torch.uint8)
    if crc_bits:
        crc_checks = [check_crc24a(word) for word in decoded_words]
        decoded_payload = decoded_words[:, :payload_per_block]
    else:
        crc_checks = [True] * num_blocks
        decoded_payload = decoded_words
    decoded = decoded_payload.reshape(-1)[: bits.numel()]
    bit_errors = int(torch.count_nonzero(decoded != bits).item())
    crc_failed_blocks = sum(not passed for passed in crc_checks)

    coded_bits = int(coded.numel())
    channel_uses = int(transmitted.numel())
    estimated = estimate_link_budget(
        int(bits.numel()),
        information_block_bits=information_block_bits,
        coded_block_bits=coded_block_bits,
        bits_per_symbol=bits_per_symbol,
        crc_bits=crc_bits,
    )
    if channel_uses != estimated.channel_uses:
        raise RuntimeError("measured channel uses do not match the deterministic budget")
    stats = DigitalLinkStats(
        payload_bits=int(bits.numel()),
        transport_bits=int(information_words.numel()),
        information_block_bits=int(information_block_bits),
        coded_block_bits=int(coded_block_bits),
        num_code_blocks=int(num_blocks),
        information_padding_bits=int(information_padding),
        coded_bits=coded_bits,
        modulation_padding_bits=int(modulation_padding),
        bits_per_symbol=int(bits_per_symbol),
        channel_uses=channel_uses,
        nominal_code_rate=float(encoder.coderate),
        effective_payload_rate=float(bits.numel() / coded_bits),
        snr_db=float(snr_db),
        decoder_iterations=int(decoder_iterations),
        bit_errors=bit_errors,
        decode_success=bit_errors == 0 and crc_failed_blocks == 0,
        crc_pass=crc_failed_blocks == 0,
        crc_failed_blocks=int(crc_failed_blocks),
        base_graph=str(getattr(encoder, "_bg", "unknown")),
        lifting_size=int(getattr(encoder, "_z", -1)),
        crc_bits=int(crc_bits),
        total_crc_bits=int(num_blocks * crc_bits),
    )
    return DigitalLinkResult(decoded_bits=decoded, stats=stats)


def _validate_bits(bits: torch.Tensor) -> torch.Tensor:
    if not isinstance(bits, torch.Tensor):
        raise TypeError("bits must be a torch.Tensor")
    flat = bits.reshape(-1)
    if flat.numel() and not bool(torch.all((flat == 0) | (flat == 1))):
        raise ValueError("bits must contain only 0 and 1")
    return flat.to(torch.uint8)


def _validate_bits_per_symbol(bits_per_symbol: int) -> None:
    if bits_per_symbol not in {2, 4, 6, 8}:
        raise ValueError("bits_per_symbol must be one of 2, 4, 6, or 8")


def _qam_constellation(
    bits_per_symbol: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_bits_per_symbol(bits_per_symbol)
    order = 2**bits_per_symbol
    labels_as_int = torch.arange(order, dtype=torch.int64, device=device)
    shifts = torch.arange(
        bits_per_symbol - 1,
        -1,
        -1,
        dtype=torch.int64,
        device=device,
    )
    labels = ((labels_as_int[:, None] >> shifts) & 1).to(torch.uint8)
    axis_bits = bits_per_symbol // 2
    i_gray = _bits_to_integer(labels[:, :axis_bits])
    q_gray = _bits_to_integer(labels[:, axis_bits:])
    i_index = _gray_to_binary(i_gray, axis_bits)
    q_index = _gray_to_binary(q_gray, axis_bits)
    levels_per_axis = 2**axis_bits
    i_level = 2 * i_index - (levels_per_axis - 1)
    q_level = 2 * q_index - (levels_per_axis - 1)
    normalization = math.sqrt(2.0 * (order - 1) / 3.0)
    points = torch.complex(i_level.to(torch.float32), q_level.to(torch.float32))
    return points / normalization, labels


def _bits_to_integer(bits: torch.Tensor) -> torch.Tensor:
    width = bits.shape[1]
    weights = 2 ** torch.arange(width - 1, -1, -1, dtype=torch.int64, device=bits.device)
    return torch.sum(bits.to(torch.int64) * weights, dim=1)


def _gray_to_binary(gray: torch.Tensor, width: int) -> torch.Tensor:
    binary = gray.clone()
    shift = 1
    while shift < width:
        binary = binary ^ (binary >> shift)
        shift <<= 1
    return binary

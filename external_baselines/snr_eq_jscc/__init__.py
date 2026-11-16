"""Paper-based SNR-EQ-JSCC reproduction components."""

from .reproduction import (
    SNREQConfig,
    SNRConditionedWindowAttention,
    correlation_penalty,
    snr_eq_loss,
)

__all__ = [
    "SNREQConfig",
    "SNRConditionedWindowAttention",
    "correlation_penalty",
    "snr_eq_loss",
]

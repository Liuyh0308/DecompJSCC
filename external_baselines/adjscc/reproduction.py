"""PyTorch reproduction of the published ADJSCC image model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ADJSCCConfig:
    transmit_channels: int = 16

    @property
    def bandwidth_ratio(self) -> float:
        return self.transmit_channels / 96.0


class GeneralizedDivisiveNormalization(nn.Module):
    def __init__(self, channels: int, *, inverse: bool = False):
        super().__init__()
        self.inverse = inverse
        self.beta_reparam = nn.Parameter(torch.ones(channels))
        gamma = torch.zeros(channels, channels)
        gamma.fill_diagonal_(math.sqrt(0.1))
        self.gamma_reparam = nn.Parameter(gamma)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        beta = 1e-6 + self.beta_reparam.square()
        gamma = self.gamma_reparam.square().unsqueeze(-1).unsqueeze(-1)
        norm = F.conv2d(inputs.square(), gamma, beta).clamp_min(1e-12).sqrt()
        return inputs * norm if self.inverse else inputs / norm


class AdaptiveFeatureScaling(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 16, 1)
        self.dense1 = nn.Linear(channels + 1, hidden)
        self.dense2 = nn.Linear(hidden, channels)

    def forward(
        self,
        features: torch.Tensor,
        snr_db: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        snr = _snr_column(snr_db, features.shape[0], features.device, features.dtype)
        pooled = features.mean(dim=(2, 3))
        gate = torch.sigmoid(self.dense2(F.relu(self.dense1(torch.cat([pooled, snr], dim=1)))))
        return features * gate[:, :, None, None], gate


class EncoderLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.gdn = GeneralizedDivisiveNormalization(out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.gdn(self.conv(inputs))


class DecoderLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        if stride == 1:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                output_padding=stride - 1,
            )
        self.igdn = GeneralizedDivisiveNormalization(out_channels, inverse=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.igdn(self.conv(inputs))


class ADJSCCEncoder(nn.Module):
    def __init__(self, transmit_channels: int):
        super().__init__()
        channels = [3, 256, 256, 256, 256, transmit_channels]
        kernels = [9, 5, 5, 5, 5]
        strides = [2, 2, 1, 1, 1]
        self.layers = nn.ModuleList(
            EncoderLayer(channels[index], channels[index + 1], kernels[index], strides[index])
            for index in range(5)
        )
        self.prelu = nn.ModuleList(nn.PReLU(256) for _ in range(4))
        self.attention = nn.ModuleList(AdaptiveFeatureScaling(256) for _ in range(4))

    def forward(
        self,
        image: torch.Tensor,
        snr_db: torch.Tensor | float,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = image
        gates = []
        for index in range(4):
            features = self.prelu[index](self.layers[index](features))
            features, gate = self.attention[index](features, snr_db)
            gates.append(gate)
        return self.layers[4](features), gates


class ADJSCCDecoder(nn.Module):
    def __init__(self, transmit_channels: int):
        super().__init__()
        channels = [transmit_channels, 256, 256, 256, 256, 3]
        kernels = [5, 5, 5, 5, 9]
        strides = [1, 1, 1, 2, 2]
        self.layers = nn.ModuleList(
            DecoderLayer(channels[index], channels[index + 1], kernels[index], strides[index])
            for index in range(5)
        )
        self.prelu = nn.ModuleList(nn.PReLU(256) for _ in range(4))
        self.attention = nn.ModuleList(AdaptiveFeatureScaling(256) for _ in range(4))

    def forward(
        self,
        features: torch.Tensor,
        snr_db: torch.Tensor | float,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        gates = []
        for index in range(4):
            features = self.prelu[index](self.layers[index](features))
            features, gate = self.attention[index](features, snr_db)
            gates.append(gate)
        return torch.sigmoid(self.layers[4](features)), gates


class ADJSCC(nn.Module):
    def __init__(
        self,
        config: ADJSCCConfig | None = None,
        *,
        channel: nn.Module | None = None,
    ):
        super().__init__()
        self.config = config or ADJSCCConfig()
        self.encoder = ADJSCCEncoder(self.config.transmit_channels)
        self.decoder = ADJSCCDecoder(self.config.transmit_channels)
        self.channel = channel

    def forward(
        self,
        image: torch.Tensor,
        snr_db: torch.Tensor | float,
        *,
        noise_seed: int | None = None,
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor | float | list[torch.Tensor]]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [batch, 3, height, width]")
        latent, encoder_gates = self.encoder(image, snr_db)
        if self.channel is None:
            noisy_latent = adaptive_awgn_channel(latent, snr_db, seed=noise_seed)
        else:
            if noise_seed is not None:
                torch.manual_seed(int(noise_seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(noise_seed))
            noisy_latent = self.channel(latent, snr_db)
        reconstruction, decoder_gates = self.decoder(noisy_latent, snr_db)
        cbr = latent[0].numel() / 2.0 / image[0].numel()
        output: dict[str, torch.Tensor | float | list[torch.Tensor]] = {
            "reconstructed": reconstruction,
            "reconstruction": reconstruction,
            "cbr": float(cbr),
            "mse": F.mse_loss(reconstruction * 255.0, image * 255.0),
            "loss": F.mse_loss(reconstruction * 255.0, image * 255.0),
        }
        if return_aux:
            output.update(
                {
                    "latent": latent,
                    "noisy_latent": noisy_latent,
                    "encoder_gates": encoder_gates,
                    "decoder_gates": decoder_gates,
                }
            )
        return output


def adaptive_awgn_channel(
    features: torch.Tensor,
    snr_db: torch.Tensor | float,
    *,
    seed: int | None = None,
) -> torch.Tensor:
    if features.ndim < 2:
        raise ValueError("features must include batch and feature dimensions")
    flat = features.reshape(features.shape[0], -1)
    if flat.shape[1] % 2:
        raise ValueError("the per-sample feature count must be even")
    complex_dim = flat.shape[1] // 2
    symbols = torch.complex(flat[:, :complex_dim], flat[:, complex_dim:])
    energy = symbols.abs().square().sum(dim=1, keepdim=True).clamp_min(1e-12)
    symbols = symbols * torch.sqrt(torch.tensor(complex_dim, device=features.device) / energy)
    snr = _snr_column(snr_db, features.shape[0], features.device, features.dtype)
    component_std = torch.sqrt(10.0 ** (-snr / 10.0) / 2.0)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=features.device)
        generator.manual_seed(int(seed))
    real_noise = torch.randn(
        symbols.shape,
        dtype=features.dtype,
        device=features.device,
        generator=generator,
    )
    imag_noise = torch.randn(
        symbols.shape,
        dtype=features.dtype,
        device=features.device,
        generator=generator,
    )
    noisy = symbols + torch.complex(real_noise, imag_noise) * component_std
    restored = torch.cat([noisy.real, noisy.imag], dim=1)
    return restored.reshape_as(features)


def _snr_column(
    snr_db: torch.Tensor | float,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    snr = torch.as_tensor(snr_db, dtype=dtype, device=device)
    if snr.numel() == 1:
        return snr.reshape(1, 1).expand(batch_size, 1)
    if snr.numel() != batch_size:
        raise ValueError("snr_db must be scalar or contain one value per sample")
    return snr.reshape(batch_size, 1)

"""
Exponential Moving Average (EMA) of model weights.

EMA weights tend to produce significantly better samples than the raw
training weights for diffusion models — worth the small memory overhead
of keeping a second copy.

Usage
-----
    ema = EMA(unet, decay=0.9999, update_after_step=100)

    # inside training loop, after optimizer.step():
    ema.step(unet)

    # for validation / sampling, swap EMA weights in temporarily:
    with ema.average_parameters(unet):
        samples = pipeline(...)

    # or save EMA weights separately:
    ema.save(unet, "ema_unet.pt")
"""

import copy
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn


class EMA:
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 100,
    ):
        """
        Parameters
        ----------
        model             : the model whose weights will be tracked
        decay             : EMA coefficient; higher = slower update
        update_after_step : skip EMA updates for this many steps to let
                            the model warm up first
        """
        self.decay = decay
        self.update_after_step = update_after_step
        self.step_count = 0

        # Shadow copy — kept on same device as model, no gradients
        self.shadow: nn.Module = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def step(self, model: nn.Module) -> None:
        """Update shadow weights from current model weights."""
        self.step_count += 1
        if self.step_count < self.update_after_step:
            # Before warm-up: just copy weights directly
            self._copy_weights(model)
            return

        for shadow_param, model_param in zip(
            self.shadow.parameters(), model.parameters()
        ):
            shadow_param.data.mul_(self.decay).add_(
                model_param.data, alpha=1.0 - self.decay
            )

        # Also handle buffers (e.g. BatchNorm running stats)
        for shadow_buf, model_buf in zip(
            self.shadow.buffers(), model.buffers()
        ):
            shadow_buf.copy_(model_buf)

    def _copy_weights(self, model: nn.Module) -> None:
        for shadow_param, model_param in zip(
            self.shadow.parameters(), model.parameters()
        ):
            shadow_param.data.copy_(model_param.data)

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """
        Context manager that temporarily loads EMA weights into `model`
        for inference, then restores the original weights on exit.

        Example
        -------
            with ema.average_parameters(unet):
                pred = unet(latents, t, encoder_hidden_states)
        """
        # Save original weights
        original_state = copy.deepcopy(model.state_dict())
        # Load EMA weights
        model.load_state_dict(self.shadow.state_dict())
        try:
            yield
        finally:
            # Restore original weights
            model.load_state_dict(original_state)

    def save(self, path: str) -> None:
        """Save EMA shadow weights to disk."""
        torch.save(self.shadow.state_dict(), path)

    def load(self, path: str, map_location=None) -> None:
        """Load EMA shadow weights from disk."""
        self.shadow.load_state_dict(
            torch.load(path, map_location=map_location)
        )

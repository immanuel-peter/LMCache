# SPDX-License-Identifier: Apache-2.0
"""Google TPU (TorchTPU) platform helpers.

Targets TorchTPU (``torch_tpu``), which registers ``torch.device("tpu")`` via
PyTorch's ``rename_privateuse1_backend`` mechanism, together with the
``torch.tpu`` device module.

Scope: **engine-driven** multiprocess transfer only. A TorchTPU tensor's
``data_ptr()`` addresses host-side buffer metadata rather than HBM, so neither
pointer-based kernels nor the LMCache-driven IPC handle path can address it;
:class:`~lmcache.v1.platform.devices.tpu.device_ops.TpuDeviceOps` moves whole
blocks with tensor ops through host staging instead.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any

# Third Party
import torch

# First Party
from lmcache.v1.platform.base.device_spec import DeviceSpec

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.device_ops import DeviceOps


class _TpuDeviceModule:
    """``torch.tpu`` as LMCache's generic ``torch_dev`` call sites need it.

    On PyTorch >= 2.11 TorchTPU raises ``RuntimeError`` when code outside torch
    calls the device-module functions below and points to ``torch.accelerator``.
    They cannot be replaced on ``torch.tpu`` itself:
    ``torch.accelerator.is_available`` and ``device_count`` delegate back to the
    device module, so a replacement that calls them recurses. This adapter
    leaves ``torch.tpu`` as TorchTPU installed it and forwards every other
    attribute to it.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(torch.tpu, name)  # type: ignore[attr-defined]

    def is_available(self) -> bool:
        """``torch.tpu.is_available()`` through ``torch.accelerator``."""
        return torch.accelerator.is_available()

    def device_count(self) -> int:
        """``torch.tpu.device_count()`` through ``torch.accelerator``."""
        return torch.accelerator.device_count()

    def synchronize(self, device: int | str | torch.device | None = None) -> None:
        """``torch.tpu.synchronize()`` through ``torch.accelerator``."""
        torch.accelerator.synchronize(device)

    def current_device(self) -> int:
        """``torch.tpu.current_device()`` through ``torch.accelerator``."""
        return torch.accelerator.current_device_index()

    def set_device(self, device: int | str | torch.device) -> None:
        """``torch.tpu.set_device()`` through ``torch.accelerator``."""
        torch.accelerator.set_device_index(device)

    def current_stream(
        self, device: int | str | torch.device | None = None
    ) -> torch.Stream:
        """``torch.tpu.current_stream()`` through ``torch.accelerator``."""
        return torch.accelerator.current_stream(device)


class TpuDeviceSpec(DeviceSpec):
    """TorchTPU device specification for the detection registry."""

    @property
    def device_type(self) -> str:
        return "tpu"

    @property
    def torch_module_name(self) -> str:
        return "tpu"

    @property
    def ops_cls(self) -> type[DeviceOps]:
        # First Party
        from lmcache.v1.platform.devices.tpu.device_ops import TpuDeviceOps

        return TpuDeviceOps

    def is_available(self) -> bool:
        """Check TorchTPU availability.

        Imports ``torch_tpu``, which registers the ``tpu`` device on a TPU
        host unless PyTorch's device-backend autoload already has. Availability
        is then read through ``torch.accelerator`` rather than
        ``torch.tpu.is_available()``: on PyTorch >= 2.11 TorchTPU reserves the
        ``torch.tpu`` device-module functions for the framework and raises
        ``RuntimeError`` when other packages call them.

        Returns:
            bool: ``True`` when ``torch_tpu`` imports and the current
            accelerator is an available TPU, ``False`` otherwise.
        """
        try:
            # Third Party
            import torch_tpu  # noqa: F401
        except ImportError:
            return False

        accelerator = torch.accelerator.current_accelerator(check_available=True)
        return accelerator is not None and accelerator.type == "tpu"

    def resolve_torch_module(self, torch: Any) -> Any | None:
        """Return an adapter over ``torch.tpu`` instead of the module itself.

        The adapter routes the functions TorchTPU reserves for the framework
        through ``torch.accelerator``, so the generic ``torch_dev.<method>()``
        call sites throughout LMCache work on TPU without a TPU-specific
        branch.

        Args:
            torch: The imported ``torch`` package.

        Returns:
            Any | None: The adapter, or ``None`` when ``torch.tpu`` is not
            registered.
        """
        if getattr(torch, "tpu", None) is None:
            return None
        return _TpuDeviceModule()

    def is_async_engine_driven_store_available(self) -> bool:
        """Report that TorchTPU KV tensors cannot be gathered off-thread.

        vllm-torchtpu donates the KV buffers to each compiled forward step, so
        a gather racing a step fails with ``Buffer has been deleted or
        donated``.

        Returns:
            bool: Always ``False``.
        """
        return False

    def is_lmcache_driven_available(self) -> bool:
        """Report that TorchTPU cannot serve the LMCache-driven path.

        That path ships KV tensors to the server as IPC handles to device
        memory, and a TorchTPU tensor exposes no address of its HBM buffer.

        Returns:
            bool: Always ``False``.
        """
        return False

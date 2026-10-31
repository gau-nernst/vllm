# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dedicated storage for a PP stage's outgoing intermediate tensors.

An NCCL send completes only when the next stage posts its receive, and the
sender's next forward would overwrite the outputs being sent (CUDA graph
outputs are static; relayed aux states alias the receive buffer). Sending this
snapshot moves the wait to the next copy into the buffer.
"""

from math import prod
from typing import Any

import torch

from vllm.distributed.parallel_state import GroupCoordinator, Handle
from vllm.utils.math_utils import round_up


def _layout(tensors: dict[str, Any]) -> tuple[dict[str, int], int]:
    """Aligned byte offset of each tensor in the buffer, and the bytes used."""
    offsets: dict[str, int] = {}
    end = 0
    for key, value in tensors.items():
        if isinstance(value, torch.Tensor):
            offsets[key] = round_up(end, 256)
            end = offsets[key] + value.nbytes
    return offsets, end


def packed_nbytes(tensors: dict[str, Any]) -> int:
    """Buffer bytes needed to send `tensors`."""
    return _layout(tensors)[1]


def unpack_flat(tensors: dict[str, Any]) -> dict[str, Any]:
    """Eval only: rebuild per-key tensor views from a flat send's payload."""
    flat = tensors.pop("__flat__")
    schema = tensors.pop("__schema__")
    for key, offset, shape, dtype_str in schema:
        dtype = getattr(torch, dtype_str.removeprefix("torch."))
        nbytes = prod(shape) * dtype.itemsize
        tensors[key] = flat[offset : offset + nbytes].view(dtype).view(shape)
    return tensors


class PPSendBuffer:
    def __init__(
        self, nbytes: int, device: torch.device, depth: int = 1, flat: bool = False
    ) -> None:
        # Eval only: depth > 1 rotates through that many buffers.
        # Eval only: flat sends the packed buffer as one tensor plus a schema.
        self.flat = flat
        self.buffers = [
            torch.empty(nbytes, dtype=torch.uint8, device=device) for _ in range(depth)
        ]
        self.buffer = self.buffers[0]
        self._works: list[list[Handle]] = [[] for _ in range(depth)]
        self._slot = 0

    def send(
        self,
        tensors: dict[str, Any],
        pp_group: GroupCoordinator,
        all_gather_group: GroupCoordinator | None,
    ) -> None:
        offsets, nbytes = _layout(tensors)
        devices = {tensors[k].device for k in offsets}
        if devices - {self.buffer.device}:
            raise RuntimeError(
                f"PP send tensors on {devices}, send buffer on {self.buffer.device}"
            )
        if nbytes > self.buffer.numel():
            shapes = {k: tuple(v.shape) for k, v in tensors.items() if k in offsets}
            raise RuntimeError(
                f"PP send of {nbytes} bytes {shapes} exceeds the "
                f"{self.buffer.numel()}-byte send buffer sized from the profile run"
            )
        # The previous snapshot's sends must have read the buffer. This is a
        # stream wait, so the CPU keeps going.
        i = self._slot
        self._slot = (i + 1) % len(self.buffers)
        buffer = self.buffers[i]
        for handle in self._works[i]:
            handle.wait()
        if self.flat and offsets:
            # Eval only: one send of the packed bytes; the schema rides the
            # metadata channel so the receiver can rebuild views.
            schema = []
            for key in offsets:
                value = tensors[key]
                schema.append((key, offsets[key], tuple(value.shape), str(value.dtype)))
                view = buffer[offsets[key] : offsets[key] + value.nbytes]
                view.view(value.dtype).view(value.shape).copy_(value)
            staged = {"__flat__": buffer[:nbytes], "__schema__": schema}
            for key, value in tensors.items():
                if key not in offsets:
                    staged[key] = value
        else:
            staged = {}
            for key, value in tensors.items():
                if key not in offsets:
                    staged[key] = value
                    continue
                start = offsets[key]
                view = buffer[start : start + value.nbytes]
                staged[key] = view.view(value.dtype).view(value.shape)
                staged[key].copy_(value)
        handles = pp_group.isend_tensor_dict(staged, all_gather_group=all_gather_group)
        # The metadata handle is reaped by the group coordinator.
        self._works[i] = handles[1:]

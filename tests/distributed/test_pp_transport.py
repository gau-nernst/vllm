# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP transports carry every step's tensors intact through a PP chain, with
varying sizes, stages that take turns lagging behind, and middle stages that
relay a received tensor as is."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import ray
import torch

from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.distributed.pp_transport import PPTransport
from vllm.distributed.pp_transport.copy_engine import CopyEngineTransport
from vllm.distributed.pp_transport.nccl import NcclTransport
from vllm.distributed.pp_transport.sdma import SdmaTransport
from vllm.platforms import current_platform

from ..utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)

MAX_ROWS, WIDTH, STEPS = 4096, 1024, 40
KEYS = ("hidden_states", "residual")


def _rows(step: int) -> int:
    return [MAX_ROWS, 1, 1337, 512][step % 4]


def _run_chain(
    make: Callable[[Any, dict | None, dict | None], PPTransport],
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    torch.accelerator.set_device_index(rank)
    init_test_distributed_environment(1, pp_size, rank, distributed_init_port)
    pp_group = get_pp_group()
    first, last = pp_group.is_first_rank, pp_group.is_last_rank

    def buffers() -> dict[str, torch.Tensor]:
        return {
            k: torch.zeros(MAX_ROWS, WIDTH, dtype=torch.bfloat16, device="cuda")
            for k in KEYS
        }

    persistent = None if first else buffers()
    profile_send = None if last else buffers()
    transport = make(pp_group, persistent, profile_send)

    lag = torch.randn(4096, 4096, device="cuda")
    for step in range(STEPS):
        transport.before_forward()
        # Senders often lag (a receiver reading too early sees stale values),
        # the last stage sometimes (a sender overwriting too early corrupts
        # values it has not read yet).
        if (step % 2 == 0) if not last else (step % 7 == 3):
            for _ in range(20):
                lag @ lag
        if first:
            h = torch.full(
                (_rows(step), WIDTH), step % 100, dtype=torch.bfloat16, device="cuda"
            )
            transport.send({"hidden_states": h, "residual": -h}, pp_group)
            continue
        received = transport.recv(pp_group).tensors
        if not last:
            # The residual is relayed as is, e.g. a view of a receive buffer.
            out = dict(received, hidden_states=received["hidden_states"] + 1)
            transport.send(out, pp_group)
            continue
        if step % 7 == 3:
            # Read late: the senders must not overwrite what was received.
            for _ in range(20):
                lag @ lag
        want = step % 100 + pp_size - 2
        assert received["hidden_states"].shape == (_rows(step), WIDTH)
        assert (received["hidden_states"] == want).all(), step
        assert (received["residual"] == -(step % 100)).all(), step
    if isinstance(transport, SdmaTransport):
        transport.wait_sent()
    torch.accelerator.synchronize()
    # A stage's last metadata send completes only once the next stage
    # receives it, so no stage may exit before the chain is done.
    torch.distributed.barrier(group=pp_group.cpu_group)


def _copy_engine(pp_group, persistent, profile_send) -> PPTransport:
    config = SimpleNamespace(
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp_group.world_size,
            tensor_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )
    transport, reason = CopyEngineTransport.create(
        config, pp_group, persistent, profile_send
    )
    assert transport is not None, reason
    return transport


def _sdma(pp_group, persistent, profile_send) -> PPTransport:
    config = SimpleNamespace(
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp_group.world_size,
            tensor_parallel_size=1,
            prefill_context_parallel_size=1,
            use_ubatching=False,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=MAX_ROWS),
    )
    transport, reason = SdmaTransport.create(config, pp_group, persistent)
    assert transport is not None, reason
    return transport


@ray.remote(num_gpus=1, max_calls=1)
def nccl_chain_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _run_chain(
        lambda pp_group, persistent, profile_send: NcclTransport(
            get_tp_group(), profile_send
        ),
        pp_size,
        rank,
        distributed_init_port,
    )


@ray.remote(num_gpus=1, max_calls=1)
def copy_engine_chain_worker(
    monkeypatch, tp_size, pp_size, rank, distributed_init_port
):
    _run_chain(_copy_engine, pp_size, rank, distributed_init_port)


@ray.remote(num_gpus=1, max_calls=1)
def sdma_chain_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _run_chain(_sdma, pp_size, rank, distributed_init_port)


def _run(monkeypatch: pytest.MonkeyPatch, pp_size: int, test_target) -> None:
    # NVML, not CUDA: the test process must not initialize CUDA.
    if test_target is copy_engine_chain_worker and not (
        current_platform.is_cuda()
        and current_platform.is_fully_connected(list(range(pp_size)))
    ):
        pytest.skip("copy engines need NVLink-connected CUDA GPUs")
    if test_target is sdma_chain_worker and not current_platform.is_rocm():
        pytest.skip("SDMA engines need ROCm")
    multi_process_parallel(monkeypatch, 1, pp_size, test_target)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "test_target", [nccl_chain_worker, copy_engine_chain_worker, sdma_chain_worker]
)
def test_pp_transport_pp2(monkeypatch: pytest.MonkeyPatch, test_target):
    _run(monkeypatch, 2, test_target)


@multi_gpu_test(num_gpus=4)
@pytest.mark.parametrize(
    "test_target", [nccl_chain_worker, copy_engine_chain_worker, sdma_chain_worker]
)
def test_pp_transport_pp4(monkeypatch: pytest.MonkeyPatch, test_target):
    _run(monkeypatch, 4, test_target)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP transport on SDMA copy engines (ROCm, one node).

The receiving stage owns a ring of slots shared with the sender over CUDA IPC. A
sender thread waits for the step's event, copies the tensors into the step's slot
on SDMA engines (no CUs) and bumps a ``ready`` counter in host shared memory; the
receiver bumps ``freed`` once the step that read a slot has finished on the GPU.

The forward reads the received tensors in place, and the sender thread reads the
forward's outputs after it has returned, so this needs eager mode: a CUDA graph
replay would overwrite its outputs. A middle stage may relay views of its
receive slot; the slot is freed only once this rank's sender has copied them.
"""

import ctypes
import mmap
import os
import queue
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.pp_transport import PPTransport
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)

NUM_SLOTS = 3
ALIGN = 256

_HSA_DEVICE_TYPE_GPU = 1
_HSA_AGENT_INFO_DEVICE = 17
_HSA_AMD_AGENT_INFO_BDFID = 0xA006
_HSA_AMD_SIGNAL_AMD_GPU_ONLY = 1
_HSA_SIGNAL_CONDITION_LT = 2
_HSA_WAIT_STATE_ACTIVE = 1
# Two even engines fill one xGMI direction; odd engines hang on MI355X.
_ENGINES = (1 << 2, 1 << 4)


class _Agent(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


class _Signal(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


class _Hsa:
    """The few ROCr calls of an SDMA peer copy, over ctypes."""

    def __init__(self) -> None:
        lib = ctypes.CDLL("libhsa-runtime64.so.1")
        lib.hsa_init.restype = ctypes.c_int
        self._cb_type = ctypes.CFUNCTYPE(ctypes.c_int, _Agent, ctypes.c_void_p)
        lib.hsa_iterate_agents.argtypes = [self._cb_type, ctypes.c_void_p]
        lib.hsa_agent_get_info.argtypes = [_Agent, ctypes.c_int, ctypes.c_void_p]
        lib.hsa_amd_signal_create.argtypes = [
            ctypes.c_int64,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(_Signal),
        ]
        lib.hsa_signal_wait_scacquire.restype = ctypes.c_int64
        lib.hsa_signal_wait_scacquire.argtypes = [
            _Signal,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_uint64,
            ctypes.c_int,
        ]
        lib.hsa_signal_store_relaxed.argtypes = [_Signal, ctypes.c_int64]
        lib.hsa_amd_memory_async_copy_on_engine.argtypes = [
            ctypes.c_void_p,
            _Agent,
            ctypes.c_void_p,
            _Agent,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_void_p,
            _Signal,
            ctypes.c_uint32,
            ctypes.c_bool,
        ]
        if lib.hsa_init() != 0:
            raise RuntimeError("hsa_init failed")
        self.lib = lib

    def gpu_agent(self, device: torch.device) -> _Agent:
        p = torch.cuda.get_device_properties(device)
        want = (p.pci_bus_id << 8) | (p.pci_device_id << 3)
        found: list[_Agent] = []

        def visit(agent, _):
            kind = ctypes.c_uint32()
            self.lib.hsa_agent_get_info(
                agent, _HSA_AGENT_INFO_DEVICE, ctypes.byref(kind)
            )
            if kind.value == _HSA_DEVICE_TYPE_GPU:
                bdf = ctypes.c_uint32()
                self.lib.hsa_agent_get_info(
                    agent, _HSA_AMD_AGENT_INFO_BDFID, ctypes.byref(bdf)
                )
                if bdf.value & ~0x7 == want:
                    found.append(_Agent(agent.handle))
            return 0

        self.lib.hsa_iterate_agents(self._cb_type(visit), None)
        if not found:
            raise RuntimeError(f"no HSA agent for {device}")
        return found[0]

    def signal(self) -> _Signal:
        s = _Signal()
        if self.lib.hsa_amd_signal_create(
            1, 0, None, _HSA_AMD_SIGNAL_AMD_GPU_ONLY, ctypes.byref(s)
        ):
            raise RuntimeError("hsa_amd_signal_create failed")
        return s

    def copy(self, dst, dst_agent, src, src_agent, size, signal, engine) -> None:
        self.lib.hsa_signal_store_relaxed(signal, 1)
        if self.lib.hsa_amd_memory_async_copy_on_engine(
            dst, dst_agent, src, src_agent, size, 0, None, signal, engine, True
        ):
            raise RuntimeError("hsa_amd_memory_async_copy_on_engine failed")

    def wait(self, signal) -> None:
        while (
            self.lib.hsa_signal_wait_scacquire(
                signal,
                _HSA_SIGNAL_CONDITION_LT,
                1,
                1_000_000_000,
                _HSA_WAIT_STATE_ACTIVE,
            )
            > 0
        ):
            pass


class _Counters:
    """Two monotonic uint64 counters in host shared memory: ready (sender) and
    freed (receiver)."""

    def __init__(self, name: str, create: bool) -> None:
        path = f"/dev/shm/{name}"
        if create:
            with open(path, "wb") as f:
                f.write(b"\0" * 16)
        with open(path, "r+b") as f:
            self._mm = mmap.mmap(f.fileno(), 16)
        self.ready = ctypes.c_uint64.from_buffer(self._mm, 0)
        self.freed = ctypes.c_uint64.from_buffer(self._mm, 8)
        self.path = path

    @staticmethod
    def wait(counter: ctypes.c_uint64, at_least: int) -> None:
        spins = 0
        while counter.value < at_least:
            spins += 1
            time.sleep(0 if spins < 1000 else 20e-6)


def _layout(tensors: list[tuple[str, torch.Tensor]]) -> tuple[list[int], int]:
    offsets, end = [], 0
    for _, t in tensors:
        offsets.append(end)
        end += (t.numel() * t.element_size() + ALIGN - 1) // ALIGN * ALIGN
    return offsets, end


class _Sender:
    """The sending side of one stage boundary (this rank -> dst)."""

    def __init__(self, dst: int, max_num_tokens: int) -> None:
        self.dst = dst
        self.max_num_tokens = max_num_tokens
        self.seq = 0
        self.slots: list[torch.Tensor] | None = None
        self._pending: list[Any] = []
        self._jobs: queue.Queue = queue.Queue()
        self._error: BaseException | None = None

    def _setup(self, tensors, num_tokens: int) -> dict:
        per_token = sum(t.numel() * t.element_size() for _, t in tensors) / max(
            1, num_tokens
        )
        slot_bytes = int(per_token * self.max_num_tokens) + ALIGN * (len(tensors) + 1)
        name = f"vllm_pp_sdma_{uuid.uuid4().hex}"
        self.counters = _Counters(name, create=True)
        self.hsa = _Hsa()
        self.device = torch.device("cuda", torch.accelerator.current_device_index())
        return dict(slot_bytes=slot_bytes, num_slots=NUM_SLOTS, counters=name)

    def _open_slots(self, shared) -> None:
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        self.slots = [rebuild_cuda_tensor(*args) for args in shared]
        peer = self.slots[0].device
        self.src_agent = self.hsa.gpu_agent(self.device)
        self.dst_agent = self.hsa.gpu_agent(peer)
        self.signals = [self.hsa.signal() for _ in _ENGINES]
        threading.Thread(target=self._loop, daemon=True, name="pp-sdma-send").start()
        logger.info(
            "PP transport on SDMA: %d slots of %.0f MiB in PP rank %d, engines %s",
            len(self.slots),
            self.slots[0].numel() / 2**20,
            self.dst,
            [e.bit_length() - 1 for e in _ENGINES],
        )

    def send(self, tensor_dict: dict[str, Any], pp_group: "GroupCoordinator") -> None:
        if self._error is not None:
            raise RuntimeError("PP SDMA hand-off failed") from self._error
        tensors = [
            (k, v.contiguous())
            for k, v in tensor_dict.items()
            if isinstance(v, torch.Tensor)
        ]
        others = {
            k: v for k, v in tensor_dict.items() if not isinstance(v, torch.Tensor)
        }
        num_tokens = max((t.shape[0] for _, t in tensors if t.dim()), default=0)
        setup = None
        if self.slots is None:
            setup = self._setup(tensors, num_tokens)
        self.seq += 1
        slot = self.seq % NUM_SLOTS
        offsets, total = _layout(tensors)
        meta = dict(
            seq=self.seq,
            slot=slot,
            tensors=[
                (k, tuple(t.shape), t.dtype, off)
                for (k, t), off in zip(tensors, offsets)
            ],
            others=others,
            setup=setup,
        )
        self._pending = [h for h in self._pending if not h.is_completed()]
        self._pending.append(pp_group.isend_object(meta, dst=self.dst))
        if setup is not None:
            self._open_slots(pp_group.recv_object(src=self.dst))
        assert self.slots is not None
        if total > self.slots[slot].numel():
            raise RuntimeError(
                f"PP SDMA slot of {self.slots[slot].numel()} B < step of {total} B"
            )
        done = torch.cuda.Event()
        done.record()
        self._jobs.put((self.seq, slot, tensors, offsets, done))

    def wait_sent(self) -> None:
        """Block until everything given so far has been copied out."""
        if self.slots is not None:
            _Counters.wait(self.counters.ready, self.seq)

    def _loop(self) -> None:
        try:
            assert self.slots is not None
            slots = self.slots
            torch.accelerator.set_device_index(self.device)
            while True:
                seq, slot, tensors, offsets, done = self._jobs.get()
                done.synchronize()
                _Counters.wait(self.counters.freed, seq - NUM_SLOTS)
                base = slots[slot].data_ptr()
                for (_, t), off in zip(tensors, offsets):
                    size = t.numel() * t.element_size()
                    if size == 0:
                        continue
                    chunk = (size // len(_ENGINES) + ALIGN - 1) // ALIGN * ALIGN
                    used = []
                    for k, engine in enumerate(_ENGINES):
                        lo = k * chunk
                        if lo >= size:
                            break
                        n = min(chunk, size - lo)
                        self.hsa.copy(
                            base + off + lo,
                            self.dst_agent,
                            t.data_ptr() + lo,
                            self.src_agent,
                            n,
                            self.signals[k],
                            engine,
                        )
                        used.append(self.signals[k])
                    for s in used:
                        self.hsa.wait(s)
                del tensors
                self.counters.ready.value = seq
        except BaseException as e:  # surfaced on the next send()
            self._error = e
            logger.exception("PP SDMA sender thread failed")


class _Receiver:
    """The receiving side of one stage boundary (src -> this rank)."""

    def __init__(self, src: int) -> None:
        self.src = src
        self.slots: list[torch.Tensor] | None = None
        self._last_seq = 0
        self._frees: queue.Queue = queue.Queue()

    def release_previous(self, after: _Sender | None) -> None:
        """Called at the top of a step: the previous step's slot is free once
        everything enqueued so far has run and ``after`` (this rank's sender,
        which may relay views of the slot) has copied out everything it was
        given so far."""
        if self._last_seq:
            done = torch.cuda.Event()
            done.record()
            sent = after.seq if after is not None and after.slots is not None else 0
            self._frees.put((self._last_seq, done, after, sent))
            self._last_seq = 0

    def _free_loop(self) -> None:
        torch.accelerator.set_device_index(self.device)
        while True:
            seq, done, after, sent = self._frees.get()
            done.synchronize()
            if sent:
                _Counters.wait(after.counters.ready, sent)
            self.counters.freed.value = seq

    def recv(self, pp_group: "GroupCoordinator") -> dict[str, Any]:
        from torch.multiprocessing.reductions import reduce_tensor

        meta = pp_group.recv_object(src=self.src)
        if meta["setup"] is not None:
            setup = meta["setup"]
            self.device = torch.device("cuda", torch.accelerator.current_device_index())
            self.slots = [
                torch.empty(setup["slot_bytes"], dtype=torch.uint8, device=self.device)
                for _ in range(setup["num_slots"])
            ]
            pp_group.send_object(
                [reduce_tensor(s)[1] for s in self.slots], dst=self.src
            )
            self.counters = _Counters(setup["counters"], create=False)
            os.unlink(self.counters.path)  # both sides hold the mapping
            threading.Thread(
                target=self._free_loop, daemon=True, name="pp-sdma-free"
            ).start()
        assert self.slots is not None
        _Counters.wait(self.counters.ready, meta["seq"])
        slot = self.slots[meta["slot"]]
        out: dict[str, Any] = dict(meta["others"])
        for key, shape, dtype, off in meta["tensors"]:
            n = 1
            for d in shape:
                n *= d
            nbytes = n * dtype.itemsize
            out[key] = slot[off : off + nbytes].view(dtype).view(shape)
        self._last_seq = meta["seq"]
        return out


class SdmaTransport(PPTransport):
    """Sends to the next stage and receives from the previous one, whichever
    this rank has. The receive slots are allocated on the first step."""

    recv_in_place = True

    def __init__(
        self,
        pp_group: "GroupCoordinator",
        max_num_tokens: int,
        persistent: dict[str, torch.Tensor] | None,
    ) -> None:
        rank = pp_group.rank_in_group
        self.tx = None if pp_group.is_last_rank else _Sender(rank + 1, max_num_tokens)
        self.rx = None if pp_group.is_first_rank else _Receiver(rank - 1)
        if self.rx is not None and persistent is not None:
            # As the sender sizes them, from the same tensors.
            row_bytes = sum(
                t[0].numel() * t.element_size() for t in persistent.values()
            )
            slot_bytes = row_bytes * max_num_tokens + ALIGN * (len(persistent) + 1)
            self.unprofiled_bytes = NUM_SLOTS * slot_bytes

    def before_forward(self) -> None:
        if self.rx is not None:
            self.rx.release_previous(after=self.tx)

    def send(self, tensors: dict[str, Any], pp_group: "GroupCoordinator") -> None:
        assert self.tx is not None
        self.tx.send(tensors, pp_group)

    def recv(self, pp_group: "GroupCoordinator") -> IntermediateTensors:
        assert self.rx is not None
        return IntermediateTensors(self.rx.recv(pp_group))

    def wait_sent(self) -> None:
        if self.tx is not None:
            self.tx.wait_sent()

    @classmethod
    def create(
        cls,
        vllm_config: "VllmConfig",
        pp_group: "GroupCoordinator",
        persistent: dict[str, torch.Tensor] | None,
    ) -> tuple["SdmaTransport | None", str | None]:
        """Every rank returns a transport, or None and the same reason."""
        from vllm.config import CUDAGraphMode
        from vllm.distributed.parallel_state import in_the_same_node_as
        from vllm.platforms import current_platform

        parallel_config = vllm_config.parallel_config
        if not current_platform.is_rocm():
            return None, "not ROCm"
        if not vllm_config.use_v2_model_runner:
            return None, "needs the V2 model runner"
        if parallel_config.pipeline_parallel_size <= 1:
            return None, "no pipeline parallelism"
        if (
            parallel_config.tensor_parallel_size != 1
            or parallel_config.prefill_context_parallel_size != 1
        ):
            return None, "needs TP=1 and PCP=1"
        if vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            return None, "needs eager mode (cudagraph_mode NONE)"
        # Microbatched steps copy into the persistent buffers, so a middle stage
        # would relay views of those, which the next receive overwrites.
        if parallel_config.pipeline_parallel_size > 2 and parallel_config.use_ubatching:
            return None, "PP > 2 does not support microbatching"
        if not all(in_the_same_node_as(pp_group.cpu_group, source_rank=0)):
            return None, "PP stages span several nodes"
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        return cls(pp_group, max_num_tokens, persistent), None

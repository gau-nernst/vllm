# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, cute
from cutlass.cute.nvgpu import cpasync, warp
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import mma_bf16, simple_tma_copy


class ChunkKKTKernel:
    def __init__(self, head_dim: int, num_heads: int, num_stages: int = 2):
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_stages = num_stages

        # hard-coded
        self.BT = 64
        self.lower_bound = -5.0

    @cute.jit
    def _make_tma_args(
        self,
        tensor: cute.Tensor,
        BT: cutlass.Constexpr[int],
        dim: cutlass.Constexpr[int],
        num_stages: int,
        op: cpasync.TmaCopyOp,
    ):
        # logical layout: [BT, dim]
        # permute for TMA: [dim/64, BT, 64] with swizzling
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        tma_slayout = cute.make_layout(
            (BT, 1, (64, dim // 64)),
            stride=(64, 0, (1, BT * 64)),
        )
        tma_slayout = cute.make_composed_layout(swizzle_128B, 0, tma_slayout)
        smem_slayout = cute.make_layout(
            (BT, 1, (64, dim // 64), num_stages),
            stride=(64, 0, (1, BT * 64), BT * dim),
        )
        smem_slayout = cute.make_composed_layout(swizzle_128B, 0, smem_slayout)

        atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            cute.logical_divide(tensor, (None, None, 64)),
            tma_slayout,
            cta_tiler=(BT, 1, dim),
        )
        return atom, tma_tensor, smem_slayout

    @cute.jit
    def __call__(
        self,
        Q: cute.Tensor,
        K: cute.Tensor,
        g: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        total_chunks: cute.Tensor,
        num_sms: Int32,
        stream: CUstream,
    ):
        BT = self.BT
        head_dim = self.head_dim
        num_stages = self.num_stages

        tma_g2s = cpasync.CopyBulkTensorTileG2SOp()
        Q_args = self._make_tma_args(Q, BT, head_dim, num_stages, tma_g2s)
        K_args = self._make_tma_args(K, BT, head_dim, num_stages, tma_g2s)

        grid = (num_sms // self.num_heads, self.num_heads, 1)
        block = (5 * 32, 1, 1)
        self.kernel(
            Q_args, K_args, g, A_log, dt_bias, cu_seqlens, chunk_indices, total_chunks
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        Q_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        K_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        g: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        total_chunks: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, head_id, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()

        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32

        BT = self.BT
        head_dim = self.head_dim
        num_stages = self.num_stages

        Q_tma_atom, tmaQ, sQ_layout = Q_args
        K_tma_atom, tmaK, sK_layout = K_args

        def allocate_tensor(smem, dtype, layout):
            return smem.allocate_tensor(
                dtype, layout.outer, byte_alignment=128, swizzle=layout.inner
            )

        smem = cutlass.utils.SmemAllocator()
        sQ = allocate_tensor(smem, BFloat16, sQ_layout)[None, 0, None, None]
        sK = allocate_tensor(smem, BFloat16, sK_layout)[None, 0, None, None]

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)

        # prepare ldmatrix/stmatrix ops
        ldsm_op = warp.LdMatrix8x8x16bOp(num_matrices=4)
        stsm_op = warp.StMatrix8x8x16bOp(num_matrices=4)
        ldsm_trans_op = warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True)
        ldsm_atom = cute.make_copy_atom(ldsm_op, BFloat16)
        stsm_atom = cute.make_copy_atom(stsm_op, BFloat16)  # noqa: F841
        ldsm_trans_atom = cute.make_copy_atom(  # noqa: F841
            ldsm_trans_op, BFloat16
        )

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tma_empty_mbar + i, 128)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(Q_tma_atom)
            cpasync.prefetch_descriptor(K_tma_atom)
        cute.arch.sync_threads()

        num_global_chunks = total_chunks[0]
        if warp_id == 4:
            # TMA warp
            stage_id = 0
            parity = 1

            for global_chunk_id in range(bid, num_global_chunks, grid_x):
                seq_id = chunk_indices[global_chunk_id, 0]
                chunk_id = chunk_indices[global_chunk_id, 1]
                bos = cu_seqlens[seq_id]

                mbar = tma_full_mbar + stage_id
                gQ = cute.local_tile(
                    cute.domain_offset((bos, 0), tmaQ[None, head_id, None]),
                    tiler=(BT, head_dim),
                    coord=(chunk_id, 0),
                )
                gK = cute.local_tile(
                    cute.domain_offset((bos, 0), tmaK[None, head_id, None]),
                    tiler=(BT, head_dim),
                    coord=(chunk_id, 0),
                )

                cute.arch.mbarrier_wait(tma_empty_mbar + stage_id, parity)
                with cute.arch.elect_one():
                    STAGE_SIZE = BT * head_dim * 2 * 2
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, STAGE_SIZE)
                simple_tma_copy(Q_tma_atom, gQ, sQ[None, None, stage_id], mbar)
                simple_tma_copy(K_tma_atom, gK, sK[None, None, stage_id], mbar)

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

        else:
            # remaining warps
            stage_id = 0
            parity = 0

            # pre-compute ldmatrix addresses (A operand)
            # shape before: (BT, (64, head_dim/64), num_stages)
            # shape after: ((16,4), ((8,2), (4, head_dim/64)), num_stages)
            sQ_ldsm = cute.logical_divide(sQ, (16, cute.make_layout((8, 2)), None))
            sK_ldsm = cute.logical_divide(sK, (16, cute.make_layout((8, 2)), None))

            # shape: (8, (4, head_dim/64), num_stages)
            sQ_ldsm = sQ_ldsm[
                (lane_id % 16, warp_id), ((None, lane_id // 16), None), None
            ]
            sK_ldsm = sK_ldsm[
                (lane_id % 16, warp_id), ((None, lane_id // 16), None), None
            ]

            # B operand
            # shape before: (BT, (64, head_dim/64), num_stages)
            # shape after: (((8,2),4), ((8,2), (4, head_dim/64)), num_stages)
            sK_right_ldsm = cute.logical_divide(
                sK, (cute.make_layout((8, 2)), cute.make_layout((8, 2)), None)
            )

            # shape: (8, (4, head_dim/64), num_stages)
            sK_right_ldsm = sK_right_ldsm[
                ((lane_id % 8, lane_id // 16), warp_id),
                ((None, (lane_id // 8) % 2), None),
                None,
            ]

            for global_chunk_id in range(bid, num_global_chunks, grid_x):
                # TODO: compute gate and beta

                cute.arch.mbarrier_wait(tma_full_mbar + stage_id, parity)

                qk = cute.make_rmem_tensor((4, 2), Float32)
                kkt = cute.make_rmem_tensor((4, 2), Float32)

                qk.fill(0.0)
                kkt.fill(0.0)

                # MMA_K=16
                for i in cutlass.range_constexpr(head_dim // 16):
                    q = cute.make_rmem_tensor(8, BFloat16)
                    k = cute.make_rmem_tensor(8, BFloat16)
                    k_right = cute.make_rmem_tensor((4, 2), BFloat16)

                    cute.copy(ldsm_atom, sQ_ldsm[None, i, stage_id], q)
                    cute.copy(ldsm_atom, sK_ldsm[None, i, stage_id], k)
                    cute.copy(
                        ldsm_atom,
                        sK_right_ldsm[None, i, stage_id],
                        cute.group_modes(k_right, 0),
                    )

                    qk[None, 0] = mma_bf16(q, k_right[None, 0], qk[None, 0])
                    qk[None, 1] = mma_bf16(q, k_right[None, 1], qk[None, 1])
                    kkt[None, 0] = mma_bf16(k, k_right[None, 0], kkt[None, 0])
                    kkt[None, 1] = mma_bf16(k, k_right[None, 1], kkt[None, 1])

                cute.arch.mbarrier_arrive(tma_empty_mbar + stage_id)

                # TODO: causal mask qk, then store to smem (for TMA store)

                # TODO: strict lower mask kkt

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

    @cache
    @staticmethod
    def compile(head_dim: int, num_heads: int, num_stages: int = 2):
        total_t = cute.sym_int()
        total_chunks_n = cute.sym_int()
        num_sequences = cute.sym_int()

        Q = make_fake_tensor(BFloat16, (total_t, num_heads, head_dim), divisibility=16)
        K = make_fake_tensor(BFloat16, (total_t, num_heads, head_dim), divisibility=16)
        g = make_fake_tensor(Float32, (total_t, num_heads), divisibility=1)
        A_log = make_fake_tensor(Float32, (num_heads,), divisibility=4)
        dt_bias = make_fake_tensor(Float32, (num_heads,), divisibility=4)
        cu_seqlens = make_fake_tensor(Int32, (num_sequences,), divisibility=1)
        chunk_indices = make_fake_tensor(Int32, (total_chunks_n, 2), divisibility=2)
        total_chunks = make_fake_tensor(Int32, (1,), divisibility=1)

        kernel = ChunkKKTKernel(head_dim, num_heads, num_stages)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            Q,
            K,
            g,
            A_log,
            dt_bias,
            cu_seqlens,
            chunk_indices,
            total_chunks,
            Int32(148),
            stream,
            options="--enable-tvm-ffi",
        )


def make_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    if cu_seqlens.device.type != "cuda":
        raise ValueError("cu_seqlens must be a CUDA tensor")

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()
    indices: list[tuple[int, int]] = []
    for seq_id, seqlen in enumerate(lengths):
        for chunk_id in range((int(seqlen) + chunk_size - 1) // chunk_size):
            indices.append((seq_id, chunk_id))

    if not indices:
        raise ValueError("at least one non-empty chunk is required")
    return torch.tensor(indices, dtype=torch.int32, device=cu_seqlens.device)


def kkt_qk_cutedsl(
    Q: torch.Tensor,
    K: torch.Tensor,
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    total_chunks: torch.Tensor | None = None,
    num_sms: int | None = None,
    num_stages: int = 2,
) -> None:
    if Q.ndim != 3 or K.ndim != 3:
        raise ValueError("Q and K must have shape [total_tokens, num_heads, head_dim]")
    if Q.shape != K.shape:
        raise ValueError(f"Q and K shapes must match, got {Q.shape=} {K.shape=}")

    _, num_heads, head_dim = Q.shape
    if head_dim != 128:
        raise ValueError(f"kernel currently expects head_dim=128, got {head_dim}")

    if total_chunks is None:
        total_chunks = torch.tensor(
            [chunk_indices.shape[0]], dtype=torch.int32, device=Q.device
        )
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(Q.device).multi_processor_count

    ChunkKKTKernel.compile(head_dim, num_heads, num_stages)(
        Q,
        K,
        g,
        A_log,
        dt_bias,
        cu_seqlens,
        chunk_indices,
        total_chunks,
        num_sms,
    )


def _parse_seqlens(raw: str, batch: int, seqlen: int) -> list[int]:
    if raw:
        values = [int(part) for part in raw.split(",") if part.strip()]
        if not values:
            raise ValueError("--seqlens must contain at least one integer")
        return values
    return [seqlen] * batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile and launch kernel_kkt_qk.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=64)
    parser.add_argument("--seqlens", type=str, default="")
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this CuteDSL kernel")
    if args.head_dim != 128:
        raise ValueError("kernel_kkt_qk currently hard-codes head_dim=128")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    seqlens = _parse_seqlens(args.seqlens, args.batch, args.seqlen)
    offsets = [0]
    for length in seqlens:
        offsets.append(offsets[-1] + length)

    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)
    total_tokens = offsets[-1]
    Q = torch.randn(
        total_tokens, args.heads, args.head_dim, device=device, dtype=torch.bfloat16
    )
    K = torch.randn_like(Q)
    g = torch.randn(total_tokens, args.heads, device=device, dtype=torch.float32)
    A_log = torch.randn(args.heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(args.heads, device=device, dtype=torch.float32)
    chunk_indices = make_chunk_indices(cu_seqlens, chunk_size=64)
    total_chunks = torch.tensor(
        [chunk_indices.shape[0]], dtype=torch.int32, device=device
    )

    kkt_qk_cutedsl(
        Q,
        K,
        g,
        A_log,
        dt_bias,
        cu_seqlens,
        chunk_indices,
        total_chunks,
        num_stages=args.stages,
    )
    torch.accelerator.synchronize()

    print(
        "launched kernel_kkt_qk",
        f"total_tokens={total_tokens}",
        f"heads={args.heads}",
        f"head_dim={args.head_dim}",
        f"chunks={chunk_indices.shape[0]}",
    )


if __name__ == "__main__":
    main()

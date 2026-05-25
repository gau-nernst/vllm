# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import cvt, mma_bf16, simple_tma_copy


@dsl_user_op
def f32_rcp_rn(x: Float32, *, loc=None, ip=None) -> Float32:
    out = llvm.inline_asm(
        T.f32(),
        [x.ir_value(loc=loc, ip=ip)],
        "rcp.rn.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(out)


@cute.jit
def sigmoid(x: Float32) -> Float32:
    return f32_rcp_rn(Float32(1.0) + cute.math.exp(-x, fastmath=True))


class KDAPreRecurrentKernel:
    def __init__(self, head_dim: int, num_heads: int, num_stages: int = 2) -> None:
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_stages = num_stages

        # hard-coded
        self.BT = 16
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
        V: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        total_chunks: cute.Tensor,
        scale: Float32,
        num_sms: Int32,
        stream: CUstream,
    ):
        BT = self.BT
        head_dim = self.head_dim
        num_stages = self.num_stages

        tma_g2s = cpasync.CopyBulkTensorTileG2SOp()
        Q_args = self._make_tma_args(Q, BT * 4, head_dim, num_stages, tma_g2s)
        K_args = self._make_tma_args(K, BT * 4, head_dim, num_stages, tma_g2s)
        V_args = self._make_tma_args(V, BT * 4, head_dim, num_stages, tma_g2s)
        a_args = self._make_tma_args(a, BT * 4, head_dim, num_stages, tma_g2s)

        grid = (num_sms // self.num_heads, self.num_heads, 1)
        block = (5 * 32, 1, 1)
        self.kernel(
            Q_args,
            K_args,
            V_args,
            a_args,
            b,
            A_log,
            dt_bias,
            cu_seqlens,
            chunk_indices,
            total_chunks,
            scale,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        Q_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        K_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        V_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        a_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        total_chunks: cute.Tensor,
        scale: Float32,
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
        V_tma_atom, tmaV, sV_layout = V_args
        a_tma_atom, tmaa, sa_layout = a_args

        def allocate_tensor(smem, dtype, layout):
            return smem.allocate_tensor(
                dtype, layout.outer, byte_alignment=128, swizzle=layout.inner
            )

        smem = cutlass.utils.SmemAllocator()
        sQ = allocate_tensor(smem, BFloat16, sQ_layout)[None, 0, None, None]
        sK = allocate_tensor(smem, BFloat16, sK_layout)[None, 0, None, None]
        sV = allocate_tensor(smem, BFloat16, sV_layout)[None, 0, None, None]
        sa = allocate_tensor(smem, BFloat16, sa_layout)[None, 0, None, None]

        # to store k * exp(-g_cu)
        sK_right = allocate_tensor(
            smem, BFloat16, cute.slice_(sK_layout, (None, None, None, 0))
        )[None, 0, None]

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

        cp_op = cute.nvgpu.CopyUniversalOp()
        cp_8B_atom = cute.make_copy_atom(cp_op, Int32, num_bits_per_copy=64)
        cp_16B_atom = cute.make_copy_atom(cp_op, Int32, num_bits_per_copy=128)

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tma_empty_mbar + i, 128)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(Q_tma_atom)
            cpasync.prefetch_descriptor(K_tma_atom)
            cpasync.prefetch_descriptor(V_tma_atom)
            cpasync.prefetch_descriptor(a_tma_atom)
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

                def this_tile(tma, bos, chunk_id):
                    return cute.local_tile(
                        cute.domain_offset((bos, 0), tma[None, head_id, None]),
                        tiler=(BT * 4, head_dim),
                        coord=(chunk_id, 0),
                    )

                gQ = this_tile(tmaQ, bos, chunk_id)
                gK = this_tile(tmaK, bos, chunk_id)
                gV = this_tile(tmaV, bos, chunk_id)
                ga = this_tile(tmaa, bos, chunk_id)
                mbar = tma_full_mbar + stage_id

                cute.arch.mbarrier_wait(tma_empty_mbar + stage_id, parity)
                with cute.arch.elect_one():
                    STAGE_SIZE = (BT * 4) * head_dim * 4 * 2
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, STAGE_SIZE)
                simple_tma_copy(Q_tma_atom, gQ, sQ[None, None, stage_id], mbar)
                simple_tma_copy(K_tma_atom, gK, sK[None, None, stage_id], mbar)
                simple_tma_copy(V_tma_atom, gV, sV[None, None, stage_id], mbar)
                simple_tma_copy(a_tma_atom, ga, sa[None, None, stage_id], mbar)

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
            # shape before: (BT, (64, head_dim/64))
            # shape after: (((8,2),4), ((8,2), (4, head_dim/64)))
            sK_right_ldsm = cute.logical_divide(
                sK_right, (cute.make_layout((8, 2)), cute.make_layout((8, 2)))
            )

            # shape: (8, (4, head_dim/64))
            sK_right_ldsm = sK_right_ldsm[
                ((lane_id % 8, lane_id // 16), warp_id),
                ((None, (lane_id // 8) % 2), None),
            ]

            # each warp handles BT
            sQ_thr = cute.local_tile(sQ, (BT, 4, num_stages), (warp_id, lane_id, 0))
            sK_thr = cute.local_tile(sK, (BT, 4, num_stages), (warp_id, lane_id, 0))
            sa_thr = cute.local_tile(sa, (BT, 4, num_stages), (warp_id, lane_id, 0))
            sK_right_thr = cute.local_tile(sK_right, (BT, 4), (warp_id, lane_id))

            # preload stuff for gate
            A_ = cute.math.exp(A_log[head_id].to(Float32), fastmath=True)
            dt_bias_thr = cute.make_rmem_tensor(4, Float32)
            cute.copy(
                cp_16B_atom,
                cute.local_tile(dt_bias[head_id, None], (4,), (lane_id,)),
                dt_bias_thr,
            )

            # row/col indices of MMA acc fragment
            row_indices = cute.make_rmem_tensor(2, Int32)
            row_indices[0] = lane_id // 4
            row_indices[1] = lane_id // 4 + 8
            row_indices = row_indices.load().reshape((1, 2, 1))

            col_indices = cute.make_rmem_tensor(4, Int32)
            col_indices[0] = (lane_id % 4) * 2
            col_indices[1] = (lane_id % 4) * 2 + 1
            col_indices[2] = (lane_id % 4) * 2 + 8
            col_indices[3] = (lane_id % 4) * 2 + 9
            col_indices = col_indices.load().reshape((2, 1, 2))

            for global_chunk_id in range(bid, num_global_chunks, grid_x):
                seq_id = chunk_indices[global_chunk_id, 0]
                chunk_id = chunk_indices[global_chunk_id, 1]
                bos = cu_seqlens[seq_id]
                chunk_size = bos - chunk_id * (BT * 4)

                g_cu = cute.make_rmem_tensor(4, Float32)
                g_cu.fill(0.0)

                # TODO: separate QKVa into a->QK->V
                cute.arch.mbarrier_wait(tma_full_mbar + stage_id, parity)

                ##### stage 1: compute gate and scale Q/K #####
                for i in cutlass.range_constexpr(BT):
                    if warp_id * BT + i < chunk_size:
                        Q_thr = cute.make_rmem_tensor(4, BFloat16)
                        K_thr = cute.make_rmem_tensor(4, BFloat16)
                        a_thr = cute.make_rmem_tensor(4, BFloat16)

                        cute.copy(cp_8B_atom, sQ_thr[i, None, stage_id], Q_thr)
                        cute.copy(cp_8B_atom, sK_thr[i, None, stage_id], K_thr)
                        cute.copy(cp_8B_atom, sa_thr[i, None, stage_id], a_thr)

                        a_f32 = cvt.bf16x2_to_fp32x2(cute.recast_tensor(a_thr, Uint32))
                        q_f32 = cvt.bf16x2_to_fp32x2(cute.recast_tensor(Q_thr, Uint32))
                        k_f32 = cvt.bf16x2_to_fp32x2(cute.recast_tensor(K_thr, Uint32))
                        k_f32_right = cute.make_rmem_tensor_like(k_f32)

                        for j in cutlass.range_constexpr(4):
                            g = self.lower_bound * sigmoid(
                                A_ * (a_f32[j] + dt_bias_thr[j])
                            )
                            g_cu[j] += g

                            decay = cute.math.exp(g_cu[j])
                            k_f32_right[j] = k_f32[j] * f32_rcp_rn(decay)
                            q_f32[j] *= decay * scale
                            k_f32[j] *= decay

                        Q_thr.store(q_f32.load().to(BFloat16))
                        K_thr.store(k_f32.load().to(BFloat16))
                        K_right_thr = cute.make_rmem_tensor_like(K_thr)
                        K_right_thr.store(k_f32_right.load().to(BFloat16))

                        # store back to existing TMA buffers
                        cute.copy(cp_8B_atom, Q_thr, sQ_thr[i, None, stage_id])
                        cute.copy(cp_8B_atom, K_thr, sK_thr[i, None, stage_id])
                        cute.copy(cp_8B_atom, K_right_thr, sK_right_thr[i, None])

                # TODO: store g_cu as g_last. then g_cu registers are freed

                ##### stage 2: Q @ K.T and K @ K.T MMA #####
                qk = cute.make_rmem_tensor((4, 2), Float32)
                kkt = cute.make_rmem_tensor((4, 2), Float32)

                qk.fill(0.0)
                kkt.fill(0.0)

                # MMA_K=16
                # TODO: we may want to issue a lot of ldmatrix at once,
                # then let the compiler decide registers reuse.
                for i in cutlass.range_constexpr(head_dim // 16):
                    q = cute.make_rmem_tensor(8, BFloat16)
                    k = cute.make_rmem_tensor(8, BFloat16)
                    k_right = cute.make_rmem_tensor((4, 2), BFloat16)

                    cute.copy(ldsm_atom, sQ_ldsm[None, i, stage_id], q)
                    cute.copy(ldsm_atom, sK_ldsm[None, i, stage_id], k)
                    cute.copy(
                        ldsm_atom,
                        sK_right_ldsm[None, i],
                        cute.group_modes(k_right, 0),
                    )

                    qk[None, 0] = mma_bf16(q, k_right[None, 0], qk[None, 0])
                    qk[None, 1] = mma_bf16(q, k_right[None, 1], qk[None, 1])
                    kkt[None, 0] = mma_bf16(k, k_right[None, 0], kkt[None, 0])
                    kkt[None, 1] = mma_bf16(k, k_right[None, 1], kkt[None, 1])

                cute.arch.mbarrier_arrive(tma_empty_mbar + stage_id)

                # TODO: causal mask qk, then store to smem (for TMA store)
                # p = cute.where(
                #     row_indices >= col_indices, qk.load().reshape((2, 2, 2)), 0.0
                # )

                # TODO: strict lower mask kkt
                # TODO: multiply by beta
                # A = cute.where(
                #     row_indices > col_indices, kkt.load().reshape((2, 2, 2)), 0.0
                # )

                # TODO: Newton-Schulz inverse

                # TODO: UW projection

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
        V = make_fake_tensor(BFloat16, (total_t, num_heads, head_dim), divisibility=16)
        a = make_fake_tensor(BFloat16, (total_t, num_heads, head_dim), divisibility=16)
        b = make_fake_tensor(BFloat16, (total_t, num_heads), divisibility=1)
        A_log = make_fake_tensor(Float32, (num_heads,), divisibility=4)
        dt_bias = make_fake_tensor(Float32, (num_heads, head_dim), divisibility=4)
        cu_seqlens = make_fake_tensor(Int32, (num_sequences,), divisibility=1)
        chunk_indices = make_fake_tensor(Int32, (total_chunks_n, 2), divisibility=2)
        total_chunks = make_fake_tensor(Int32, (1,), divisibility=1)

        kernel = KDAPreRecurrentKernel(head_dim, num_heads, num_stages)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            Q,
            K,
            V,
            a,
            b,
            A_log,
            dt_bias,
            cu_seqlens,
            chunk_indices,
            total_chunks,
            Float32(1.0),
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
    V: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    total_chunks: torch.Tensor | None = None,
    scale: float | None = None,
    num_sms: int | None = None,
    num_stages: int = 2,
) -> None:
    if Q.ndim != 3 or K.ndim != 3:
        raise ValueError("Q and K must have shape [total_tokens, num_heads, head_dim]")
    if Q.shape != K.shape:
        raise ValueError(f"Q and K shapes must match, got {Q.shape=} {K.shape=}")

    _, num_heads, head_dim = Q.shape

    if total_chunks is None:
        total_chunks = torch.tensor(
            [chunk_indices.shape[0]], dtype=torch.int32, device=Q.device
        )
    if scale is None:
        scale = K.shape[-1] ** -0.5
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(Q.device).multi_processor_count

    KDAPreRecurrentKernel.compile(head_dim, num_heads, num_stages)(
        Q,
        K,
        V,
        a,
        b,
        A_log,
        dt_bias,
        cu_seqlens,
        chunk_indices,
        total_chunks,
        scale,
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
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_default_device("cuda")
    torch.manual_seed(args.seed)
    seqlens = _parse_seqlens(args.seqlens, args.batch, args.seqlen)
    offsets = [0]
    for length in seqlens:
        offsets.append(offsets[-1] + length)

    cu_seqlens = torch.tensor(offsets, dtype=torch.int32)
    total_tokens = offsets[-1]
    HEAD_DIM = 128
    Q = torch.randn(total_tokens, args.heads, HEAD_DIM, dtype=torch.bfloat16)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    a = torch.randn_like(Q)
    b = torch.randn(total_tokens, args.heads, dtype=torch.bfloat16)
    A_log = torch.randn(args.heads, dtype=torch.float32)
    dt_bias = torch.randn(args.heads, HEAD_DIM, dtype=torch.float32)
    chunk_indices = make_chunk_indices(cu_seqlens, chunk_size=16 * 4)
    total_chunks = torch.tensor([chunk_indices.shape[0]], dtype=torch.int32)

    kkt_qk_cutedsl(
        Q,
        K,
        V,
        a,
        b,
        A_log,
        dt_bias,
        cu_seqlens,
        chunk_indices,
        total_chunks,
    )
    torch.accelerator.synchronize()

    print(
        "launched kernel_kkt_qk",
        f"total_tokens={total_tokens}",
        f"heads={args.heads}",
        f"chunks={chunk_indices.shape[0]}",
    )


if __name__ == "__main__":
    main()

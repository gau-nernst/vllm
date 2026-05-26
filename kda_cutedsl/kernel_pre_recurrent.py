# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import math
import statistics
from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cutlass_dsl import T, dsl_user_op
from flashinfer.testing import bench_gpu_time_with_cupti
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import (
    _tcgen05,
    cvt,
    fence_before_tma_store,
    mma_bf16,
    simple_tma_copy,
)


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


class KDAChunkPreRecurrentKernel:
    # hard-coded
    BT = 16
    lower_bound = -5.0
    num_warps = 10

    def __init__(self, head_dim: int, num_heads: int, num_stages: int = 2) -> None:
        assert head_dim == 128
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_stages = num_stages

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
        Q_decay: cute.Tensor,
        K_decay: cute.Tensor,
        U: cute.Tensor,
        W: cute.Tensor,
        P: cute.Tensor,
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
        tma_s2g = cpasync.CopyBulkTensorTileS2GOp()
        Q_args = self._make_tma_args(Q, BT * 4, head_dim, num_stages, tma_g2s)
        K_args = self._make_tma_args(K, BT * 4, head_dim, num_stages, tma_g2s)
        V_args = self._make_tma_args(V, BT * 4, head_dim, num_stages, tma_g2s)
        a_args = self._make_tma_args(a, BT * 4, head_dim, num_stages, tma_g2s)
        Q_decay_args = self._make_tma_args(Q_decay, BT * 4, head_dim, 1, tma_s2g)
        K_decay_args = self._make_tma_args(K_decay, BT * 4, head_dim, 1, tma_s2g)
        U_args = self._make_tma_args(U, BT * 4, head_dim, 1, tma_s2g)
        W_args = self._make_tma_args(W, BT * 4, head_dim, 1, tma_s2g)

        grid = (num_sms // self.num_heads, self.num_heads, 1)
        block = (self.num_warps * 32, 1, 1)
        self.kernel(
            Q_args,
            K_args,
            V_args,
            a_args,
            b,
            A_log,
            dt_bias,
            Q_decay_args,
            K_decay_args,
            U_args,
            W_args,
            P,
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
        Q_decay_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        K_decay_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        U_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        W_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        P: cute.Tensor,
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
        lower_bound = self.lower_bound
        head_dim = self.head_dim
        num_stages = self.num_stages

        Q_tma_atom, tmaQ, sQ_layout = Q_args
        K_tma_atom, tmaK, sK_layout = K_args
        V_tma_atom, tmaV, sV_layout = V_args
        a_tma_atom, tmaa, sa_layout = a_args
        Q_decay_tma_atom, tmaQ_decay, _ = Q_decay_args
        K_decay_tma_atom, tmaK_decay, _ = K_decay_args
        U_tma_atom, tmaU, sU_layout = U_args
        W_tma_atom, tmaW, sW_layout = W_args

        def allocate_tensor(smem, dtype, layout):
            return smem.allocate_tensor(
                dtype, layout.outer, byte_alignment=128, swizzle=layout.inner
            )

        # allocate smem
        smem = cutlass.utils.SmemAllocator()
        sQ = allocate_tensor(smem, BFloat16, sQ_layout)[None, 0, None, None]
        sK = allocate_tensor(smem, BFloat16, sK_layout)[None, 0, None, None]
        sV = allocate_tensor(smem, BFloat16, sV_layout)[None, 0, None, None]
        sa = allocate_tensor(smem, BFloat16, sa_layout)[None, 0, None, None]
        sU = allocate_tensor(smem, BFloat16, sU_layout)[None, 0, None, 0]
        sW = allocate_tensor(smem, BFloat16, sW_layout)[None, 0, None, 0]

        # to store k * exp(-g_cu)
        sK_right = allocate_tensor(
            smem, BFloat16, cute.slice_(sK_layout, (None, None, None, 0))
        )[None, 0, None]

        # workspace for transpose data during Newton-Schulz inverse
        # layout is chosen to avoid bank conflicts without using swizzling
        # logically, this is (16,16) tile for each warp
        inv_workspace = smem.allocate_tensor(
            BFloat16,
            cute.make_layout((16, (8, 2), 4), stride=(8, (1, 16 * 8), 16 * 16)),
            byte_alignment=128,
        )

        sbeta = smem.allocate_array(Float32, BT * 4)

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)
        inv_mbar = smem.allocate_array(Int64, num_stages)
        tmem_full_mbar = smem.allocate_array(Int64, 2)
        tmem_empty_mbar = smem.allocate_array(Int64, 2)

        taddr = smem.allocate_array(Int32, 1)

        # allocate tmem
        # U and W occupy BT*4 columns each
        tmem = 0
        assert tmem + (BT * 4) * 2 * 2 <= 512

        # prepare ldmatrix/stmatrix ops
        ldsm_op = warp.LdMatrix8x8x16bOp(num_matrices=4)
        stsm_op = warp.StMatrix8x8x16bOp(num_matrices=4)
        ldsm_trans_op = warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True)
        stsm_trans_op = warp.StMatrix8x8x16bOp(num_matrices=4, transpose=True)
        ldsm_atom = cute.make_copy_atom(ldsm_op, BFloat16)
        stsm_atom = cute.make_copy_atom(stsm_op, BFloat16)
        ldsm_trans_atom = cute.make_copy_atom(ldsm_trans_op, BFloat16)
        stsm_trans_atom = cute.make_copy_atom(stsm_trans_op, BFloat16)

        cp_op = cute.nvgpu.CopyUniversalOp()
        cp_8B_atom = cute.make_copy_atom(cp_op, Int32, num_bits_per_copy=64)
        cp_16B_atom = cute.make_copy_atom(cp_op, Int32, num_bits_per_copy=128)

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tma_empty_mbar + i, 2)  # MMA and TMA store
                    cute.arch.mbarrier_init(inv_mbar + i, 128)
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(tmem_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tmem_empty_mbar + i, 128)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(Q_tma_atom)
            cpasync.prefetch_descriptor(K_tma_atom)
            cpasync.prefetch_descriptor(V_tma_atom)
            cpasync.prefetch_descriptor(a_tma_atom)
            cpasync.prefetch_descriptor(Q_decay_tma_atom)
            cpasync.prefetch_descriptor(K_decay_tma_atom)
        cute.arch.sync_threads()

        num_global_chunks = total_chunks[0]
        if warp_id == 9:
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

        elif warp_id == 8:
            # MMA warp
            _tcgen05.alloc(taddr)

            tma_stage = 0
            tma_parity = 0

            tmem_stage = 0
            tmem_parity = 1

            idesc = _tcgen05.make_bf16_idesc(head_dim, BT, transpose_A=True)

            # LBO is ignored for K-major
            sdesc_template = _tcgen05.make_sdesc_128B_swizzle((BT * 4) * 128)

            for global_chunk_id in range(bid, num_global_chunks, grid_x):
                w_tmem = tmem + (BT * 4) * 2 * tmem_stage
                u_tmem = w_tmem + BT * 4

                k_addr = sK[None, None, tma_stage].iterator.toint()
                v_addr = sV[None, None, tma_stage].iterator.toint()
                k_desc = sdesc_template | (k_addr >> 4)
                v_desc = sdesc_template | (v_addr >> 4)

                # assumed layout: (16, 64) with 128B swizzling
                aib_addr = inv_workspace.iterator.toint()
                aib_desc = sdesc_template | (aib_addr >> 4)

                cute.arch.mbarrier_wait(tmem_empty_mbar + tmem_stage, tmem_parity)
                cute.arch.mbarrier_wait(tma_full_mbar + tma_stage, tma_parity)
                cute.arch.mbarrier_wait(inv_mbar + tma_stage, tma_parity)
                _tcgen05.fence_after_thread_sync()

                # W.T = K_decay.T @ Aib.T
                # U.T = V.T @ Aib.T
                # each MMA instruction is exactly MMA_K=BT=16 (32B)
                # TODO: separate W and U mbar
                with cute.arch.elect_one():
                    for i in cutlass.range_constexpr(4):
                        _tcgen05.mma_f16(
                            w_tmem + i * BT, k_desc, aib_desc, idesc, False
                        )
                        _tcgen05.mma_f16(
                            u_tmem + i * BT, v_desc, aib_desc, idesc, False
                        )
                        k_desc += (BT * 128) >> 4
                        v_desc += (BT * 128) >> 4
                        aib_desc += 32 >> 4
                    _tcgen05.commit(tma_empty_mbar + tma_stage)
                    _tcgen05.commit(tmem_full_mbar + tmem_stage)

                tmem_stage = (tmem_stage + 1) % 2
                if tmem_stage == 0:
                    tmem_parity ^= 1

                tma_stage = (tma_stage + 1) % num_stages
                if tma_stage == 0:
                    tma_parity ^= 1

            cute.arch.mbarrier_wait(tmem_empty_mbar + tmem_stage, tmem_parity)
            _tcgen05.dealloc()

        elif warp_id >= 4:
            # INV warps
            warp_id_ = warp_id % 4
            tid_ = tid % 128

            stage_id = 0
            parity = 0

            # pre-compute ldmatrix addresses (A operand)
            # shape before: (BT, (64, head_dim/64), num_stages)
            # shape after: ((16,4), ((8,2), (4, head_dim/64)), num_stages)
            sQ_ldsm = cute.logical_divide(sQ, (16, cute.make_layout((8, 2)), None))
            sK_ldsm = cute.logical_divide(sK, (16, cute.make_layout((8, 2)), None))

            # shape: (8, (4, head_dim/64), num_stages)
            sQ_ldsm = sQ_ldsm[
                (lane_id % 16, warp_id_), ((None, lane_id // 16), None), None
            ]
            sK_ldsm = sK_ldsm[
                (lane_id % 16, warp_id_), ((None, lane_id // 16), None), None
            ]

            # B operand
            # shape before: (BT, (64, head_dim/64))
            # shape after: (((8,2),4), ((8,2), (4, head_dim/64)))
            sK_right_ldsm = cute.logical_divide(
                sK_right, (cute.make_layout((8, 2)), cute.make_layout((8, 2)))
            )

            # shape: (8, (4, head_dim/64))
            sK_right_ldsm = sK_right_ldsm[
                ((lane_id % 8, lane_id // 16), warp_id_),
                ((None, (lane_id // 8) % 2), None),
            ]

            # each warp handles BT
            sQ_thr = cute.local_tile(sQ, (BT, 4, num_stages), (warp_id_, lane_id, 0))
            sK_thr = cute.local_tile(sK, (BT, 4, num_stages), (warp_id_, lane_id, 0))
            sa_thr = cute.local_tile(sa, (BT, 4, num_stages), (warp_id_, lane_id, 0))
            sK_right_thr = cute.local_tile(sK_right, (BT, 4), (warp_id_, lane_id))

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

            # pre-compute address for inv workspace
            # original shape: (16, (8, 2), 4)
            sA_ldsm = inv_workspace[lane_id % 16, (None, lane_id // 16), warp_id_]

            # we use (16,64) w/ 128B swizzling layout for MMA
            # alias memory with inv workspace
            sAib = cute.make_tensor(
                inv_workspace.iterator,
                layout=cute.make_composed_layout(
                    cute.make_swizzle(3, 3, 3),
                    0,
                    cute.make_layout(((8, 2, 4), 16)),
                ),
            )
            sAib_ldsm = sAib[(None, lane_id // 16, warp_id_), lane_id % 16]

            for global_chunk_id in range(bid, num_global_chunks, grid_x):
                seq_id = chunk_indices[global_chunk_id, 0]
                chunk_id = chunk_indices[global_chunk_id, 1]
                bos = cu_seqlens[seq_id]
                eos = cu_seqlens[seq_id + 1]
                chunk_size = eos - chunk_id * (BT * 4)

                # pre-compute beta
                if tid_ < BT * 4:
                    beta = Float32(0.0)
                    if tid_ < chunk_size:
                        b_ = b[bos + chunk_id * (BT * 4) + tid_, head_id]
                        beta = sigmoid(b_.to(Float32))
                    sbeta[tid_] = beta

                g_cu = cute.make_rmem_tensor(4, Float32)
                g_cu.fill(0.0)

                # TODO: separate QKVa into a->QK->V
                if warp_id_ == 0:
                    cute.arch.mbarrier_wait(tma_full_mbar + stage_id, parity)
                cute.arch.barrier(barrier_id=1, number_of_threads=128)

                ##### stage 1: compute gate and scale Q/K #####
                for i in cutlass.range_constexpr(BT):
                    if warp_id_ * BT + i < chunk_size:
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
                            g_cu[j] += lower_bound * sigmoid(
                                A_ * (a_f32[j] + dt_bias_thr[j])
                            )
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
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                fence_before_tma_store()
                if warp_id_ == 3:
                    Q_dst = cute.local_tile(
                        tmaQ_decay[None, head_id, None],
                        tiler=(BT * 4, head_dim),
                        coord=(global_chunk_id, 0),
                    )
                    K_dst = cute.local_tile(
                        tmaK_decay[None, head_id, None],
                        tiler=(BT * 4, head_dim),
                        coord=(global_chunk_id, 0),
                    )
                    simple_tma_copy(Q_decay_tma_atom, sQ[None, None, stage_id], Q_dst)
                    simple_tma_copy(K_decay_tma_atom, sK[None, None, stage_id], K_dst)
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_commit_group()

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

                ##### stage 3: mask qk and kkt #####
                # P = causal(QK)
                # use sA to convert ldmatrix layout to normal layout
                p_f32 = qk.load().reshape((2, 2, 2))
                p_f32 = cute.where(row_indices >= col_indices, p_f32, 0.0)
                p_bf16 = cute.make_rmem_tensor(8, BFloat16)
                p_bf16.store(p_f32.to(BFloat16))
                cute.copy(stsm_atom, p_bf16, sA_ldsm)
                cute.arch.sync_warp()
                t_ = bos + (chunk_id * 4 + warp_id_) * BT + (lane_id % 16)
                if t_ < eos:
                    cute.copy(cp_16B_atom, sA_ldsm, p_bf16)  # smem->rmem
                    p_dst = cute.local_tile(
                        P[t_, head_id, None], (8,), (lane_id // 16,)
                    )
                    cute.copy(cp_16B_atom, p_bf16, p_dst)  # rmem->gmem

                # multiply kkt by beta
                beta_row = cute.make_rmem_tensor(2, Float32)
                beta_row[0] = sbeta[warp_id_ * BT + (lane_id // 4)]
                beta_row[1] = sbeta[warp_id_ * BT + (lane_id // 4) + 8]
                beta_row = beta_row.load().reshape((1, 2, 1))

                # strict lower mask
                A_f32 = kkt.load().reshape((2, 2, 2)) * beta_row
                A_f32 = cute.where(row_indices > col_indices, A_f32, 0.0)
                A_bf16 = cute.make_rmem_tensor(8, BFloat16)
                A_bf16.store(A_f32.to(BFloat16))
                cute.copy(stsm_atom, A_bf16, sA_ldsm)
                cute.arch.sync_warp()

                ##### stage 4: Newton-Schulz inverse for inv(I+A) #####
                #   Ai_new = 2 Ai - Ai @ M @ Ai
                #   where M = I + A
                # TODO: make this a helper to share with GDN?
                zeros_f32 = cute.make_rmem_tensor(4, Float32)
                zeros_f32.fill(0.0)

                def set_diagonal(A: cute.Tensor, lane_id: Int32):
                    "Set the diagonal to 1s"
                    if lane_id % 9 == 0:
                        A[0] = (A[0] & Uint32(0xFFFF0000)) | Uint32(0x00003F80)
                        A[3] = (A[3] & Uint32(0xFFFF0000)) | Uint32(0x00003F80)
                    elif lane_id % 9 == 4:
                        A[0] = (A[0] & Uint32(0x0000FFFF)) | Uint32(0x3F800000)
                        A[3] = (A[3] & Uint32(0x0000FFFF)) | Uint32(0x3F800000)

                Ai_bf16 = cute.make_rmem_tensor(8, BFloat16)
                mma_B_bf16 = cute.make_rmem_tensor(8, BFloat16)
                M_bf16 = cute.make_rmem_tensor(8, BFloat16)
                acc = cute.make_rmem_tensor((4, 2), Float32)

                # share the same storage
                Ai = cute.recast_tensor(Ai_bf16, Uint32)
                mma_B = cute.logical_divide(cute.recast_tensor(mma_B_bf16, Uint32), 2)
                M = cute.logical_divide(cute.recast_tensor(M_bf16, Uint32), 2)

                # initial guess: Ai = I-A
                cute.copy(ldsm_atom, sA_ldsm, Ai_bf16)
                for i in cutlass.range_constexpr(4):
                    Ai[i] ^= Uint32(0x80008000)  # negate A
                set_diagonal(Ai, lane_id)

                # (4, 2)
                Ai_f32 = cute.logical_divide(cvt.bf16x2_to_fp32x2(Ai), 4)

                # M is holding -(I+A), stay constant throughout the iterations
                cute.copy(ldsm_trans_atom, sA_ldsm, M_bf16)
                set_diagonal(M, lane_id)
                for i in cutlass.range_constexpr(4):
                    M[i] ^= Uint32(0x80008000)

                # 3 rounds of Newton-Schulz
                for _ in cutlass.range_constexpr(3):
                    # First MMA: -AiM = Ai @ (-M)
                    cute.copy(stsm_atom, Ai_bf16, sA_ldsm)
                    cute.arch.sync_warp()
                    acc[None, 0] = mma_bf16(Ai, M[None, 0], zeros_f32)
                    acc[None, 1] = mma_bf16(Ai, M[None, 1], zeros_f32)
                    Ai_bf16.store(acc.load().to(BFloat16))

                    # Second MMA: Ai_new = 2Ai + (-AiM) @ Ai
                    for j in cutlass.range_constexpr(8):
                        Ai_f32[j] *= 2.0
                    cute.copy(ldsm_trans_atom, sA_ldsm, mma_B_bf16)
                    Ai_f32[None, 0] = mma_bf16(Ai, mma_B[None, 0], Ai_f32[None, 0])
                    Ai_f32[None, 1] = mma_bf16(Ai, mma_B[None, 1], Ai_f32[None, 1])
                    Ai_bf16.store(Ai_f32.load().to(BFloat16))

                # beta scaling
                beta_col = cute.make_rmem_tensor(4, Float32)
                beta_col[0] = sbeta[warp_id_ * BT + (lane_id % 4) * 2 + 0]
                beta_col[1] = sbeta[warp_id_ * BT + (lane_id % 4) * 2 + 1]
                beta_col[2] = sbeta[warp_id_ * BT + (lane_id % 4) * 2 + 8]
                beta_col[3] = sbeta[warp_id_ * BT + (lane_id % 4) * 2 + 9]
                beta_col = beta_col.load().reshape((2, 1, 2))

                Aib_f32 = Ai_f32.load().reshape((2, 2, 2)) * beta_col
                Aib_bf16 = cute.make_rmem_tensor(8, BFloat16)
                Aib_bf16.store(Aib_f32.to(BFloat16))

                # store to smem for tcgen05
                # barrier is required since we alias sAib with inv_workspace
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                cute.copy(stsm_atom, Aib_bf16, sAib_ldsm)
                fence_before_tma_store()
                _tcgen05.fence_before_thread_sync()
                cute.arch.mbarrier_arrive(inv_mbar + stage_id)

                # we use Q/K TMA buffers to do TMA store. hence, we must
                # wait for Q/K TMA store to finish to release the buffer.
                if warp_id_ == 3:
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        cute.arch.mbarrier_arrive(tma_empty_mbar + stage_id)

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

        else:
            # epilogue warps
            stage_id = 0
            parity = 0

            # pre-compute ldmatrix address
            # shape before: (BT*4, (64, head_dim/64))
            # shape after: (((8,2), BT*4/16), ((8,2,2), (2, head_dim/64)))
            tiler_16x32 = (cute.make_layout((8, 2)), cute.make_layout((8, 2, 2)))
            sU_stsm = cute.logical_divide(sU, tiler_16x32)
            sW_stsm = cute.logical_divide(sW, tiler_16x32)

            # shape: (BT/4, 8, 2)
            sU_stsm = sU_stsm[
                ((lane_id % 8, lane_id // 16), None),
                ((None, (lane_id // 8) % 2, None), warp_id),
            ]
            sW_stsm = sW_stsm[
                ((lane_id % 8, lane_id // 16), None),
                ((None, (lane_id // 8) % 2, None), warp_id),
            ]

            # swap modes for cute.copy()
            sU_stsm = cute.make_tensor(
                sU_stsm.iterator, cute.select(sU_stsm.layout, mode=[1, 0, 2])
            )
            sW_stsm = cute.make_tensor(
                sW_stsm.iterator, cute.select(sW_stsm.layout, mode=[1, 0, 2])
            )

            for global_chunk_id in range(bid, num_global_chunks, grid_x):
                w_tmem = tmem + (BT * 4) * 2 * stage_id
                u_tmem = w_tmem + BT * 4

                # wait for UW MMA and previous UW TMA store to finish
                if warp_id == 0:
                    cute.arch.mbarrier_wait(tmem_full_mbar + stage_id, parity)
                elif warp_id == 3:
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                _tcgen05.fence_after_thread_sync()

                for i in cutlass.range_constexpr(2):
                    # there are (BT * 4) columns. 16x256b covers 8 columns
                    t_row = warp_id * 32 + i * 16
                    w_f32 = _tcgen05.ld(t_row, w_tmem, "16x256b", BT * 4 // 8)

                    # pack to BF16
                    w_bf16 = cute.make_rmem_tensor((8, BT // 4), BFloat16)
                    w_bf16.store(w_f32.to(BFloat16))
                    cute.copy(stsm_trans_atom, w_bf16, sW_stsm[None, None, i])

                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                fence_before_tma_store()
                if warp_id == 3:
                    W_dst = cute.local_tile(
                        tmaW[None, head_id, None],
                        tiler=(BT * 4, head_dim),
                        coord=(global_chunk_id, 0),
                    )
                    simple_tma_copy(W_tma_atom, sW, W_dst)

                for i in cutlass.range_constexpr(2):
                    # there are (BT * 4) columns. 16x256b covers 8 columns
                    t_row = warp_id * 32 + i * 16
                    u_f32 = _tcgen05.ld(t_row, u_tmem, "16x256b", BT * 4 // 8)

                    if cutlass.const_expr(i == 1):
                        cute.arch.mbarrier_arrive(tmem_empty_mbar + stage_id)

                    # pack to BF16
                    u_bf16 = cute.make_rmem_tensor((8, BT // 4), BFloat16)
                    u_bf16.store(u_f32.to(BFloat16))
                    cute.copy(stsm_trans_atom, u_bf16, sU_stsm[None, None, i])

                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                fence_before_tma_store()
                if warp_id == 3:
                    U_dst = cute.local_tile(
                        tmaU[None, head_id, None],
                        tiler=(BT * 4, head_dim),
                        coord=(global_chunk_id, 0),
                    )
                    simple_tma_copy(U_tma_atom, sU, U_dst)
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_commit_group()

                stage_id = (stage_id + 1) % 2
                if stage_id == 0:
                    parity ^= 1

    @cache
    @staticmethod
    def compile(head_dim: int, num_heads: int, num_stages: int = 2):
        total_t = cute.sym_int()
        pad_t = cute.sym_int()

        Q = make_fake_tensor(BFloat16, (total_t, num_heads, head_dim), divisibility=16)
        K = make_fake_tensor(BFloat16, (total_t, num_heads, head_dim), divisibility=16)
        V = make_fake_tensor(BFloat16, (total_t, num_heads, head_dim), divisibility=16)
        a = make_fake_tensor(BFloat16, (total_t, num_heads, head_dim), divisibility=16)
        b = make_fake_tensor(BFloat16, (total_t, num_heads), divisibility=1)
        A_log = make_fake_tensor(Float32, (num_heads,), divisibility=4)
        dt_bias = make_fake_tensor(Float32, (num_heads, head_dim), divisibility=4)
        Q_decay = make_fake_tensor(
            BFloat16, (pad_t, num_heads, head_dim), divisibility=16
        )
        K_decay = make_fake_tensor(
            BFloat16, (pad_t, num_heads, head_dim), divisibility=16
        )
        U = make_fake_tensor(BFloat16, (pad_t, num_heads, head_dim), divisibility=16)
        W = make_fake_tensor(BFloat16, (pad_t, num_heads, head_dim), divisibility=16)
        P = make_fake_tensor(
            BFloat16,
            (total_t, num_heads, KDAChunkPreRecurrentKernel.BT),
            divisibility=16,
        )
        cu_seqlens = make_fake_tensor(Int32, (cute.sym_int(),), divisibility=1)
        chunk_indices = make_fake_tensor(Int32, (cute.sym_int(), 2), divisibility=2)
        total_chunks = make_fake_tensor(Int32, (1,), divisibility=1)

        kernel = KDAChunkPreRecurrentKernel(head_dim, num_heads, num_stages)
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
            Q_decay,
            K_decay,
            U,
            W,
            P,
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
    Q_decay: torch.Tensor,
    K_decay: torch.Tensor,
    U: torch.Tensor,
    W: torch.Tensor,
    P: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    total_chunks: torch.Tensor,
    scale: float | None = None,
    num_sms: int | None = None,
    num_stages: int = 2,
) -> None:
    _, num_heads, head_dim = Q.shape

    if scale is None:
        scale = K.shape[-1] ** -0.5
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(Q.device).multi_processor_count

    KDAChunkPreRecurrentKernel.compile(head_dim, num_heads, num_stages)(
        Q,
        K,
        V,
        a,
        b,
        A_log,
        dt_bias,
        Q_decay,
        K_decay,
        U,
        W,
        P,
        cu_seqlens,
        chunk_indices,
        total_chunks,
        scale,
        num_sms,
    )


def pre_recurrent_pytorch_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    BT = KDAChunkPreRecurrentKernel.BT
    chunk_size = BT * 4
    total_tokens, num_heads, head_dim = Q.shape
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    pad_t = sum(
        (int(length) + chunk_size - 1) // chunk_size * chunk_size for length in lengths
    )

    Q_ref = torch.zeros(
        pad_t, num_heads, head_dim, dtype=torch.bfloat16, device=Q.device
    )
    K_ref = torch.zeros_like(Q_ref)
    U_ref = torch.zeros_like(Q_ref)
    W_ref = torch.zeros_like(Q_ref)
    P_ref = torch.zeros(
        total_tokens, num_heads, BT, dtype=torch.bfloat16, device=Q.device
    )

    gate_scale = A_log.float().exp().view(1, num_heads, 1)
    gate_bias = dt_bias.float().view(1, num_heads, head_dim)
    beta = b.float().sigmoid()
    eye = torch.eye(BT, dtype=torch.float32, device=Q.device)

    padded_start = 0
    for seq_id, seq_len_t in enumerate(lengths):
        seq_len = int(seq_len_t)
        bos = int(cu_seqlens[seq_id].item())
        for tile_start in range(0, seq_len, BT):
            tile_len = min(BT, seq_len - tile_start)
            token_start = bos + tile_start
            out_start = padded_start + tile_start
            token_slice = slice(token_start, token_start + tile_len)
            out_slice = slice(out_start, out_start + tile_len)

            g = KDAChunkPreRecurrentKernel.lower_bound * torch.sigmoid(
                gate_scale * (a[token_slice].float() + gate_bias)
            )
            g_cu = g.cumsum(dim=0)
            decay = g_cu.exp()

            q_decay = (Q[token_slice].float() * (decay * scale)).to(torch.bfloat16)
            k_decay = (K[token_slice].float() * decay).to(torch.bfloat16)
            k_right = (K[token_slice].float() / decay).to(torch.bfloat16)

            Q_ref[out_slice] = q_decay
            K_ref[out_slice] = k_decay

            for head_id in range(num_heads):
                q_h = q_decay[:, head_id].float()
                k_h = k_decay[:, head_id].float()
                kr_h = k_right[:, head_id].float()
                v_h = V[token_slice, head_id].float()
                beta_h = beta[token_slice, head_id]

                p = torch.tril(q_h @ kr_h.T)
                P_ref[token_slice, head_id, :tile_len] = p.to(torch.bfloat16)

                kk = k_h @ kr_h.T
                strict_l = torch.tril(kk * beta_h[:, None], diagonal=-1)
                inv = torch.linalg.inv(eye[:tile_len, :tile_len] + strict_l)
                aib = (inv * beta_h).to(torch.bfloat16).float()
                U_ref[out_slice, head_id] = (aib @ v_h).to(torch.bfloat16)
                W_ref[out_slice, head_id] = (aib @ k_h).to(torch.bfloat16)

        padded_start += (seq_len + chunk_size - 1) // chunk_size * chunk_size

    return Q_ref, K_ref, U_ref, W_ref, P_ref


def check_pre_recurrent_matches_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    Q_decay: torch.Tensor,
    K_decay: torch.Tensor,
    U: torch.Tensor,
    W: torch.Tensor,
    P: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
) -> bool:
    refs = pre_recurrent_pytorch_reference(
        Q, K, V, a, b, A_log, dt_bias, cu_seqlens, scale
    )

    BT = KDAChunkPreRecurrentKernel.BT
    chunk_size = BT * 4
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    valid_rows = torch.zeros(Q_decay.shape[0], dtype=torch.bool, device=Q.device)
    p_mask = torch.zeros_like(P, dtype=torch.bool)
    padded_start = 0
    for seq_id, seq_len_t in enumerate(lengths):
        seq_len = int(seq_len_t)
        bos = int(cu_seqlens[seq_id].item())
        for tile_start in range(0, seq_len, BT):
            tile_len = min(BT, seq_len - tile_start)
            token_start = bos + tile_start
            out_start = padded_start + tile_start
            token_slice = slice(token_start, token_start + tile_len)
            out_slice = slice(out_start, out_start + tile_len)
            valid_rows[out_slice] = True
            p_mask[token_slice, :, :tile_len] = True
        padded_start += (seq_len + chunk_size - 1) // chunk_size * chunk_size

    def report(
        name: str, actual: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor
    ) -> bool:
        actual_f = actual.float()[mask]
        ref_f = ref.float()[mask]
        print(f"{name}:")
        if actual_f.numel() == 0:
            print("  status: SKIP")
            print("  reason: no valid elements checked")
            return True

        abs_diff = (actual_f - ref_f).abs()
        allowed = 7e-2 + 7e-2 * ref_f.abs()
        bad = abs_diff > allowed
        max_abs, flat_idx = abs_diff.max(dim=0)
        ref_abs = ref_f.abs()
        rel_diff = abs_diff / torch.where(
            ref_abs > 0, ref_abs, torch.ones_like(ref_abs)
        )
        max_rel = rel_diff.max()
        ok = not bool(bad.any())
        status = "PASS" if ok else "FAIL"
        bad_count = int(bad.sum().item())
        total = actual_f.numel()

        bad_pct = 100.0 * bad_count / total
        print(f"  status: {status}")
        print(f"  bad: {bad_count}/{total} ({bad_pct:.2f}%)")
        print(
            f"  max_abs: {float(max_abs.item()):.6g}, "
            f"max_rel: {float(max_rel.item()):.6g}"
        )
        print(
            f"  absmax: actual={float(actual_f.abs().max().item()):.6g}, "
            f"ref={float(ref_f.abs().max().item()):.6g}"
        )
        if not ok:
            flat_idx_i = int(flat_idx.item())
            checked_indices = torch.nonzero(mask, as_tuple=False)
            original_idx = tuple(int(v) for v in checked_indices[flat_idx_i].tolist())
            print(
                f"  worst: idx={original_idx}, "
                f"actual={float(actual_f[flat_idx_i].item()):.6g}, "
                f"ref={float(ref_f[flat_idx_i].item()):.6g}, "
                f"abs={float(abs_diff[flat_idx_i].item()):.6g}, "
                f"rel={float(rel_diff[flat_idx_i].item()):.6g}"
            )
        return ok

    print("pre-recurrent correctness")
    ok = True
    valid_rows_3d = valid_rows[:, None, None].expand_as(Q_decay)
    ok &= report("Q_decay", Q_decay, refs[0], valid_rows_3d)
    ok &= report("K_decay", K_decay, refs[1], valid_rows_3d)
    ok &= report("U", U, refs[2], valid_rows_3d)
    ok &= report("W", W, refs[3], valid_rows_3d)
    ok &= report("P", P, refs[4], p_mask)
    return ok


def benchmark_pre_recurrent(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    Q_decay: torch.Tensor,
    K_decay: torch.Tensor,
    U: torch.Tensor,
    W: torch.Tensor,
    P: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    total_chunks: torch.Tensor,
    scale: float,
) -> float:
    def launch() -> None:
        kkt_qk_cutedsl(
            Q,
            K,
            V,
            a,
            b,
            A_log,
            dt_bias,
            Q_decay,
            K_decay,
            U,
            W,
            P,
            cu_seqlens,
            chunk_indices,
            total_chunks,
            scale=scale,
        )

    timings = bench_gpu_time_with_cupti(launch)
    return statistics.median(timings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile and launch kernel_kkt_qk.")
    parser.add_argument("--seqlens", type=int, nargs="+", default=[63, 129, 512, 2045])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_default_device("cuda")
    torch.manual_seed(args.seed)
    seqlens = args.seqlens
    offsets = [0]
    for length in seqlens:
        offsets.append(offsets[-1] + length)

    cu_seqlens = torch.tensor(offsets, dtype=torch.int32)
    BT = KDAChunkPreRecurrentKernel.BT
    total_tokens = offsets[-1]
    pad_t = ((torch.tensor(seqlens) + (BT * 4 - 1)) // (BT * 4) * (BT * 4)).sum().item()
    HEAD_DIM = 128
    scale = float(HEAD_DIM**-0.5)
    Q = torch.randn(total_tokens, args.heads, HEAD_DIM, dtype=torch.bfloat16)
    K = torch.randn_like(Q)
    Q = torch.nn.functional.normalize(Q.float(), p=2, dim=-1).to(torch.bfloat16)
    K = torch.nn.functional.normalize(K.float(), p=2, dim=-1).to(torch.bfloat16)
    V = torch.randn_like(Q)
    a = torch.randn_like(Q)
    b = torch.randn(total_tokens, args.heads, dtype=torch.bfloat16)
    A = torch.empty(args.heads, dtype=torch.float32).uniform_(0.0, 16.0)
    A_log = torch.log(A)
    dt = torch.exp(
        torch.rand(args.heads, HEAD_DIM, dtype=torch.float32)
        * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    )
    dt = torch.clamp(dt, min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))

    Q_decay = torch.empty(pad_t, args.heads, HEAD_DIM, dtype=torch.bfloat16)
    K_decay = torch.empty_like(Q_decay)
    U = torch.empty_like(Q_decay)
    W = torch.empty_like(Q_decay)
    P = torch.empty(total_tokens, args.heads, BT, dtype=torch.bfloat16)

    chunk_indices = make_chunk_indices(cu_seqlens, chunk_size=BT * 4)
    total_chunks = torch.tensor([chunk_indices.shape[0]], dtype=torch.int32)

    kkt_qk_cutedsl(
        Q,
        K,
        V,
        a,
        b,
        A_log,
        dt_bias,
        Q_decay,
        K_decay,
        U,
        W,
        P,
        cu_seqlens,
        chunk_indices,
        total_chunks,
        scale=scale,
    )
    torch.accelerator.synchronize()

    check_passed = check_pre_recurrent_matches_reference(
        Q,
        K,
        V,
        a,
        b,
        A_log,
        dt_bias,
        Q_decay,
        K_decay,
        U,
        W,
        P,
        cu_seqlens,
        scale,
    )

    mean_ms = benchmark_pre_recurrent(
        Q,
        K,
        V,
        a,
        b,
        A_log,
        dt_bias,
        Q_decay,
        K_decay,
        U,
        W,
        P,
        cu_seqlens,
        chunk_indices,
        total_chunks,
        scale,
    )
    mean_us = mean_ms * 1e3
    tokens_per_s = total_tokens / (mean_ms * 1e-3)
    chunks_per_s = chunk_indices.shape[0] / (mean_ms * 1e-3)
    print("pre-recurrent benchmark")
    print(f"  mean: {mean_us:.3f} us")
    print(f"  tokens/s: {tokens_per_s:.6g}")
    print(f"  chunks/s: {chunks_per_s:.6g}")

    print(
        "launched kernel_kkt_qk",
        f"total_tokens={total_tokens}",
        f"heads={args.heads}",
        f"chunks={chunk_indices.shape[0]}",
        f"check={'passed' if check_passed else 'failed'}",
    )


if __name__ == "__main__":
    main()

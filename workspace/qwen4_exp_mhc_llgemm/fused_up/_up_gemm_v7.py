# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""v7 mHC UP GEMM prototype: tcgen05, MMA_M=128, one accumulator per CTA.

Design (from nvjet SASS inspection + discussion):

- swap-AB: channels on tcgen05-M (128), tokens on N (mma_n buckets).
- MMA_M=128 packs all 4 streams x 32 channels into ONE accumulator
  (streams share the same x[M,320] operand, so packing is free -- no
  block-diagonal waste). Lane L of the accumulator holds
  stream (L%32)//8, channel (L//32)*8 + L%8 (+ cta channel base).
- grid.x = 10240/128 = 80 CTAs (matches nvjet's dense tiling; grid.y covers
  M > mma_n).
- K=320 = 5 blocks of BK=64, EXACTLY 5 smem stages: the ring never wraps,
  so there are no empty barriers, no stage release, no phase toggles.
  Fully unrolled fire-and-forget mainloop.
- 4 warps / 128 threads total (8 in the fused epilogue for MMA_N >= 64),
  no warp-specialized epilogue:
    warp 0: mbarrier init, TMA weights (no PDL wait), PDL wait, TMA
            activations + xn, launch_dependents.
    warp 1: TMA descriptor prefetch (overlaps warp 0's init), tmem alloc
            (full 512 cols, base = 0), MMA (5 stage-arrival waits x 4
            UTCHMMA of K=16), one commit + drain wait at the end.
    then bar.sync, all warps run the epilogue.
- Weights are loaded with a 4D TMA view (K, c_lo=8, s=4, c_hi) of the
  [10240, 320] weight matrix: one box (64, 8, 4, 4) per stage fills the
  128-row tile 8-row-stream-interleaved, so the gate-mix epilogue mixes all
  4 streams intra-warp (butterfly shuffles), and the GEMM lane mapping
  above falls out of the TMA fill order for free.

Two epilogues:
- gemm_only: plain store of raw f32->bf16 accs to out[M, 10240] (probe
  scaffold for nvjet comparisons).
- fused (default for production): sigmoid gate-mix. The GEMM produces gate
  logits acc_s[M, 2560] per stream; out[m, ch] = (1/HC) * sum_s
  sigmoid(bf16(acc_s[m, ch])) * xn_s[m, ch], production accumulation order
  ((s0+s1)+s2)+s3 exactly. Lane L holds stream L//8 (of the warp-local 32),
  channel (L//... ) -- the 8-row interleave makes each warp own all 4
  streams for 8 channels, so the mix is 3 butterfly shuffles (xor 8/16/24),
  no cross-warp traffic. xn is TMA'd into smem during the mainloop (one
  (8, HC, 4, MMA_N) box, channel-major so epilogue reads are conflict-free).
  The mixed [MMA_N, 32] tile is staged in smem and stored with one TMA
  store box per CTA (32 contiguous channels x MMA_N tokens; streams
  already collapsed).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Boolean, Float32, Int32, Int64
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import while_generate, yield_out

from vllm.cute_utils import _tcgen05, simple_tma_copy


def _spin_wait(bar):
    """Non-suspending mbarrier wait: tight mbarrier.test_wait spin.

    mbarrier_wait lowers to mbarrier.try_wait, which may suspend the warp;
    the wake latency after the last TMA stage lands sits on the critical
    path. test_wait never suspends, so the spinning warp observes the phase
    flip within a few cycles.
    """
    with while_generate([Boolean(False)], lambda done: done == Boolean(False)) as (
        done,
    ):
        done = cute.arch.mbarrier_test_wait(bar, 0)
        yield_out([done])


def _sigmoid_f32(x):
    return 1.0 / (1.0 + cute.math.exp(x * (-1.0)))


def _sigmoid_f32_tanh(x):
    # 1 MUFU.TANH (tanh.approx.f32) instead of MUFU.EX2 + MUFU.RCP.
    return 0.5 * cute.math.tanh(0.5 * x, approx=True) + 0.5


class UpGemmV7Kernel:
    """mHC UP GEMM [M,320] @ [10240,320]^T -> [M,10240], MMA_M=128.

    One compilation per (mma_n, use_pdl) config; M dynamic <= mma_n * grid.y.
    """

    def __init__(
        self,
        k: int = 320,
        n: int = 10240,
        hc: int = 4,
        mma_n: int = 32,  # padded tokens per tile (tcgen05 N)
        bk: int = 64,
        use_pdl: bool = False,
        sigmoid_mode: str = "tanh",  # "exact" (exp+rcp) or "tanh" (approx)
        gemm_only: bool = True,  # plain GEMM epilogue: store raw accs [M, N]
        weight_evict_first: bool = False,  # L2 EVICT_FIRST on weight TMA
        pdl_wait_first: bool = False,  # probe: PDL wait before weight TMA
        xn_pre_wait: bool = False,  # probe: issue xn TMA before the PDL wait
        spin_wait: bool = False,  # probe: non-suspending test_wait spins
        probe_phase: str = "full",  # "full" | "no_epi" | "stream" (gemm_only)
        epi_split_override: int | None = None,  # probe: force epilogue split
    ):
        self.k = k
        self.n = n
        self.hc = hc
        self.hc_dim = n // hc
        self.mma_n = mma_n
        self.bk = bk
        self.use_pdl = use_pdl
        self.sigmoid_mode = sigmoid_mode
        self.gemm_only = gemm_only
        # v7 reads each weight slice exactly once (80 CTAs x unique 128-row
        # slice, no cross-CTA reuse), so EVICT_FIRST only frees L2 for the
        # activations/xn that ARE shared. (Unlike v5, where token tiles
        # re-read weights and the hint was harmful.)
        self.weight_evict_first = weight_evict_first
        # Probe only: move the PDL wait BEFORE the weight stream (kills the
        # pre-streaming overlap). Default off: weights issue pre-wait so they
        # stream during the predecessor kernel. Ignored when use_pdl=False.
        self.pdl_wait_first = pdl_wait_first
        # Probe: issue the xn TMA together with the weight stream, BEFORE the
        # PDL wait. xn is the chain input (it predates the down kernel), so
        # this is safe as long as the up kernel's launch cannot overlap xn's
        # producer — i.e. only when the down kernel fires launch_dependents
        # after its own griddepcontrol_wait, or in the 2-kernel bench chain
        # where down is the first node. Hides the xn read latency that the
        # fused epilogue otherwise pays after the wait.
        self.xn_pre_wait = xn_pre_wait
        # Probe: replace the mbarrier_wait (suspending try_wait) on the
        # MMA-critical barriers with non-suspending test_wait spins.
        self.spin_wait = spin_wait
        # Perf-probe only (requires gemm_only): "stream" = W+A TMA, no
        # MMA/epilogue; "no_epi" = MMA but no epilogue; "serial_mma" = wait
        # for ALL stages, then issue all MMAs (isolates MMA throughput);
        # "mma_nowait" = issue all MMAs on stage-0 smem with no stage waits
        # (garbage math; measures issue+drain overlap). Used to decompose
        # where the gap above the pure-stream line goes.
        assert probe_phase in (
            "full",
            "no_epi",
            "stream",
            "serial_mma",
            "mma_nowait",
            "commit_only",
            "no_fence",
            "late_mma_1",
            "late_mma_4",
        )
        assert probe_phase == "full" or gemm_only
        self.probe_phase = probe_phase
        assert sigmoid_mode in ("exact", "tanh")
        assert k == 320 and bk == 64 and hc == 4
        assert n % (hc * 32) == 0
        assert mma_n % 8 == 0 and mma_n <= 192
        self.k_tiles = k // bk  # == 5: stages never wrap
        self.ch_per_cta = 32  # channels per stream per CTA (128 lanes / 4)
        self.mma_k_blocks = bk // 16  # 4 UTCHMMA of K=16 per stage
        # Epilogue warpgroups: each thread serially loops over the token
        # tile (~31 ns/token/thread measured), so wide tiles split tokens
        # across warpgroups (warps w, w+4, ... share the same 32 tmem lanes,
        # disjoint token slices). Sweep (probe_v7_episplit.py, fused):
        # EPI4 best at mma_n=32/64, EPI8 at 96/128; bit-identical across
        # splits (each token is mixed by exactly one thread either way).
        # gemm_only and non-"full" probes stay at EPI1: the gemm_only
        # epilogue is tmem-read-bound (EPI warpgroups share the same 32 tmem
        # lanes, so splits don't add read throughput) and 128*EPI threads
        # inflate launch/init for the probe phases.
        if self.gemm_only or probe_phase != "full" or mma_n < 32:
            default_split = 1
        elif mma_n <= 64:
            default_split = 4
        else:
            default_split = 8
        self.epi_split = epi_split_override or default_split
        assert mma_n % self.epi_split == 0
        self.threads_per_cta = 128 * self.epi_split

    @cute.jit
    def __call__(
        self,
        mW: cute.Tensor,  # [N, K] bf16, K-major
        mA: cute.Tensor,  # [M, K] bf16, K-major
        mXn: cute.Tensor,  # [M, N] bf16 (fused only; dead arg in gemm_only)
        mOut: cute.Tensor,  # [M, N] (gemm_only) or [M, N//HC] (fused) bf16
        stream,
    ):
        HC_DIM: cutlass.Constexpr = self.hc_dim
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        elems: cutlass.Constexpr = 128 * 8 // BFloat16.width

        def staged(rows, stages):
            lay = cute.make_layout(
                (rows, (elems, self.bk // elems), stages),
                stride=(elems, (1, rows * elems), rows * self.bk),
            )
            return cute.make_composed_layout(swizzle_128B, 0, lay)

        op = cpasync.CopyBulkTensorTileG2SOp()
        # 4D weight view: (K, c_lo=8, s=4, c_hi) so one TMA box per stage
        # fills the 128-row smem tile 8-row-stream-interleaved:
        # smem row r = c_lo + 8*s + 32*t, c_hi = bid*4 + t.
        # (A 3D stream-innermost variant was A/B'd: bit-identical and
        # perf-neutral -- the weight stream is per-CTA-bytes-bound, not
        # box-shape-bound. See TRACKING.md.)
        mW4 = cute.make_tensor(
            mW.iterator,
            cute.make_layout(
                (self.k, 8, self.hc, HC_DIM // 8),
                stride=(1, self.k, self.k * HC_DIM, self.k * 8),
            ),
        )
        # Rank-5 smem layout matching the 4D box + stage mode; the linear
        # order is row r = c_lo + 8*s + 32*t within a 128-row stage.
        w_slayout = cute.make_composed_layout(
            swizzle_128B,
            0,
            cute.make_layout(
                (elems, 8, self.hc, 4, self.k_tiles),
                stride=(1, 64, 64 * 8, 64 * 32, 64 * 128),
            ),
        )
        W_tma = cpasync.make_tiled_tma_atom(
            op,
            mW4,
            w_slayout,
            (self.bk, 8, self.hc, 4),
        )
        A_tma = cpasync.make_tiled_tma_atom(
            op, mA, staged(self.mma_n, self.k_tiles), (self.mma_n, self.bk)
        )
        # TMA store for the fused epilogue: (32 ch, MMA_N tok) box per CTA
        # over a (channel, token) view of the output. Built in both modes
        # (dead in gemm_only) to keep the kernel signature fixed.
        OUT_COLS: cutlass.Constexpr = self.n if self.gemm_only else HC_DIM
        mOT = cute.make_tensor(
            mOut.iterator,
            cute.make_layout(
                (OUT_COLS, cute.size(mOut, mode=[0])), stride=(1, OUT_COLS)
            ),
        )
        O_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mOT,
            cute.make_layout((32, self.mma_n), stride=(1, 32)),
            (32, self.mma_n),
        )
        # xn TMA (fused only): 4D view (c_lo=8, s=4, c_hi, tok) of the
        # [M, N] gate-values tensor; one box (8, 4, 4, MMA_N) per CTA fills
        # smem channel-major -> lane L reads row L conflict-free.
        mX4 = cute.make_tensor(
            mXn.iterator,
            cute.make_layout(
                (8, self.hc, HC_DIM // 8, cute.size(mXn, mode=[0])),
                stride=(1, HC_DIM, 8, self.n),
            ),
        )
        X_tma = cpasync.make_tiled_tma_atom(
            op,
            mX4,
            cute.make_layout((8, self.hc, 4, self.mma_n), stride=(1, 8, 32, 128)),
            (8, self.hc, 4, self.mma_n),
        )

        smem_bytes = (
            (128 * self.bk + self.mma_n * self.bk) * self.k_tiles * 2
            # fused: xn tile (128*MMA_N) + 32*MMA_N staging; gemm_only:
            # 128*MMA_N staging (no xn).
            + (160 if not self.gemm_only else 128) * self.mma_n * 2
            + (self.k_tiles + 2) * 8
            + 4
            + 1024  # alignment slack
        )
        self.kernel(W_tma, A_tma, O_tma, X_tma, mOut).launch(
            grid=(
                self.n // 128,
                cute.ceil_div(cute.size(mOut, mode=[0]), self.mma_n),
                1,
            ),
            block=(self.threads_per_cta, 1, 1),
            smem=smem_bytes,
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        W_tma: cpasync.TmaInfo,
        A_tma: cpasync.TmaInfo,
        O_tma: cpasync.TmaInfo,
        X_tma: cpasync.TmaInfo,
        mOut: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, bidm, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        MMA_N: cutlass.Constexpr = self.mma_n
        BK: cutlass.Constexpr = self.bk
        KT: cutlass.Constexpr = self.k_tiles
        HC: cutlass.Constexpr = self.hc
        HC_DIM: cutlass.Constexpr = self.hc_dim

        M = cute.size(mOut, mode=[0])

        smem = cutlass.utils.SmemAllocator()
        sW = smem.allocate_tensor(
            BFloat16,
            W_tma.smem_layout.outer,
            byte_alignment=1024,
            swizzle=W_tma.smem_layout.inner,
        )
        sA = smem.allocate_tensor(
            BFloat16,
            A_tma.smem_layout.outer,
            byte_alignment=1024,
            swizzle=A_tma.smem_layout.inner,
        )
        bar_full = smem.allocate_array(Int64, KT)
        bar_mma_done = smem.allocate_array(Int64, 1)
        bar_xn = smem.allocate_array(Int64, 1)
        taddr = smem.allocate(Int32, 4)  # tmem alloc result (unused; base = 0)
        # xn tile (fused only): (8 c_lo, 4 s, 4 c_hi, MMA_N tok), channel-major.
        if cutlass.const_expr(not self.gemm_only):
            sX = smem.allocate_tensor(
                BFloat16,
                cute.make_layout((8, HC, 4, MMA_N), stride=(1, 8, 32, 128)),
                byte_alignment=128,
            )
        # Fused-epilogue staging: (32 ch, MMA_N tok), channels contiguous.
        # gemm_only: (4 s, 32 ch, MMA_N tok) so all 128 rows stage at once
        # (one TMA store box per stream quarter; each lane owns exactly one
        # (s, ch) row). The 4 stream quarter-warps' 16B stores share banks
        # (32*MMA_N*2 B stream stride is bank-aligned) — a 4-way conflict —
        # but padding can't fix it: TMA needs 128B-aligned box bases and the
        # bank period is 128B. Measured cheaper than scalar gmem stores.
        sO = smem.allocate_tensor(
            BFloat16,
            cute.make_layout(
                (HC, 32, MMA_N) if self.gemm_only else (32, MMA_N),
                stride=(32 * MMA_N, 1, 32) if self.gemm_only else (1, 32),
            ),
            byte_alignment=128,
        )

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(KT):
                    cute.arch.mbarrier_init(bar_full + i, 1)
                cute.arch.mbarrier_init(bar_mma_done, 1)
                cute.arch.mbarrier_init(bar_xn, 1)
        elif warp_idx == 1:
            # Descriptor prefetches overlap warp 0's mbarrier init (they are
            # independent L2 reads), so warp 0 can issue TMA immediately
            # after the barrier.
            cpasync.prefetch_descriptor(W_tma.atom)
            cpasync.prefetch_descriptor(A_tma.atom)
        if cutlass.const_expr(not self.gemm_only):
            if warp_idx == 2:
                cpasync.prefetch_descriptor(X_tma.atom)
                cpasync.prefetch_descriptor(O_tma.atom)
        # 1-SM, CTA-local barriers: cp.async.bulk accesses its mbarrier
        # operand via the GENERIC proxy (PTX ISA), so the bar.sync below
        # already publishes the inits -- fence.mbarrier_init is a
        # cluster-scope release and has nothing to do here.
        cute.arch.barrier()

        stage_bytes = Int32((128 + MMA_N) * BK * 2)

        if warp_idx == 0:
            # Issue order is PDL-dependent (one box per stage per operand, 5
            # total, fire-and-forget; the ring never wraps so there is
            # nothing to wait for):
            # - PDL on (chains): all 5 weight boxes pre-wait, so they
            #   pre-stream during the predecessor kernel; PDL wait; then
            #   activations (they read data the predecessor writes).
            # - PDL off (standalone): per-stage W/A interleave, so stage 0
            #   completes after one weight stage instead of the full stream
            #   (~0.1-0.2 us at mid M; neutral at small M).
            # - pdl_wait_first probe: PDL wait BEFORE everything -- kills the
            #   pre-stream overlap; used to attribute the chain speedup.
            pol = Int64(0x12F0000000000000) if self.weight_evict_first else None
            # (BK, 8, HC, 4) box tiles; RestK=KT, RestCHI = grid.x.
            gW_tiles = cute.local_tile(
                W_tma.tma_tensor, (BK, 8, HC, 4), (None, None, None, None)
            )
            # (MMA_N, BK) box tiles; RestM tiles = grid.y, RestK = KT.
            gA_tiles = cute.local_tile(A_tma.tma_tensor, (MMA_N, BK), (None, None))

            def issue_w(kt):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(bar_full + kt, stage_bytes)
                simple_tma_copy(
                    W_tma.atom,
                    gW_tiles[None, None, None, None, kt, 0, 0, bid],
                    sW[None, None, None, None, kt],
                    bar_full + kt,
                    cache_policy=pol,
                )

            def issue_a(kt):
                simple_tma_copy(
                    A_tma.atom,
                    gA_tiles[None, None, bidm, kt],
                    sA[None, None, kt],
                    bar_full + kt,
                )

            def issue_xn():
                # (8, HC, 4, MMA_N) box tiles; rest modes (1, 1, c_hi, tok).
                gX_tiles = cute.local_tile(
                    X_tma.tma_tensor, (8, HC, 4, MMA_N), (None, None, None, None)
                )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar_xn, Int32(128 * MMA_N * 2)
                    )
                simple_tma_copy(
                    X_tma.atom,
                    gX_tiles[None, None, None, None, 0, 0, bid, bidm],
                    sX,
                    bar_xn,
                )

            xn_pre: cutlass.Constexpr = (
                self.xn_pre_wait
                and not self.gemm_only
                and self.use_pdl
                and not self.pdl_wait_first
            )
            if cutlass.const_expr(self.use_pdl and not self.pdl_wait_first):
                for kt in cutlass.range_constexpr(KT):
                    issue_w(kt)
                if cutlass.const_expr(xn_pre):
                    # xn predates the down kernel (chain input), so it can
                    # stream with the weights; the epilogue's bar_xn wait then
                    # never exposes gmem latency.
                    issue_xn()
                cute.arch.griddepcontrol_wait()
                for kt in cutlass.range_constexpr(KT):
                    issue_a(kt)
            elif cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
                for kt in cutlass.range_constexpr(KT):
                    issue_w(kt)
                for kt in cutlass.range_constexpr(KT):
                    issue_a(kt)
            else:
                for kt in cutlass.range_constexpr(KT):
                    issue_w(kt)
                    issue_a(kt)
            if cutlass.const_expr(not self.gemm_only and not xn_pre):
                # xn gate values: produced before the down kernel, but the
                # PDL wait is what transitively orders us after xn's producer
                # when the down kernel fires launch_dependents at CTA start;
                # default placement stays after the wait. 32KB/CTA (at
                # MMA_N=128), lands during the mainloop; the epilogue waits on
                # bar_xn.
                issue_xn()
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()
        elif warp_idx == 1:
            if cutlass.const_expr(self.probe_phase == "stream"):
                # Stream probe: drain the TMA barriers, no MMA at all.
                for kt in cutlass.range_constexpr(KT):
                    cute.arch.mbarrier_wait(bar_full + kt, 0)
            elif cutlass.const_expr(self.probe_phase == "commit_only"):
                # Commit-latency probe: alloc + commit an EMPTY mma group,
                # no MMAs. commit_only - stream ~= commit/signal latency.
                _tcgen05.alloc(taddr)
                _tcgen05.commit(bar_mma_done)
            elif cutlass.const_expr(self.probe_phase == "mma_nowait"):
                # Issue all MMAs on stage-0 smem with no stage waits
                # (garbage math, timing only): measures MMA issue+drain
                # fully overlapped with the TMA stream.
                _tcgen05.alloc(taddr)
                idesc = _tcgen05.make_bf16_idesc(128, MMA_N)
                sdesc = _tcgen05.make_sdesc_128B_swizzle(0)
                a_desc = sdesc | (sW[None, None, None, None, 0].iterator.toint() >> 4)
                b_desc = sdesc | (sA[None, None, 0].iterator.toint() >> 4)
                for kt in cutlass.range_constexpr(KT):
                    for k_block in cutlass.range_constexpr(self.mma_k_blocks):
                        _tcgen05.mma_f16(
                            0, a_desc, b_desc, idesc, kt > 0 or k_block > 0
                        )
                        a_desc += 32 >> 4
                        b_desc += 32 >> 4
                _tcgen05.commit(bar_mma_done)
            else:
                # Full 512-column tmem alloc; the base is always 0 (in-tree /
                # gn-kernels idiom), so the accumulator lives at tmem col 0.
                _tcgen05.alloc(taddr)

                idesc = _tcgen05.make_bf16_idesc(128, MMA_N)
                sdesc = _tcgen05.make_sdesc_128B_swizzle(0)

                if cutlass.const_expr(self.probe_phase == "serial_mma"):
                    # Wait for ALL stages first, then issue all MMAs:
                    # serial_mma - stream ~= pure MMA throughput + commit.
                    for kt in cutlass.range_constexpr(KT):
                        cute.arch.mbarrier_wait(bar_full + kt, 0)
                    for kt in cutlass.range_constexpr(KT):
                        a_desc = sdesc | (
                            sW[None, None, None, None, kt].iterator.toint() >> 4
                        )
                        b_desc = sdesc | (sA[None, None, kt].iterator.toint() >> 4)
                        _tcgen05.fence_after_thread_sync()
                        for k_block in cutlass.range_constexpr(self.mma_k_blocks):
                            _tcgen05.mma_f16(
                                0, a_desc, b_desc, idesc, kt > 0 or k_block > 0
                            )
                            a_desc += 32 >> 4
                            b_desc += 32 >> 4
                elif cutlass.const_expr(
                    self.probe_phase in ("late_mma_1", "late_mma_4")
                ):
                    # Drain-latency probe: wait for the whole stream, then
                    # issue ONE k-block (late_mma_1) or one full stage
                    # (late_mma_4) and commit. time - stream ~= tcgen05 pipe
                    # depth + commit signal for a minimal batch.
                    for kt in cutlass.range_constexpr(KT):
                        cute.arch.mbarrier_wait(bar_full + kt, 0)
                    a_desc = sdesc | (
                        sW[None, None, None, None, KT - 1].iterator.toint() >> 4
                    )
                    b_desc = sdesc | (sA[None, None, KT - 1].iterator.toint() >> 4)
                    _tcgen05.fence_after_thread_sync()
                    n_mma: cutlass.Constexpr = (
                        1 if self.probe_phase == "late_mma_1" else self.mma_k_blocks
                    )
                    for k_block in cutlass.range_constexpr(n_mma):
                        _tcgen05.mma_f16(0, a_desc, b_desc, idesc, False)
                        a_desc += 32 >> 4
                        b_desc += 32 >> 4
                else:
                    for kt in cutlass.range_constexpr(KT):
                        if cutlass.const_expr(self.spin_wait):
                            _spin_wait(bar_full + kt)
                        else:
                            cute.arch.mbarrier_wait(bar_full + kt, 0)
                        a_desc = sdesc | (
                            sW[None, None, None, None, kt].iterator.toint() >> 4
                        )
                        b_desc = sdesc | (sA[None, None, kt].iterator.toint() >> 4)
                        # Probe: "no_fence" drops the per-stage tcgen05 fence
                        # (TMA->MMA ordering is via the mbarrier, async proxy
                        # to async proxy; the fence is meant for generic-proxy
                        # smem writes, which this kernel has none of).
                        if cutlass.const_expr(self.probe_phase != "no_fence"):
                            _tcgen05.fence_after_thread_sync()
                        for k_block in cutlass.range_constexpr(self.mma_k_blocks):
                            _tcgen05.mma_f16(
                                0, a_desc, b_desc, idesc, kt > 0 or k_block > 0
                            )
                            a_desc += 32 >> 4
                            b_desc += 32 >> 4
                _tcgen05.commit(bar_mma_done)

        # Per-lane accumulator mapping: lane-quarter wq (0..3) maps to tmem
        # lanes wq*32..+31 holding row = stream (L%32)//8, channel
        # bid*32 + (L//32)*8 + L%8. With EPI warpgroups, warpgroup th
        # covers token slice [th*TN, th*TN+TN) of this CTA's token tile.
        EPI: cutlass.Constexpr = self.epi_split
        TN: cutlass.Constexpr = MMA_N // EPI
        ww = tid // 32
        lane = tid % 32
        wq = ww % 4
        th = ww // 4
        lg = wq * 32 + lane
        s = (lg % 32) // 8
        ch = bid * 32 + (lg // 32) * 8 + (lg % 8)
        row = s * HC_DIM + ch
        t0 = bidm * MMA_N + th * TN

        # Fused: xn landed during the mainloop — preload this thread's TN
        # gate values into registers BEFORE waiting on the MMA drain, hiding
        # the smem read latency under the commit/drain.
        if cutlass.const_expr(self.probe_phase == "full" and not self.gemm_only):
            cute.arch.mbarrier_wait(bar_xn, 0)
            rX = cute.make_rmem_tensor(TN, Float32)
            for tt in cutlass.range_constexpr(TN):
                rX[tt] = sX[lane % 8, s, wq, t0 - bidm * MMA_N + tt].to(Float32)
        # MMA-drain wait: ALL warps wait on the commit barrier directly —
        # one less wake + bar.sync hop than warp-1-waits-then-barrier. (The
        # stream probe runs no MMA, so there is nothing to wait on.)
        if cutlass.const_expr(self.probe_phase == "stream"):
            cute.arch.barrier()
        elif cutlass.const_expr(self.spin_wait):
            _spin_wait(bar_mma_done)
        else:
            cute.arch.mbarrier_wait(bar_mma_done, 0)

        # tcgen05.ld vector length must be a power of 2 dividing TN: use the
        # lowbit (64 -> 64, 48 -> 16, 96 -> 32, 192 -> 64), capped at 64.
        CH: cutlass.Constexpr = min(64, TN & -TN)
        rG = cute.make_rmem_tensor(CH, Float32)
        if cutlass.const_expr(self.probe_phase != "full"):
            # probe only: skip the epilogue ("no_epi") or MMA+epilogue
            # ("stream"); the kernel exits after the barrier above.
            pass
        elif cutlass.const_expr(self.gemm_only):
            # Plain GEMM epilogue: stage the raw accs (128 rows x MMA_N tok)
            # in smem, one (32 ch x MMA_N tok) TMA store box per stream
            # quarter. Lane L owns (s = lane//8, ch-in-oct = lane%8); TMA
            # clips tokens >= M. (Replaces the old scalar strided stores,
            # which made the gemm_only probe unfair vs nvjet.)
            for c in cutlass.range_constexpr(0, TN, CH):
                rG.store(_tcgen05.ld(wq * 32, th * TN + c, "32x32b", CH))
                _tcgen05.wait_ld()
                cute.arch.fence_view_async_tmem_load()
                for tt in cutlass.range_constexpr(CH):
                    tok = t0 + c + tt
                    if tok < M:
                        sO[s, wq * 8 + (lane % 8), th * TN + c + tt] = rG[tt].to(
                            BFloat16
                        )
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()
            if warp_idx == 0:
                gO_tiles = cute.local_tile(O_tma.tma_tensor, (32, MMA_N), (None, None))
                for s_q in cutlass.range_constexpr(HC):
                    simple_tma_copy(
                        O_tma.atom,
                        sO[s_q, None, None],
                        gO_tiles[None, None, s_q * (HC_DIM // 32) + bid, bidm],
                    )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
        else:
            # Fused sigmoid gate-mix. Production rounding boundary: fp32 gate
            # -> bf16 -> fp32 -> sigmoid; s-ordered serial mix; scale 1/HC.
            # Lane L holds stream s=L//8; the butterfly shuffles fetch the
            # other streams' terms for the same channel, so stream-0 lanes
            # accumulate ((s0+s1)+s2)+s3 exactly and store. xn was preloaded
            # into rX before the MMA-drain wait (see above).
            for c in cutlass.range_constexpr(0, TN, CH):
                rG.store(_tcgen05.ld(wq * 32, th * TN + c, "32x32b", CH))
                _tcgen05.wait_ld()
                cute.arch.fence_view_async_tmem_load()
                for tt in cutlass.range_constexpr(CH):
                    g = rG[tt].to(BFloat16).to(Float32)
                    if cutlass.const_expr(self.sigmoid_mode == "tanh"):
                        sig = _sigmoid_f32_tanh(g)
                    else:
                        sig = _sigmoid_f32(g)
                    term = sig * rX[c + tt]
                    otr8 = cute.arch.shuffle_sync_bfly(term, 8)
                    otr16 = cute.arch.shuffle_sync_bfly(term, 16)
                    otr24 = cute.arch.shuffle_sync_bfly(term, 24)
                    if s == 0:
                        acc = ((term + otr8) + otr16) + otr24
                        sO[wq * 8 + (lane % 8), th * TN + c + tt] = (
                            acc * (1.0 / HC)
                        ).to(BFloat16)
            # Stage -> one TMA store box (32 ch x MMA_N tok) per CTA.
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()
            if warp_idx == 0:
                gO_tiles = cute.local_tile(O_tma.tma_tensor, (32, MMA_N), (None, None))
                simple_tma_copy(O_tma.atom, sO, gO_tiles[None, None, bid, bidm])
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)

        cute.arch.barrier()
        if cutlass.const_expr(self.probe_phase != "stream"):
            if warp_idx == 1:
                _tcgen05.dealloc()


class UpGemmV7:
    """Torch wrapper: mma_n bucket dispatch, one compile per (bucket, pdl).

    gemm_only=True: out = a @ w^T, [M, N] (probe scaffold; xn unused).
    gemm_only=False: fused gate-mix; out[m, ch] =
        (1/HC) * sum_s sigmoid(bf16((a @ w_s^T)[m, ch])) * xn_s[m, ch],
    xn is [M, N] bf16, out is [M, N//HC] bf16.
    """

    def __init__(
        self,
        *,
        k: int = 320,
        n: int = 10240,
        hc: int = 4,
        use_pdl: bool = False,
        mma_n_override: int | None = None,
        sigmoid_mode: str = "tanh",
        gemm_only: bool = True,
        weight_evict_first: bool = False,
        pdl_wait_first: bool = False,
        xn_pre_wait: bool = False,
        spin_wait: bool = False,
        probe_phase: str = "full",
        epi_split_override: int | None = None,
    ):
        self._k = k
        self._n = n
        self._hc = hc
        self._use_pdl = use_pdl
        self._mma_n_override = mma_n_override
        self._sigmoid_mode = sigmoid_mode
        self._gemm_only = gemm_only
        self._weight_evict_first = weight_evict_first
        self._pdl_wait_first = pdl_wait_first
        self._xn_pre_wait = xn_pre_wait
        self._spin_wait = spin_wait
        self._probe_phase = probe_phase
        self._epi_split_override = epi_split_override
        self._compiled: dict[int, object] = {}

    def _compile(self, mma_n: int):
        import cutlass.cute as cute
        from cutlass import BFloat16
        from quack.compile_utils import make_fake_tensor

        kernel = UpGemmV7Kernel(
            k=self._k,
            n=self._n,
            hc=self._hc,
            mma_n=mma_n,
            use_pdl=self._use_pdl,
            sigmoid_mode=self._sigmoid_mode,
            gemm_only=self._gemm_only,
            weight_evict_first=self._weight_evict_first,
            pdl_wait_first=self._pdl_wait_first,
            xn_pre_wait=self._xn_pre_wait,
            spin_wait=self._spin_wait,
            probe_phase=self._probe_phase,
            epi_split_override=self._epi_split_override,
        )
        m = cute.sym_int()
        w = make_fake_tensor(BFloat16, (self._n, self._k), divisibility=8)
        a = make_fake_tensor(BFloat16, (m, self._k), divisibility=8)
        xn = make_fake_tensor(BFloat16, (m, self._n), divisibility=8)
        out_cols = self._n if self._gemm_only else self._n // self._hc
        out = make_fake_tensor(BFloat16, (m, out_cols), divisibility=8)
        return cute.compile(
            kernel,
            w,
            a,
            xn,
            out,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    def __call__(
        self,
        a: torch.Tensor,
        w: torch.Tensor,
        xn: torch.Tensor | None = None,
    ) -> torch.Tensor:
        M = a.shape[0]
        if M > 1024:
            raise ValueError("up gemm v7 requires M <= 1024")
        if self._mma_n_override is not None:
            mma_n = self._mma_n_override
        else:
            # Widest bucket that covers M in ONE token tile: grid.y > 1 puts
            # >148 CTAs on the machine (grid.x=80, 1 CTA/SM) = 2-wave cliff.
            # Fused mode caps at 128: the xn smem tile (128*MMA_N*2 B) would
            # push smem past the 227KB limit at 192.
            buckets = (
                (8, 16, 32, 64, 96, 128, 192)
                if self._gemm_only
                else (8, 16, 32, 64, 96, 128)
            )
            mma_n = next((b for b in buckets if b >= M), buckets[-1])
        for t, name in ((a, "a"),) + (() if self._gemm_only else ((xn, "xn"),)):
            if t.stride(1) != 1 or t.stride(0) % 8 != 0:
                raise ValueError(
                    f"up v7 requires row-major {name} with a 16B-aligned row stride"
                )
        if not self._gemm_only and (xn is None or xn.shape != (M, self._n)):
            raise ValueError("fused up v7 requires xn of shape [M, N]")
        if mma_n not in self._compiled:
            self._compiled[mma_n] = self._compile(mma_n)
        out_cols = self._n if self._gemm_only else self._n // self._hc
        out = torch.empty(M, out_cols, dtype=torch.bfloat16, device=a.device)
        # gemm_only: xn is dead code in the kernel; pass out as a dummy
        # (compiled fake xn matches out's [M, N] shape in this mode).
        self._compiled[mma_n](w, a, out if self._gemm_only else xn, out)
        return out

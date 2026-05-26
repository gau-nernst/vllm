# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import statistics
from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, cute
from cutlass.cute.nvgpu import cpasync
from flashinfer.testing import bench_gpu_time_with_cupti
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import _tcgen05


class KDAChunkRecurrentKernel:
    """Skeleton for the KDA recurrent/output stage.

    For each sequence/head, iterate over 16-token tiles:
        v_new = u - w @ h.T
        h_new = (h + v_new.T @ k_right) * exp(g_last)
        o = q_decay @ h.T + p @ v_new

    The `h + ...` update follows the Triton implementation in kda_triton.py.
    """

    BT = 16
    chunk_size = 64
    num_warps = 10

    def __init__(
        self,
        head_dim: int,
        value_dim: int,
        num_heads: int,
        num_stages: int = 2,
    ) -> None:
        assert head_dim == 128
        assert value_dim == 128
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.num_stages = num_stages

    @cute.jit
    def _make_tile_tma_args(
        self,
        tensor: cute.Tensor,
        dim: cutlass.Constexpr[int],
        op: cpasync.TmaCopyOp,
        stages: cutlass.Constexpr[int],
    ):
        # Logical tile: [BT, dim]. Physical row storage is already padded by
        # the pre-recurrent stage; the caller chooses the tile coordinate.
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        slayout = cute.make_layout(
            (self.BT, 1, (64, dim // 64), stages),
            stride=(64, 0, (1, self.BT * 64), self.BT * dim),
        )
        slayout = cute.make_composed_layout(swizzle_128B, 0, slayout)
        atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            cute.logical_divide(tensor, (None, None, 64)),
            slayout,
            cta_tiler=(self.BT, 1, dim),
        )
        return atom, tma_tensor, slayout

    @cute.jit
    def _make_p_tma_args(
        self,
        tensor: cute.Tensor,
        op: cpasync.TmaCopyOp,
        stages: cutlass.Constexpr[int],
    ):
        # P is stored by original token position: [total_tokens, H, BT].
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        slayout = cute.make_layout(
            (self.BT, 1, (8, self.BT // 8), stages),
            stride=(self.BT, 0, (1, self.BT * 8), self.BT * self.BT),
        )
        slayout = cute.make_composed_layout(swizzle_128B, 0, slayout)
        atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            cute.logical_divide(tensor, (None, None, 8)),
            slayout,
            cta_tiler=(self.BT, 1, self.BT),
        )
        return atom, tma_tensor, slayout

    @cute.jit
    def _make_h_tma_args(
        self,
        tensor: cute.Tensor,
        op: cpasync.TmaCopyOp,
        stages: cutlass.Constexpr[int],
    ):
        # H tile: [value_dim, head_dim].
        num_elems = 128 // (tensor.element_type.width // 8)
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        slayout = cute.make_layout(
            (1, self.value_dim, (num_elems, self.head_dim // num_elems), stages),
            stride=(
                0,
                num_elems,
                (1, self.value_dim * num_elems),
                self.value_dim * self.head_dim,
            ),
        )
        slayout = cute.make_composed_layout(swizzle_128B, 0, slayout)
        atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            cute.logical_divide(tensor, (None, None, num_elems)),
            slayout,
            cta_tiler=(1, self.value_dim, self.head_dim),
        )
        return atom, tma_tensor, slayout

    @cute.jit
    def _make_g_last_tma_args(self, tensor: cute.Tensor, op: cpasync.TmaCopyOp):
        # g_last is per 16-token tile: [num_tiles, H, head_dim].
        return self._make_tile_tma_args(tensor, self.head_dim, op, 1)

    @cute.jit
    def __call__(
        self,
        U: cute.Tensor,
        W: cute.Tensor,
        K_right: cute.Tensor,
        Q_decay: cute.Tensor,
        P: cute.Tensor,
        g_last: cute.Tensor,
        H0: cute.Tensor,
        Out: cute.Tensor,
        H_final: cute.Tensor,
        cu_seqlens: cute.Tensor,
        stream: CUstream,
    ):
        tma_g2s = cpasync.CopyBulkTensorTileG2SOp()
        tma_s2g = cpasync.CopyBulkTensorTileS2GOp()

        U_args = self._make_tile_tma_args(U, self.value_dim, tma_g2s, self.num_stages)
        W_args = self._make_tile_tma_args(W, self.head_dim, tma_g2s, self.num_stages)
        K_right_args = self._make_tile_tma_args(
            K_right, self.head_dim, tma_g2s, self.num_stages
        )
        Q_decay_args = self._make_tile_tma_args(
            Q_decay, self.head_dim, tma_g2s, self.num_stages
        )
        P_args = self._make_p_tma_args(P, tma_g2s, self.num_stages)
        G_last_args = self._make_g_last_tma_args(g_last, tma_g2s)
        H0_args = self._make_h_tma_args(H0, tma_g2s, 1)
        Out_args = self._make_tile_tma_args(Out, self.value_dim, tma_s2g, 1)
        H_final_args = self._make_h_tma_args(H_final, tma_s2g, 1)

        grid = (self.num_heads, H0.shape[0], 1)
        block = (self.num_warps * 32, 1, 1)
        self.kernel(
            U_args,
            W_args,
            K_right_args,
            Q_decay_args,
            P_args,
            G_last_args,
            H0_args,
            Out_args,
            H_final_args,
            cu_seqlens,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        U_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        W_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        K_right_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        Q_decay_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        P_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        G_last_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        H0_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        O_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        H_final_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        cu_seqlens: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        # head_id, seq_id, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        # lane_id = tid % 32

        # BT = self.BT
        # head_dim = self.head_dim
        # value_dim = self.value_dim
        # num_stages = self.num_stages

        # # Descriptor unpacking is intentionally kept here so the eventual
        # # implementation has the right TMA surface and shared-memory layouts.
        # U_tma_atom, tmaU, sU_layout = U_args
        # W_tma_atom, tmaW, sW_layout = W_args
        # K_right_tma_atom, tmaK_right, sK_right_layout = K_right_args
        # Q_decay_tma_atom, tmaQ_decay, sQ_decay_layout = Q_decay_args
        # P_tma_atom, tmaP, sP_layout = P_args
        # G_last_tma_atom, tmaG_last, sG_last_layout = G_last_args
        # H0_tma_atom, tmaH0, sH0_layout = H0_args
        # O_tma_atom, tmaO, sO_layout = O_args
        # H_final_tma_atom, tmaH_final, sH_final_layout = H_final_args

        smem = cutlass.utils.SmemAllocator()
        taddr = smem.allocate_array(Int32, 1)

        if warp_id == 9:
            # TMA warp
            pass

        elif warp_id == 8:
            # MMA warp
            _tcgen05.alloc(taddr)
            _tcgen05.dealloc()

        elif warp_id >= 4:
            # xx warps
            pass
            # warp_id_ = warp_id % 4
            # tid_ = tid % 128

        else:
            # xx warps
            pass

    @cache
    @staticmethod
    def compile(
        head_dim: int,
        value_dim: int,
        num_heads: int,
        num_stages: int = 2,
    ):
        total_t = cute.sym_int()
        pad_t = cute.sym_int()
        num_tiles = cute.sym_int()
        num_seqs = cute.sym_int()
        cu_entries = cute.sym_int()

        U = make_fake_tensor(BFloat16, (pad_t, num_heads, value_dim), divisibility=16)
        W = make_fake_tensor(BFloat16, (pad_t, num_heads, head_dim), divisibility=16)
        K_right = make_fake_tensor(
            BFloat16, (pad_t, num_heads, head_dim), divisibility=16
        )
        Q_decay = make_fake_tensor(
            BFloat16, (pad_t, num_heads, head_dim), divisibility=16
        )
        P = make_fake_tensor(
            BFloat16, (total_t, num_heads, KDAChunkRecurrentKernel.BT), divisibility=16
        )
        g_last = make_fake_tensor(
            Float32, (num_tiles, num_heads, head_dim), divisibility=16
        )
        H0 = make_fake_tensor(
            Float32, (num_seqs, num_heads, value_dim, head_dim), divisibility=16
        )
        Out = make_fake_tensor(
            BFloat16, (total_t, num_heads, value_dim), divisibility=16
        )
        H_final = make_fake_tensor(
            Float32, (num_seqs, num_heads, value_dim, head_dim), divisibility=16
        )
        cu_seqlens = make_fake_tensor(Int32, (cu_entries,), divisibility=1)

        kernel = KDAChunkRecurrentKernel(head_dim, value_dim, num_heads, num_stages)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            U,
            W,
            K_right,
            Q_decay,
            P,
            g_last,
            H0,
            Out,
            H_final,
            cu_seqlens,
            stream,
            options="--enable-tvm-ffi",
        )


def _padded_tokens(seqlens: list[int], chunk_size: int) -> int:
    return sum(
        (length + chunk_size - 1) // chunk_size * chunk_size for length in seqlens
    )


def _num_tiles(seqlens: list[int], tile_size: int) -> int:
    return sum((length + tile_size - 1) // tile_size for length in seqlens)


def kda_recurrent_cutedsl(
    U: torch.Tensor,
    W: torch.Tensor,
    K_right: torch.Tensor,
    Q_decay: torch.Tensor,
    P: torch.Tensor,
    g_last: torch.Tensor,
    H0: torch.Tensor,
    Out: torch.Tensor,
    H_final: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_stages: int = 2,
) -> None:
    _, num_heads, value_dim = U.shape
    head_dim = W.shape[-1]
    KDAChunkRecurrentKernel.compile(head_dim, value_dim, num_heads, num_stages)(
        U,
        W,
        K_right,
        Q_decay,
        P,
        g_last,
        H0,
        Out,
        H_final,
        cu_seqlens,
    )


def recurrent_pytorch_reference(
    U: torch.Tensor,
    W: torch.Tensor,
    K_right: torch.Tensor,
    Q_decay: torch.Tensor,
    P: torch.Tensor,
    g_last: torch.Tensor,
    H0: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    BT = KDAChunkRecurrentKernel.BT
    chunk_size = KDAChunkRecurrentKernel.chunk_size
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()
    total_tokens = int(cu_seqlens[-1].item())

    O_ref = torch.zeros(
        total_tokens, U.shape[1], U.shape[2], dtype=torch.bfloat16, device=U.device
    )
    H_final_ref = torch.empty_like(H0)

    padded_start = 0
    tile_id = 0
    for seq_id, seq_len_t in enumerate(lengths):
        seq_len = int(seq_len_t)
        bos = int(cu_seqlens[seq_id].item())
        h = H0[seq_id].float().clone()
        for tile_start in range(0, seq_len, BT):
            tile_len = min(BT, seq_len - tile_start)
            token_start = bos + tile_start
            token_slice = slice(token_start, token_start + tile_len)
            padded_slice = slice(
                padded_start + tile_start, padded_start + tile_start + tile_len
            )

            u = U[padded_slice].float()
            w = W[padded_slice].float()
            k_right = K_right[padded_slice].float()
            q_decay = Q_decay[padded_slice].float()
            p = P[token_slice, :, :tile_len].float()
            g = g_last[tile_id].float()

            for head_id in range(U.shape[1]):
                h_old = h[head_id]
                v_new = u[:, head_id] - w[:, head_id] @ h_old.T

                qh = q_decay[:, head_id] @ h_old.T
                pv = p[:, head_id] @ v_new
                O_ref[token_slice, head_id] = (qh + pv).to(torch.bfloat16)

                h[head_id] = (h_old + v_new.T @ k_right[:, head_id]) * torch.exp(
                    g[head_id]
                ).view(1, -1)

            tile_id += 1
        H_final_ref[seq_id] = h
        padded_start += (seq_len + chunk_size - 1) // chunk_size * chunk_size

    return O_ref, H_final_ref


def _report_tensor(
    name: str,
    actual: torch.Tensor,
    ref: torch.Tensor,
    mask: torch.Tensor,
    atol: float = 7e-2,
    rtol: float = 7e-2,
) -> bool:
    actual_f = actual.float()[mask]
    ref_f = ref.float()[mask]
    print(f"{name}:")
    if actual_f.numel() == 0:
        print("  status: SKIP")
        print("  reason: no valid elements checked")
        return True

    abs_diff = (actual_f - ref_f).abs()
    allowed = atol + rtol * ref_f.abs()
    bad = abs_diff > allowed
    max_abs, flat_idx = abs_diff.max(dim=0)
    ref_abs = ref_f.abs()
    rel_diff = abs_diff / torch.where(ref_abs > 0, ref_abs, torch.ones_like(ref_abs))
    max_rel = rel_diff.max()
    ok = not bool(bad.any())
    bad_count = int(bad.sum().item())
    total = actual_f.numel()
    print(f"  status: {'PASS' if ok else 'FAIL'}")
    print(f"  bad: {bad_count}/{total} ({100.0 * bad_count / total:.2f}%)")
    print(
        f"  max_abs: {float(max_abs.item()):.6g}, max_rel: {float(max_rel.item()):.6g}"
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


def check_recurrent_matches_reference(
    U: torch.Tensor,
    W: torch.Tensor,
    K_right: torch.Tensor,
    Q_decay: torch.Tensor,
    P: torch.Tensor,
    g_last: torch.Tensor,
    H0: torch.Tensor,
    Out: torch.Tensor,
    H_final: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> bool:
    Out_ref, Ht_ref = recurrent_pytorch_reference(
        U, W, K_right, Q_decay, P, g_last, H0, cu_seqlens
    )
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()
    valid_rows = torch.zeros(U.shape[0], dtype=torch.bool, device=U.device)
    o_mask = torch.zeros_like(Out, dtype=torch.bool)
    padded_start = 0
    for seq_id, seq_len_t in enumerate(lengths):
        seq_len = int(seq_len_t)
        bos = int(cu_seqlens[seq_id].item())
        valid_rows[padded_start : padded_start + seq_len] = True
        o_mask[bos : bos + seq_len] = True
        padded_start += (
            (seq_len + KDAChunkRecurrentKernel.chunk_size - 1)
            // KDAChunkRecurrentKernel.chunk_size
            * KDAChunkRecurrentKernel.chunk_size
        )

    print("recurrent correctness")
    ok = True
    ok &= _report_tensor("Out", Out, Out_ref, o_mask)
    ok &= _report_tensor(
        "H_final", H_final, Ht_ref, torch.ones_like(H_final, dtype=torch.bool)
    )
    return ok


def benchmark_recurrent(
    U: torch.Tensor,
    W: torch.Tensor,
    K_right: torch.Tensor,
    Q_decay: torch.Tensor,
    P: torch.Tensor,
    g_last: torch.Tensor,
    H0: torch.Tensor,
    Out: torch.Tensor,
    H_final: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> float:
    def launch() -> None:
        kda_recurrent_cutedsl(
            U,
            W,
            K_right,
            Q_decay,
            P,
            g_last,
            H0,
            Out,
            H_final,
            cu_seqlens,
        )

    timings = bench_gpu_time_with_cupti(launch)
    return statistics.median(timings)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile and launch KDA recurrent skeleton."
    )
    parser.add_argument("--seqlens", type=int, nargs="+", default=[63, 129, 512, 2045])
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_default_device("cuda")
    torch.manual_seed(args.seed)

    BT = KDAChunkRecurrentKernel.BT
    chunk_size = KDAChunkRecurrentKernel.chunk_size
    head_dim = 128
    value_dim = 128
    offsets = [0]
    for length in args.seqlens:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32)
    total_tokens = offsets[-1]
    pad_t = _padded_tokens(args.seqlens, chunk_size)
    num_tiles = _num_tiles(args.seqlens, BT)

    U = torch.randn(pad_t, args.heads, value_dim, dtype=torch.bfloat16)
    W = torch.randn(pad_t, args.heads, head_dim, dtype=torch.bfloat16) * 0.05
    K_right = torch.randn(pad_t, args.heads, head_dim, dtype=torch.bfloat16) * 0.05
    Q_decay = torch.randn(pad_t, args.heads, head_dim, dtype=torch.bfloat16) * 0.05
    P = torch.randn(total_tokens, args.heads, BT, dtype=torch.bfloat16) * 0.05
    g_last = torch.randn(num_tiles, args.heads, head_dim, dtype=torch.float32) * 0.01
    H0 = (
        torch.randn(
            len(args.seqlens), args.heads, value_dim, head_dim, dtype=torch.float32
        )
        * 0.05
    )

    Out = torch.zeros(total_tokens, args.heads, value_dim, dtype=torch.bfloat16)
    H_final = torch.zeros_like(H0)

    kda_recurrent_cutedsl(
        U, W, K_right, Q_decay, P, g_last, H0, Out, H_final, cu_seqlens
    )
    torch.accelerator.synchronize()

    check_passed = check_recurrent_matches_reference(
        U,
        W,
        K_right,
        Q_decay,
        P,
        g_last,
        H0,
        Out,
        H_final,
        cu_seqlens,
    )

    median_ms = benchmark_recurrent(
        U,
        W,
        K_right,
        Q_decay,
        P,
        g_last,
        H0,
        Out,
        H_final,
        cu_seqlens,
    )
    print("recurrent benchmark")
    print(f"  mean: {median_ms * 1e3:.3f} us")
    print(f"  tokens/s: {total_tokens / (median_ms * 1e-3):.6g}")
    print(f"  tiles/s: {num_tiles / (median_ms * 1e-3):.6g}")
    print(
        "launched kernel_recurrent",
        f"total_tokens={total_tokens}",
        f"heads={args.heads}",
        f"tiles={num_tiles}",
        f"check={'passed' if check_passed else 'failed'}",
    )


if __name__ == "__main__":
    main()

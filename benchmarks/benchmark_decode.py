# -*- coding: utf-8 -*-

import os
import math

import torch
import triton
from flash_attn import flash_attn_func

from fla.ops.retention import chunk_retention, parallel_retention
from fla.ops.delta_rule import chunk_delta_rule, fused_chunk_delta_rule, fused_recurrent_delta_rule


def get_mla_flop(b, s_q, h_q, d_qk, d_v, topk) -> float:
    flop = 2 * sum([
        h_q * d_qk * topk,
        h_q * d_v * topk
    ]) * b * s_q
    return flop


def get_delta_rule_flop(B, T, H, D) -> float:
    chunk_size = 64
    flops = 2 * (T * H * D * (6 * D + 8 * chunk_size)) * B
    return flops

def run_fused_recurrent_delta_rule(q, k, v, beta, initial_state):
    return fused_recurrent_delta_rule(q, k, v, beta, initial_state=initial_state)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[16 * 2 ** i for i in range(0, 9)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        # line_vals=['recurrent-delta', 'flash-attn', 'flash-mla'],
        line_vals=['recurrent-delta', 'flash-attn'],
        # label name for the lines
        # line_names=['recurrent-delta', 'flash-attn', 'flash-mla'],
        line_names=['recurrent-delta', 'flash-attn'],
        # line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-'), ('cyan', '-')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    )
)
def benchmark(T, provider):
    from fla.utils import device
    dtype = torch.bfloat16
    requires_grad = True
    B, H, D = 64, 128, 128
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # print(T)
    if provider == 'recurrent-delta':
        q = torch.randn(B, 1, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        k = torch.randn(B, 1, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        v = torch.randn(B, 1, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        initial_state = torch.randn(B, H, D, D, device=device, requires_grad=requires_grad, dtype=dtype)
        beta = torch.rand(B, 1, H, device=device, dtype=dtype).sigmoid()
    elif provider == 'flash-attn':
        q = torch.randn(B, 1, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        k = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider == 'recurrent-delta':
        results = triton.testing.do_bench(lambda: fused_recurrent_delta_rule(q, k, v, beta, initial_state=initial_state), quantiles=quantiles)
    elif provider == 'flash-attn':
        results = triton.testing.do_bench(lambda: flash_attn_func(q, k, v, causal=False), quantiles=quantiles)
    # elif provider == 'flash-mla':
    #     flop = get_mla_flop(B, T, H, d_qk, d_v, top_k)
    #     results = triton.testing.do_bench(lambda: flash_mla_sparse_fwd(
    #         q.squeeze(0), kv.squeeze(0), indices.squeeze(0), sm_scale), quantiles=quantiles)
    #     prefill_ans_time = results[0] / 1000
    #     prefill_flops = flop / prefill_ans_time / 1e12
    #     print("================")
    #     print(
    #         f"FlashMLA: Running on TestParam(b={B}, s_q={T}, s_kv={T}, topk={top_k}, h_q={H}, h_kv={h_kv}, d_qk={d_qk}, d_v={d_v}, dtype={dtype})")
    #     print(f"Prefill:  {prefill_ans_time * 1e6:4.0f} us, {prefill_flops:.3f} TFlops")

    return results


if __name__ == '__main__':
    benchmark.run(print_data=True, save_path='.')

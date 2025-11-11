# -*- coding: utf-8 -*-

import os
import math

import torch
import triton
from flash_attn import flash_attn_func

from fla.ops.retention import chunk_retention, parallel_retention
from fla.ops.delta_rule import chunk_delta_rule, fused_chunk_delta_rule, fused_recurrent_delta_rule

def get_flash_attn_flop(batch, seqlen, nheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def get_flash_attn_memory(B, T, H, D, dtype_bytes=2) -> float:
    # 读取 Q, K, V 和写入 O
    memory_bytes = 4 * B * T * H * D * dtype_bytes
    return memory_bytes


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




@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[256 * 2 ** i for i in range(0, 7)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        # line_vals=['chunk-delta', 'recurrent-delta', 'flash-attn'],
        line_vals=['chunk-delta', 'fused-chunk-delta', 'recurrent-delta', 'flash-attn'],
        # label name for the lines
        line_names=['chunk-delta', 'fused-chunk-delta', 'recurrent-delta', 'flash-attn'],
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
    B, H, D = 1, 128, 128
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # print(T)

    if provider == 'flash-mla':
        # flash mla
        top_k = T // 2
        h_kv = 1
        d_qk = 576
        d_v = 512
        q = torch.randn((B, T, H, d_qk), device=device, requires_grad=requires_grad, dtype=dtype) / 10
        kv = torch.randn((B, T, h_kv, d_qk), device=device, requires_grad=requires_grad, dtype=dtype) / 10
        q.clamp_(-10, 10)
        kv.clamp_(-10, 10)

        indices = torch.full((B, T, h_kv, top_k), T, dtype=torch.int32)
        for b in range(B):
            for s in range(T):
                for h in range(h_kv):
                    # NOTE We use the following method to generate indices so that most indices lies within [s_kv-20000, s_kv), which is more realistic for sparse attention
                    near_mask = torch.randint(0, 32, (min(top_k, T),)) < 31
                    cur_indices = torch.randperm(T)[:top_k]
                    cur_indices[near_mask] = torch.randint(max(0, T - 20000), T - 1, (near_mask.sum().item(),))
                    if len(cur_indices) < top_k:
                        cur_indices = torch.cat([cur_indices, torch.full((top_k - len(cur_indices),), 2147480000)])
                    cur_indices = cur_indices[torch.randperm(top_k)]
                    indices[b, s, h] = cur_indices
        indices = indices.to(q.device)
        sm_scale = 1 / math.sqrt(D)
    else:
        # linear attention & flash attention
        q = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        k = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)

    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider == 'chunk-delta':
        flop = get_delta_rule_flop(B, T, H, D)
        results = triton.testing.do_bench(lambda: chunk_delta_rule(q, k, v, beta), quantiles=quantiles)
        prefill_ans_time = results[0] / 1000
        prefill_flops = flop / prefill_ans_time / 1e12
        memory_bytes = T * H * D * (26 + 4 * D / 64 + 6 * 64 / D)
        memory_gb = memory_bytes / (1024 * 1024 * 1024)
        prefill_bandwidth = memory_gb / prefill_ans_time
        print("================")
        print(
            f"Chunked Delta Rule: Running on TestParam(b={B}, s_q={T}, s_kv={T},h_q={H}dtype={dtype})")
        print(f"Prefill:  {prefill_ans_time * 1e6:4.0f} us, {prefill_flops:.3f} TFlops")
        print(f"Prefill Bandwidth: {prefill_bandwidth:.3f} GB/s")

    if provider == 'fused-chunk-delta':
        flop = get_delta_rule_flop(B, T, H, D)
        results = triton.testing.do_bench(lambda: fused_chunk_delta_rule(q, k, v, beta), quantiles=quantiles)
        prefill_ans_time = results[0] / 1000
        prefill_flops = flop / prefill_ans_time / 1e12
        memory_bytes = T * H * D * (26 + 4 * D / 64 + 6 * 64 / D)
        memory_gb = memory_bytes / (1024 * 1024 * 1024)
        prefill_bandwidth = memory_gb / prefill_ans_time
        print("================")
        print(
            f"Fused Chunked Delta Rule: Running on TestParam(b={B}, s_q={T}, s_kv={T},h_q={H}dtype={dtype})")
        print(f"Prefill:  {prefill_ans_time * 1e6:4.0f} us, {prefill_flops:.3f} TFlops")
        print(f"Prefill Bandwidth: {prefill_bandwidth:.3f} GB/s")
        
    elif provider == 'recurrent-delta':
        results = triton.testing.do_bench(lambda: fused_recurrent_delta_rule(q, k, v, beta), quantiles=quantiles)
    elif provider == 'flash-attn':
        results = triton.testing.do_bench(lambda: flash_attn_func(q, k, v, causal=True), quantiles=quantiles)
        prefill_ans_time = results[0] / 1000
        flop = get_flash_attn_flop(B, T, H, D, causal=True)
        prefill_flops = flop / prefill_ans_time / 1e12
        memory_bytes = get_flash_attn_memory(B, T, H, D)
        memory_gb = memory_bytes / (1024 * 1024 * 1024)
        prefill_bandwidth = memory_gb / prefill_ans_time
        print("================")
        print(
            f"Flash Attention: Running on TestParam(b={B}, s_q={T}, s_kv={T},h_q={H}dtype={dtype})")
        print(f"Prefill:  {prefill_ans_time * 1e6:4.0f} us, {prefill_flops:.3f} TFlops")
        print(f"Prefill Bandwidth: {prefill_bandwidth:.3f} GB/s")

    return results


if __name__ == '__main__':
    benchmark.run(print_data=True, save_path='.')

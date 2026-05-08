Run kernel performance benchmarks to verify codegen changes.

## Kernels to benchmark

- **GEMM**: square GEMM at M=N=K in {1024, 2048, 4096, 8192, 16384} for three variants:
  - fp16: `python -m tirx_kernels.bench --kernel fp16_bf16_gemm`
  - fp8: `python -m tirx_kernels.bench --kernel fp8_blockwise_gemm`
  - nvfp4: `python -m tirx_kernels.bench --kernel nvfp4_gemm`
- **FA4** (flash_attention4): all registered configs
  - `python -m tirx_kernels.bench --kernel flash_attention4`

## Steps

1. Select the least busy GPU:
   ```bash
   export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
   ```

2. Run benchmarks for each kernel using the commands above.

3. Present results in a table: kernel x config, with times in ms.

## When to use

When modifying anything that affects code generation: kernels, op dispatches, lowering passes, codegen, device ops.

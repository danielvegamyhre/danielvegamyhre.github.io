# Benchmark Results - tma-store-overlap128

| Matrix Size (M×K×N) | Custom Kernel | PyTorch _scaled_mm | Speedup |
|---------------------|---------------|--------------------|---------|
| 2048×2048×2048 | 19.424 μs (884.47 TFLOPS) | 17.248 μs (996.05 TFLOPS) | 0.89x |
| 4096×4096×4096 | 62.400 μs (2202.55 TFLOPS) | 54.144 μs (2538.40 TFLOPS) | 0.87x |
| 8192×8192×8192 | 431.136 μs (2550.27 TFLOPS) | 409.504 μs (2684.98 TFLOPS) | 0.95x |
| 16384×16384×16384 | 3549.152 μs (2478.36 TFLOPS) | 3421.056 μs (2571.16 TFLOPS) | 0.96x |

# Benchmark Results - heuristic-for-hilbert

| Matrix Size (M×K×N) | Custom Kernel | PyTorch _scaled_mm | Speedup |
|---------------------|---------------|--------------------|---------|
| 2048×2048×2048 | 17.536 μs (979.69 TFLOPS) | 17.248 μs (996.05 TFLOPS) | 0.98x |
| 4096×4096×4096 | 62.288 μs (2206.51 TFLOPS) | 54.144 μs (2538.40 TFLOPS) | 0.87x |
| 8192×8192×8192 | 435.264 μs (2526.08 TFLOPS) | 408.416 μs (2692.14 TFLOPS) | 0.94x |
| 16384×16384×16384 | 3593.408 μs (2447.84 TFLOPS) | 3436.320 μs (2559.74 TFLOPS) | 0.96x |

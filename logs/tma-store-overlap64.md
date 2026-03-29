# Benchmark Results - tma-store-overlap64

| Matrix Size (M×K×N) | Custom Kernel | PyTorch _scaled_mm | Speedup |
|---------------------|---------------|--------------------|---------|
| 2048×2048×2048 | 17.504 μs (981.48 TFLOPS) | 17.248 μs (996.05 TFLOPS) | 0.99x |
| 4096×4096×4096 | 62.400 μs (2202.55 TFLOPS) | 54.144 μs (2538.40 TFLOPS) | 0.87x |
| 8192×8192×8192 | 431.136 μs (2550.27 TFLOPS) | 408.416 μs (2692.14 TFLOPS) | 0.95x |
| 16384×16384×16384 | 3626.912 μs (2425.23 TFLOPS) | 3439.616 μs (2557.29 TFLOPS) | 0.95x |

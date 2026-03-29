# Benchmark Results - static-index

| Matrix Size (M×K×N) | Custom Kernel | PyTorch _scaled_mm | Speedup |
|---------------------|---------------|--------------------|---------|
| 2048×2048×2048 | 19.456 μs (883.01 TFLOPS) | 17.248 μs (996.05 TFLOPS) | 0.89x |
| 4096×4096×4096 | 62.496 μs (2199.16 TFLOPS) | 54.112 μs (2539.90 TFLOPS) | 0.87x |
| 8192×8192×8192 | 463.872 μs (2370.29 TFLOPS) | 386.976 μs (2841.29 TFLOPS) | 0.83x |
| 16384×16384×16384 | 4031.520 μs (2181.83 TFLOPS) | 3186.592 μs (2760.34 TFLOPS) | 0.79x |

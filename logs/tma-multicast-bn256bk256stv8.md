# Benchmark Results - tma-multicast-bn256bk256stv8

| Matrix Size (M×K×N) | Custom Kernel | PyTorch _scaled_mm | Speedup |
|---------------------|---------------|--------------------|---------|
| 2048×2048×2048 | 21.408 μs (802.50 TFLOPS) | 17.248 μs (996.05 TFLOPS) | 0.81x |
| 4096×4096×4096 | 76.800 μs (1789.57 TFLOPS) | 54.144 μs (2538.40 TFLOPS) | 0.70x |
| 8192×8192×8192 | 515.168 μs (2134.28 TFLOPS) | 385.888 μs (2849.30 TFLOPS) | 0.75x |
| 16384×16384×16384 | 4103.040 μs (2143.80 TFLOPS) | 3308.448 μs (2658.68 TFLOPS) | 0.81x |

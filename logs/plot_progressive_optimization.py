#!/usr/bin/env python3
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(file_path):
    """Parse a log file and extract benchmark data."""
    with open(file_path, 'r') as f:
        content = f.read()

    results = []

    # Try parsing raw log format first
    pattern = r'Matrix size: M=(\d+), K=(\d+), N=(\d+)\s+Custom kernel:\s+([0-9.]+) us \(([0-9.]+) tflops\)\s+PyTorch _scaled_mm:\s+([0-9.]+) us \(([0-9.]+) tflops\)\s+Speedup: ([0-9.]+)x'
    matches = re.findall(pattern, content, re.MULTILINE)

    if matches:
        for match in matches:
            m, k, n = int(match[0]), int(match[1]), int(match[2])
            custom_time = float(match[3])
            custom_tflops = float(match[4])
            pytorch_time = float(match[5])
            pytorch_tflops = float(match[6])
            speedup = float(match[7])

            results.append({
                'M': m, 'K': k, 'N': n,
                'size': m,  # Assuming square matrices M=N=K
                'custom_time_us': custom_time,
                'custom_tflops': custom_tflops,
                'pytorch_time_us': pytorch_time,
                'pytorch_tflops': pytorch_tflops,
                'speedup': speedup
            })
    else:
        # Try parsing markdown table format
        table_pattern = r'\| M=(\d+), K=(\d+), N=(\d+) \| ([0-9.]+) us \(([0-9.]+) tflops\) \| ([0-9.]+) us \(([0-9.]+) tflops\) \| ([0-9.]+)x \|'
        matches = re.findall(table_pattern, content, re.MULTILINE)

        for match in matches:
            m, k, n = int(match[0]), int(match[1]), int(match[2])
            custom_time = float(match[3])
            custom_tflops = float(match[4])
            pytorch_time = float(match[5])
            pytorch_tflops = float(match[6])
            speedup = float(match[7])

            results.append({
                'M': m, 'K': k, 'N': n,
                'size': m,  # Assuming square matrices M=N=K
                'custom_time_us': custom_time,
                'custom_tflops': custom_tflops,
                'pytorch_time_us': pytorch_time,
                'pytorch_tflops': pytorch_tflops,
                'speedup': speedup
            })

    return results

def plot_progressive_optimization():
    """Create progressive plots showing optimization journey."""

    # Define the optimization sequence
    optimization_sequence = [
        ("initial.txt", "Initial Kernel"),
        ("bn256bk128stv4.txt", "Optimization 1"),
        ("bn256bk128stv8.txt", "Optimization 2"),
        ("bn256bk256stv8.txt", "Optimization 3"),
        ("tma-multicast-bn256bk256stv8.txt", "Optimization 4"),
        ("static-index.txt", "Optimization 5"),
        ("hilbert.txt", "Optimization 6"),
        ("l1-no-alloc.txt", "Optimization 7"),
        ("heuristic-for-hilbert.txt", "Optimization 8"),
        ("tma-store-overlap128.txt", "Optimization 9"),
        ("unified.txt", "Optimization 10"),
        ("tma-store-overlap64.txt", "Optimization 11")
    ]

    # Load cuBLAS baseline from unified.txt
    print("Loading cuBLAS baseline from unified.txt...")
    pytorch_baseline_data = parse_log_file("unified.txt")

    if not pytorch_baseline_data:
        print("Error: Could not load cuBLAS baseline data!")
        return

    pytorch_sizes = [entry['size'] for entry in pytorch_baseline_data]
    pytorch_tflops = [entry['pytorch_tflops'] for entry in pytorch_baseline_data]

    # Generate progressive plots
    for i, (filename, label) in enumerate(optimization_sequence):
        print(f"Generating plot {i+1}: up to {label}")

        plt.figure(figsize=(12, 8))

        # Colors for different optimizations
        colors = plt.cm.tab10(np.linspace(0, 1, len(optimization_sequence)))

        max_tflops = 0

        # Plot all optimizations up to the current one
        for j in range(i + 1):
            opt_filename, opt_label = optimization_sequence[j]

            if not os.path.exists(opt_filename):
                print(f"  Warning: {opt_filename} not found, skipping...")
                continue

            data = parse_log_file(opt_filename)

            if not data:
                print(f"  Warning: No data found in {opt_filename}")
                continue

            sizes = [entry['size'] for entry in data]
            custom_tflops = [entry['custom_tflops'] for entry in data]

            plt.plot(sizes, custom_tflops, 'o-', color=colors[j],
                    label=opt_label, linewidth=2, markersize=6)

            max_tflops = max(max_tflops, max(custom_tflops))

        # Plot cuBLAS baseline
        plt.plot(pytorch_sizes, pytorch_tflops, 'k--',
                label='torch._scaled_mm (cuBLAS)', linewidth=2, alpha=0.8)

        max_tflops = max(max_tflops, max(pytorch_tflops))

        # Customize the plot
        plt.xlabel('Matrix Size (M = N = K)', fontsize=12, fontweight='bold')
        plt.ylabel('Performance (TFLOPS)', fontsize=12, fontweight='bold')
        plt.title(f'MXFP8 GEMM Optimization Progress - Step {i+1}', fontsize=14, fontweight='bold')

        # Set x-axis to show matrix sizes clearly
        plt.xscale('log')
        plt.yscale('linear')

        # Start y-axis at 0
        plt.ylim(0, max_tflops * 1.1)

        # Format x-axis ticks
        unique_sizes = sorted(set(pytorch_sizes))
        plt.xticks(unique_sizes, [f'{size}' for size in unique_sizes])

        # Add grid
        plt.grid(True, alpha=0.3)

        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Tight layout
        plt.tight_layout()

        # Save the plot
        output_file = f"optimization_step_{i+1:02d}_{label.lower().replace(' ', '_')}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")

        plt.close()  # Close the figure to free memory

def main():
    print("Generating progressive optimization plots...")
    plot_progressive_optimization()
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()
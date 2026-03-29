#!/usr/bin/env python3
import os
import re
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(file_path):
    """Parse a log file and extract benchmark data."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Find all benchmark entries
    pattern = r'Matrix size: M=(\d+), K=(\d+), N=(\d+)\s+Custom kernel:\s+([0-9.]+) us \(([0-9.]+) tflops\)\s+PyTorch _scaled_mm:\s+([0-9.]+) us \(([0-9.]+) tflops\)\s+Speedup: ([0-9.]+)x'

    matches = re.findall(pattern, content, re.MULTILINE)

    results = []
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

def plot_benchmarks(log_files, output_file=None, show_pytorch=False):
    """Create a plot comparing TFLOPS across different log files."""

    plt.figure(figsize=(12, 8))

    # Colors for different files
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))

    max_tflops = 0
    pytorch_baseline_data = None

    # If showing PyTorch baseline, try to load unified.txt for baseline data
    if show_pytorch:
        unified_path = None
        # Look for unified.txt in the same directory as the first log file
        if log_files:
            base_dir = os.path.dirname(os.path.abspath(log_files[0]))
            unified_path = os.path.join(base_dir, "unified.txt")

        # Also try current directory
        if not unified_path or not os.path.exists(unified_path):
            if os.path.exists("unified.txt"):
                unified_path = "unified.txt"

        if unified_path and os.path.exists(unified_path):
            print(f"Loading PyTorch baseline from {unified_path}...")
            pytorch_baseline_data = parse_log_file(unified_path)
        else:
            print("Warning: unified.txt not found, PyTorch baseline will not be shown")
            show_pytorch = False

    for i, log_file in enumerate(log_files):
        print(f"Processing {log_file}...")
        data = parse_log_file(log_file)

        if not data:
            print(f"  No benchmark data found in {log_file}")
            continue

        # Extract data for plotting
        sizes = [entry['size'] for entry in data]
        custom_tflops = [entry['custom_tflops'] for entry in data]

        # Get a clean name for the legend
        file_name = Path(log_file).stem

        # Plot custom kernel performance
        plt.plot(sizes, custom_tflops, 'o-', color=colors[i],
                label=f'{file_name}', linewidth=2, markersize=6)

        max_tflops = max(max_tflops, max(custom_tflops))

    # Plot single PyTorch baseline from unified.txt if available
    if show_pytorch and pytorch_baseline_data:
        pytorch_sizes = [entry['size'] for entry in pytorch_baseline_data]
        pytorch_tflops = [entry['pytorch_tflops'] for entry in pytorch_baseline_data]

        plt.plot(pytorch_sizes, pytorch_tflops, 'k--',
                label='torch._scaled_mm (cuBLAS)', linewidth=2, alpha=0.8)

        max_tflops = max(max_tflops, max(pytorch_tflops))

    # Customize the plot
    plt.xlabel('Matrix Size (M = N = K)', fontsize=12, fontweight='bold')
    plt.ylabel('Performance (TFLOPS)', fontsize=12, fontweight='bold')
    plt.title('GEMM Kernel Performance Comparison', fontsize=14, fontweight='bold')

    # Set x-axis to show matrix sizes clearly
    plt.xscale('log')
    plt.yscale('linear')

    # Start y-axis at 0 as requested
    plt.ylim(0, max_tflops * 1.1)

    # Format x-axis ticks to show matrix sizes nicely
    unique_sizes = sorted(set([entry['size'] for log_file in log_files for entry in parse_log_file(log_file)]))
    plt.xticks(unique_sizes, [f'{size}' for size in unique_sizes])

    # Add grid for easier reading
    plt.grid(True, alpha=0.3)

    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Tight layout to prevent legend cutoff
    plt.tight_layout()

    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot benchmark results from log files')
    parser.add_argument('files', nargs='*',
                        help='Log files to plot (default: all *.txt files in current directory)')
    parser.add_argument('-o', '--output',
                        help='Output file name for the plot (e.g., benchmarks.png)')
    parser.add_argument('-p', '--pytorch', action='store_true',
                        help='Also show PyTorch baseline performance')
    parser.add_argument('--pattern', default='*.txt',
                        help='File pattern to match if no files specified (default: *.txt)')

    args = parser.parse_args()

    # Determine which files to process
    if args.files:
        log_files = args.files
    else:
        log_files = glob.glob(args.pattern)

    if not log_files:
        print("No log files found!")
        return

    print(f"Found {len(log_files)} log files to process:")
    for file in log_files:
        print(f"  - {file}")
    print()

    # Create the plot
    plot_benchmarks(log_files, args.output, args.pytorch)

if __name__ == "__main__":
    main()
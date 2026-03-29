#!/usr/bin/env python3
import os
import re
import glob

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
            'custom_time_us': custom_time,
            'custom_tflops': custom_tflops,
            'pytorch_time_us': pytorch_time,
            'pytorch_tflops': pytorch_tflops,
            'speedup': speedup
        })

    return results

def create_markdown_file(txt_file, data):
    """Create a markdown file with the benchmark data in table format."""
    base_name = os.path.splitext(txt_file)[0]
    md_file = f"{base_name}.md"

    with open(md_file, 'w') as f:
        f.write(f"# Benchmark Results - {os.path.basename(base_name)}\n\n")

        if not data:
            f.write("No benchmark data found in this log file.\n")
            return

        # Write table header
        f.write("| Matrix Size (M×K×N) | Custom Kernel | PyTorch _scaled_mm | Speedup |\n")
        f.write("|---------------------|---------------|--------------------|---------|\n")

        # Write table rows
        for entry in data:
            f.write(f"| {entry['M']}×{entry['K']}×{entry['N']} | "
                   f"{entry['custom_time_us']:.3f} μs ({entry['custom_tflops']:.2f} TFLOPS) | "
                   f"{entry['pytorch_time_us']:.3f} μs ({entry['pytorch_tflops']:.2f} TFLOPS) | "
                   f"{entry['speedup']:.2f}x |\n")

def main():
    # Process all txt files in current directory
    txt_files = glob.glob("*.txt")

    for txt_file in txt_files:
        print(f"Processing {txt_file}...")
        data = parse_log_file(txt_file)
        create_markdown_file(txt_file, data)
        print(f"Created {os.path.splitext(txt_file)[0]}.md")

if __name__ == "__main__":
    main()
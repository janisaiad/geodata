#!/usr/bin/env python3
"""
Benchmark computation times for rho=0.7 with various epsilon values.
Saves results to JSON and creates a plot.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from ablation_studies import run_compute_time_benchmark
import json
import matplotlib.pyplot as plt
import numpy as np

EXPERIMENTS_DIR = Path("/home/janis/4A/geodata/experiments/pixelart")
OUTPUT_DIR = Path("/home/janis/4A/geodata/experiments/pixelart")

def plot_compute_times(json_path: Path):
    """Plot compute times from JSON file."""
    
    # Load data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    epsilons = []
    compute_times = []
    
    for entry in data['compute_times']:
        if entry['compute_time_seconds'] is not None:
            epsilons.append(entry['epsilon'])
            compute_times.append(entry['compute_time_seconds'])
    
    if len(epsilons) == 0:
        print("No valid compute times found!")
        return
    
    # Convert to numpy arrays
    epsilons = np.array(epsilons)
    compute_times = np.array(compute_times)
    
    
    # Create plot with log-log scale
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epsilons, compute_times, marker='o', linewidth=2, markersize=8, color='steelblue', label='Data')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$\\varepsilon$ (Blur parameter)', fontsize=12)
    ax.set_ylabel('Compute Time (seconds)', fontsize=12)
    ax.set_title(f'Computation Time vs $\\varepsilon$ ($\\rho={data["rho"]:.1f}$, $\\lambda={data["lambda_color"]:.1f}$) [Log-Log]', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    
    # Add red vertical bar for chosen eps = 0.01
    ax.axvline(0.01, color='red', linestyle='-', linewidth=3, alpha=0.8, label='Chosen $\\varepsilon = 0.01$')
    
    # Add legend after all lines are drawn
    ax.legend(loc='best')
    
    # Highlight eps = 0.01 (without label, legend will be in caption)
    if 0.01 in epsilons:
        idx = np.where(np.abs(epsilons - 0.01) < 1e-6)[0]
        if len(idx) > 0:
            idx = idx[0]
            ax.plot(epsilons[idx], compute_times[idx], marker='o', markersize=12, 
                    color='red', markeredgewidth=2, markeredgecolor='darkred', 
                    zorder=10)
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "compute_times_rho0.7.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    # Run benchmark to generate JSON with rho=0.7
    print("Running compute time benchmark...")
    results = run_compute_time_benchmark()
    
    # Plot results
    json_path = EXPERIMENTS_DIR / "compute_times_rho0.7.json"
    if json_path.exists():
        print("\nCreating plot...")
        plot_compute_times(json_path)
    else:
        print(f"JSON file not found: {json_path}")


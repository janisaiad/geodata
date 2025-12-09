#!/usr/bin/env python3
"""
Benchmark computation times for rho=1.0 with various epsilon values.
Saves results to JSON and creates a plot.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from ablation_studies import run_compute_time_benchmark, get_5d_cloud, load_image
import json
import matplotlib.pyplot as plt
import numpy as np
import torch

EXPERIMENTS_DIR = Path("/home/janis/4A/geodata/experiments/pixelart")
OUTPUT_DIR = Path("/home/janis/4A/geodata/experiments/pixelart")
DATA_DIR = Path("/home/janis/4A/geodata/data/pixelart/images")

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
    
    # Calculate cost matrix to get C_max and C_mean
    print("Calculating cost matrix for thumb rule...")
    img_source = load_image(DATA_DIR / "salameche.webp")
    img_target = load_image(DATA_DIR / "strawberry.jpg")
    resolution = 64
    lambda_color = data['lambda_color']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_a, _, _, _, _ = get_5d_cloud(img_source.to(device), resolution, lambda_color)
    X_b, _, _, _, _ = get_5d_cloud(img_target.to(device), resolution, lambda_color)
    
    # Compute cost matrix: C = ||x - y||^2 / 2
    dist_sq = torch.cdist(X_a, X_b, p=2) ** 2
    C_matrix = dist_sq / 2.0
    C_max = C_matrix.max().item()
    C_mean = C_matrix.mean().item()
    
    # Thumb rule: eps ≈ 0.01 * C_max or eps ≈ 0.01 * C_mean
    eps_rule_max = 0.01 * C_max
    eps_rule_mean = 0.01 * C_mean
    
    print(f"C_max = {C_max:.6f}, C_mean = {C_mean:.6f}")
    print(f"Thumb rule: eps ≈ 0.01 * C_max = {eps_rule_max:.6f}")
    print(f"Thumb rule: eps ≈ 0.01 * C_mean = {eps_rule_mean:.6f}")
    print(f"Chosen eps = 0.01, satisfies rule: {eps_rule_mean <= 0.01 <= eps_rule_max}")
    
    # Create plot with log-log scale
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epsilons, compute_times, marker='o', linewidth=2, markersize=8, color='steelblue', label='Data')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$\\varepsilon$ (Blur parameter)', fontsize=12)
    ax.set_ylabel('Compute Time (seconds)', fontsize=12)
    ax.set_title(f'Computation Time vs $\\varepsilon$ ($\\rho={data["rho"]:.1f}$, $\\lambda={data["lambda_color"]:.1f}$) [Log-Log]', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    
    # Add thumb rule lines with labels
    y_min, y_max = ax.get_ylim()
    ax.axvline(eps_rule_max, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Thumb rule: $\\varepsilon \\approx 0.01 \\times C_{{max}} = {eps_rule_max:.4f}$')
    ax.axvline(eps_rule_mean, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Thumb rule: $\\varepsilon \\approx 0.01 \\times C_{{mean}} = {eps_rule_mean:.4f}$')
    
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
    plot_path = OUTPUT_DIR / "compute_times_rho1.0.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    # Run benchmark
    print("Running compute time benchmark...")
    results = run_compute_time_benchmark()
    
    # Plot results
    json_path = EXPERIMENTS_DIR / "compute_times_rho1.0.json"
    if json_path.exists():
        print("\nCreating plot...")
        plot_compute_times(json_path)
    else:
        print(f"JSON file not found: {json_path}")


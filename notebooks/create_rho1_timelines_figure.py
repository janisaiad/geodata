#!/usr/bin/env python3
"""
Create a combined figure showing timelines for different eps values with rho=1.0
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

EXPERIMENTS_DIR = Path("/Data/janis.aiad/geodata/experiments/pixelart")
OUTPUT_DIR = Path("/Data/janis.aiad/geodata/experiments/pixelart")

epsilons = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]

# Load images and create a vertical stack
fig, axes = plt.subplots(len(epsilons), 1, figsize=(11, len(epsilons) * 1.8))

for idx, eps in enumerate(epsilons):
    img_path = EXPERIMENTS_DIR / f"transport_timeline_eps{eps:.3f}_rho1.00.png"
    if img_path.exists():
        img = mpimg.imread(img_path)
        axes[idx].imshow(img)
        axes[idx].axis('off')
        # Add label on the left
        axes[idx].text(-0.02, 0.5, f'$\\varepsilon={eps:.3f}$', 
                      transform=axes[idx].transAxes, fontsize=11, 
                      rotation=0, ha='right', va='center', fontweight='bold')
    else:
        axes[idx].axis('off')
        axes[idx].text(0.5, 0.5, f'eps={eps:.3f} not found', ha='center', va='center')

plt.suptitle('Transport Timelines for Different $\\varepsilon$ with $\\rho=1.0$', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0.03, 0, 1, 0.99])
output_path = OUTPUT_DIR / "rho1_timelines_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}")


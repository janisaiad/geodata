#!/usr/bin/env python3
"""
Visualization of the adaptive sigma formula for geometric expansion compensation.

This script plots:
1. The linear intrinsic component
2. The parabolic expansion component  
3. The combined sigma(t) function
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters (from your code)
sigma_start = 1.2  # Large for pixel art
sigma_end = 0.5    # Small for photo
sigma_boost = 0.5  # Boost strength

# Time range
t = np.linspace(0, 1, 100)

# Components
sigma_intrinsic = (1 - t) * sigma_start + t * sigma_end
sigma_expansion = sigma_boost * 4 * t * (1 - t)
sigma_total = sigma_intrinsic + sigma_expansion

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Top plot: Individual components
ax1 = axes[0]
ax1.plot(t, sigma_intrinsic, 'b-', linewidth=2, label=f'Linear: (1-t)·{sigma_start} + t·{sigma_end}')
ax1.plot(t, sigma_expansion, 'r--', linewidth=2, label=f'Parabolic: {sigma_boost}·4t(1-t)')
ax1.plot(t, sigma_total, 'g-', linewidth=3, label='Total: σ(t) = σ_intrinsic + σ_expansion')
ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='t=0.5 (maximum expansion)')
ax1.set_xlabel('Time t', fontsize=12)
ax1.set_ylabel('Kernel Width σ(t)', fontsize=12)
ax1.set_title('Adaptive Kernel Width: Linear + Parabolic Components', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 1])

# Add annotations
ax1.annotate('Maximum expansion\nat t=0.5', 
             xy=(0.5, sigma_total[np.argmin(np.abs(t - 0.5))]), 
             xytext=(0.3, 1.4),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
             fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Bottom plot: Parabolic function 4t(1-t) explanation
ax2 = axes[1]
parabola = 4 * t * (1 - t)
ax2.plot(t, parabola, 'purple', linewidth=3, label='4t(1-t)')
ax2.fill_between(t, 0, parabola, alpha=0.3, color='purple')
ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Maximum at t=0.5')
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel('Time t', fontsize=12)
ax2.set_ylabel('4t(1-t)', fontsize=12)
ax2.set_title('Parabolic Boost Function: Maximum at t=0.5', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1.1])

# Add key points
key_times = [0.0, 0.25, 0.5, 0.75, 1.0]
for t_key in key_times:
    idx = np.argmin(np.abs(t - t_key))
    value = parabola[idx]
    ax2.plot(t_key, value, 'ro', markersize=8)
    ax2.annotate(f't={t_key}\n{value:.2f}', 
                xy=(t_key, value), 
                xytext=(10, 10) if t_key < 0.5 else (-10, 10),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig('/Data/janis.aiad/geodata/notebooks/sigma_expansion_visualization.png', 
            dpi=150, bbox_inches='tight')
print("Visualization saved to: sigma_expansion_visualization.png")

# Print summary table
print("\n" + "="*60)
print("SUMMARY TABLE: σ(t) Components")
print("="*60)
print(f"{'t':<8} {'σ_intrinsic':<15} {'σ_expansion':<15} {'σ_total':<15}")
print("-"*60)
for t_key in key_times:
    idx = np.argmin(np.abs(t - t_key))
    sig_int = sigma_intrinsic[idx]
    sig_exp = sigma_expansion[idx]
    sig_tot = sigma_total[idx]
    print(f"{t_key:<8.2f} {sig_int:<15.3f} {sig_exp:<15.3f} {sig_tot:<15.3f}")

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. σ_intrinsic: Linear interpolation between source and target resolutions")
print("2. σ_expansion: Parabolic boost that peaks at t=0.5 (maximum expansion)")
print("3. σ_total: Combined kernel width compensates for both effects")
print("4. At t=0.5, particles are furthest apart → largest gaps → largest σ needed")
print("="*60)

plt.show()


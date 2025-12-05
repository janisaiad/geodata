# Maximum Geometric Expansion: Adaptive Kernel Width for Image Morphing

## The Problem: Geometric Expansion During Interpolation

When morphing between two images with **different intrinsic resolutions** (e.g., pixel art → photorealistic photo), the transport map causes particles to **spread apart** during interpolation. This creates a fundamental challenge:

### Why Expansion Happens at t=0.5

1. **At t=0**: Particles are at their source positions (dense pixel art)
2. **At t=0.5**: Particles are at the **midpoint** of their trajectories
   - They've moved away from source positions
   - They haven't reached target positions yet
   - **Maximum separation** occurs here → largest gaps between particles
3. **At t=1**: Particles reach target positions (dense photo)

### The "Tearing" Phenomenon

When particles spread apart, the **density decreases**. On a fixed grid, this creates:
- **Empty pixels** (holes)
- **Visual artifacts** (tearing)
- **Loss of visual continuity**

## The Solution: Adaptive Gaussian Kernel Width σ(t)

We use **Gaussian splatting** to reconstruct the image from scattered particles. The kernel width `σ(t)` must adapt to:
1. **Intrinsic resolution difference** (pixel art vs photo)
2. **Geometric expansion** (particles spreading at t=0.5)

## The Formula Breakdown

```python
sigma_intrinsic = (1 - t) * sigma_start + t * sigma_end
sigma_expansion = sigma_boost * 4 * t * (1 - t)
sigma_t = sigma_intrinsic + sigma_expansion
```

### Component 1: Linear Interpolation (Intrinsic Resolution)

```
σ_intrinsic(t) = (1-t) · σ_start + t · σ_end
```

- **σ_start** (large, e.g., 1.2): Matches pixel art's coarse structure
- **σ_end** (small, e.g., 0.5): Matches photo's fine details
- **Linear interpolation**: Smoothly transitions between intrinsic scales

**Example:**
- t=0: σ = 1.2 (large kernels for pixel art)
- t=0.5: σ = 0.85 (intermediate)
- t=1: σ = 0.5 (small kernels for photo)

### Component 2: Parabolic Boost (Geometric Expansion)

```
σ_expansion(t) = γ · 4t(1-t)
```

Where **γ = sigma_boost** (e.g., 0.5) controls the boost strength.

#### Why 4t(1-t)?

The function **4t(1-t)** is a **parabola** with these properties:

| t | 4t(1-t) | Interpretation |
|---|---------|----------------|
| 0 | 0 | No expansion boost needed (at source) |
| 0.25 | 0.75 | Moderate expansion |
| **0.5** | **1.0** | **Maximum expansion** → maximum boost |
| 0.75 | 0.75 | Moderate expansion |
| 1 | 0 | No expansion boost needed (at target) |

**Key insight:** The parabola reaches its **maximum at t=0.5**, exactly when geometric expansion is maximum!

#### Visual Representation

```
σ_expansion(t)
    ↑
  γ |        ╱╲
    |       ╱  ╲
    |      ╱    ╲
    |     ╱      ╲
    |    ╱        ╲
  0 |___╱__________╲___→ t
    0             1
              ↑
         Maximum at t=0.5
```

### Combined Formula

```
σ(t) = (1-t)·σ_start + t·σ_end + γ·4t(1-t)
```

**Example with σ_start=1.2, σ_end=0.5, γ=0.5:**

| t | σ_intrinsic | σ_expansion | σ_total |
|---|-------------|-------------|---------|
| 0.0 | 1.20 | 0.00 | **1.20** |
| 0.25 | 1.03 | 0.38 | **1.41** |
| **0.5** | **0.85** | **0.50** | **1.35** |
| 0.75 | 0.68 | 0.38 | **1.06** |
| 1.0 | 0.50 | 0.00 | **0.50** |

Notice how σ_total is **largest around t=0.5**, compensating for the maximum geometric expansion!

## Physical Interpretation

### Why Maximum Expansion at t=0.5?

Consider a particle moving from position A to position B:

```
t=0:     A ●───────────────○ B
         (particle here)

t=0.5:   A ○───────●───────○ B
         (particle at midpoint, maximum distance from neighbors)

t=1:     A ○───────────────● B
         (particle at target)
```

At t=0.5:
- Particles are **furthest from their neighbors**
- **Density is lowest**
- **Largest gaps** appear
- **Largest kernel width needed** to fill gaps

### The Nyquist-Shannon Condition

To avoid aliasing (holes), we need:

```
σ(t) ≥ spacing / 2
```

Where `spacing = √(H×W / N_particles)` is the average distance between particles.

The parabolic boost ensures this condition is satisfied even at maximum expansion.

## Code Implementation

```python
# From 5d_transport.ipynb

# B. Sigma "Intelligent"
sigma_intrinsic = (1 - t) * self.cfg.sigma_start + t * self.cfg.sigma_end
sigma_expansion = self.cfg.sigma_boost * 4 * t * (1 - t)

# Sigma final
sigma_t = max(sigma_intrinsic + sigma_expansion, min_sigma_t * 0.8)
```

The `max(..., min_sigma_t * 0.8)` ensures we never go below the Nyquist threshold.

## Summary

1. **Linear component**: Handles intrinsic resolution difference (pixel art ↔ photo)
2. **Parabolic component**: Compensates for maximum geometric expansion at t=0.5
3. **Result**: Smooth, hole-free interpolation that adapts to both resolution and geometry

The formula elegantly combines two physical phenomena:
- **Resolution mismatch** (handled linearly)
- **Geometric expansion** (handled parabolically, maximum at midpoint)

This ensures the Gaussian kernels are **just large enough** to fill gaps without over-blurring, maintaining visual quality throughout the morphing sequence.


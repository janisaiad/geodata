# Unbalanced OT Morphing for Geometric Morphing via Unbalanced Optimal Transport

## 📓 [Open in Google Colab](https://colab.research.google.com/github/janisaiad/geodata/blob/master/notebooks/5d_transport.ipynb)

## Abstract

Image interpolation is a fundamental problem in computer vision, traditionally framed as finding a geodesic path between probability distributions in Wasserstein space. However, standard discrete Optimal Transport (OT) methods face major obstacles when applied to images with structured content, disjoint color histograms, or distinct spectral features—as in Pixel Art. Balanced OT’s strict mass conservation produces non-physical flows, leading to "ghosting" where colors are interpolated over large distances between unrelated features. Moreover, advecting discrete pixels can cause geometric tearing, leaving holes in expanded regions, while processing color channels separately destroys chromatic coherence.

To address these issues, this project introduces a unified, mathematically principled pipeline based on Joint Unbalanced Optimal Transport. We embed images into a joint 5D spatial-color space ($\mathcal{X} \times \mathcal{C} \subset \mathbb{R}^5$), enabling feature-consistent interpolation while allowing local mass variation via Csiszár divergence penalties. Discretization artifacts are mitigated by a Gaussian Splatting reconstruction scheme: transport is computed in 5D, then projected back onto the 2D image plane by kernel density estimation. A fixed kernel width ($\sigma \approx 0.5$ pixels) is shown to optimally remove tearing artifacts while preserving the discrete, quantized nature of Pixel Art and its outliers.

We validate this approach by tracking the Unbalanced Sinkhorn Divergence ($S_{\varepsilon}$) along the geodesic path, demonstrating robust, artifact-minimizing interpolation even in outlier-rich settings.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [Tests](#tests)
- [License](#license)
- [Contact](#contact)
## Installation

To install dependencies using uv, follow these steps:

1. Install uv:
   
   **macOS/Linux:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Or using wget:
   ```bash
   wget -qO- https://astral.sh/uv/install.sh | sh
   ```

   **Windows:**
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Alternatively, you can install uv using:
   - pipx (recommended): `pipx install uv`
   - pip: `pip install uv`
   - Homebrew: `brew install uv`
   - WinGet: `winget install --id=astral-sh.uv -e`
   - Scoop: `scoop install main/uv`

2. Using uv in this project:

   - Initialize a new virtual environment:
   ```bash
   uv venv
   ```

   - Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Unix
   .venv\Scripts\activate     # On Windows
   ```

   - Install dependencies from requirements.txt:
   ```bash
   uv add -r requirements.txt
   ```


   - Add a new package:
   ```bash
   uv add package_name
   ```

   - Remove a package:
   ```bash
   uv remove package_name
   ```

   - Update a package:
   ```bash
   uv pip install --upgrade package_name
   ```

   - Generate requirements.txt:
   ```bash
   uv pip freeze > requirements.txt
   ```

   - List installed packages:
   ```bash
   uv pip list
   ```

## Warning

If you're using macOS or Python 3, replace `pip` with `pip3` in line 1 of ```launch.sh```

Replace with your project folder name (which means the name of the library you are deving) in :```tests/test_env.py: ```
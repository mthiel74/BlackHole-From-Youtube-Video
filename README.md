# Black Hole Ray Tracer: General Relativity Visualization

A physically accurate ray tracer for visualizing black holes using General Relativity. This project implements geodesic integration in Schwarzschild and Kerr spacetimes to show how light bends around black holes.

![Black Hole Visualization](https://img.shields.io/badge/Physics-General%20Relativity-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌌 Overview

This project contains **scientifically accurate** black hole visualizations based on Einstein's General Relativity. Unlike artistic renderings, these simulations solve the actual geodesic equations that govern how light travels through curved spacetime.

### Features

- ✨ **Schwarzschild Black Holes**: Non-rotating black holes with spherically symmetric spacetime
- 🌀 **Kerr Black Holes**: Rotating black holes with frame-dragging effects
- 📹 **Animation Generation**: Create orbital flybys and rotating visualizations
- ⚡ **GPU-Accelerated**: Uses Numba JIT compilation for high-performance rendering
- 🎯 **Physically Accurate**: Implements actual geodesic equations from GR

## 📁 Repository Structure - Which Files to Use

### **Main Files to Use:**

| File | Description | Use This For |
|------|-------------|--------------|
| **`BlackHoleAccretion.ipynb`** | Schwarzschild black hole with comprehensive annotations | **Primary tutorial** - Start here for learning |
| **`KerrBlackHole.ipynb`** | Rotating (Kerr) black hole with frame-dragging effects | Advanced rotating black holes |
| **`SchwarzschildBHfromVideo.ipynb`** | Streamlined Schwarzschild implementation | Quick experiments and rendering |
| **`blackholeaccretion.py`** | Standalone Python script | Command-line usage without Jupyter |

### **Documentation:**

| File | Description |
|------|-------------|
| **`PHYSICS.md`** | Deep mathematical derivations and equations |
| **`EXAMPLES.md`** | Practical usage tutorials and code examples |
| **`README.md`** | This file - project overview |

### **Backup/Archive Files (Don't Use):**

These files are preserved for reference but should **not** be used:

- `*_original.ipynb` - Original notebooks before cleanup
- `BlackHoleAccretion_broken.ipynb` - Old version with errors (replaced)

### **Quick Start Guide:**

1. **Complete beginner?** → Start with `BlackHoleAccretion.ipynb`
2. **Want to understand the physics?** → Read `PHYSICS.md` alongside the notebooks
3. **Need working code examples?** → Check `EXAMPLES.md`
4. **Want rotating black holes?** → Use `KerrBlackHole.ipynb`
5. **Prefer command-line?** → Run `python blackholeaccretion.py`

## 🔬 The Physics

### Is This Real Physics or Just a Cartoon?

**This is REAL PHYSICS.** The simulations solve the geodesic equations derived from Einstein's field equations. Here's what makes it scientifically accurate:

#### Schwarzschild Metric (Non-Rotating Black Hole)

The code implements the null geodesic equations for photons:

```
d²φ/dλ² = -(2/r) × (dr/dλ) × (dφ/dλ)
d²r/dλ² = r × (dφ/dλ)² × (1 - 3GM/2rc²)
```

Where:
- `λ` = affine parameter (path length along light ray)
- `r` = radial coordinate (distance from black hole)
- `φ` = azimuthal angle
- `GM/c²` = Schwarzschild radius (Rs)

**Key Physical Features:**
- **Event Horizon** at r = Rs: Nothing can escape from inside
- **Photon Sphere** at r = 1.5Rs: Unstable orbit where light circles the black hole
- **ISCO** at r = 3Rs: Innermost Stable Circular Orbit for matter
- **Gravitational Lensing**: Light rays bend according to spacetime curvature

#### Kerr Metric (Rotating Black Hole)

For rotating black holes, the code uses the full Kerr metric in Boyer-Lindquist coordinates:

```
ds² = -(1 - 2Mr/Σ)dt² + (Σ/Δ)dr² + Σdθ² + (r² + a²)sin²θ dφ² - (4aMr sin²θ/Σ)dt dφ
```

Where:
- `a` = angular momentum per unit mass (spin parameter)
- `Σ = r² + a²cos²θ`
- `Δ = r² - 2Mr + a²`
- `M` = black hole mass

**Kerr-Specific Effects:**
- **Frame Dragging**: Spacetime itself rotates around the black hole
- **Ergosphere**: Region where nothing can remain stationary
- **Asymmetric Shadow**: The "D-shaped" shadow seen from certain angles
- **Different ISCOs**: Prograde orbits can get closer than retrograde

### Numerical Integration

The code uses **4th-order Runge-Kutta** (RK4) integration:

```python
k1 = f(x)
k2 = f(x + 0.5*h*k1)
k3 = f(x + 0.5*h*k2)
k4 = f(x + h*k3)
x_new = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
```

This provides excellent accuracy for geodesic integration with adaptive step sizes near the event horizon.

## 🚀 Installation

### Requirements

```bash
pip install numpy matplotlib numba opencv-python
```

- **NumPy**: Array operations and mathematics
- **Matplotlib**: Visualization and plotting
- **Numba**: JIT compilation for GPU-like speeds on CPU
- **OpenCV**: Video generation (cv2)

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/black-hole-raytracer.git
cd black-hole-raytracer
```

2. **Run Schwarzschild simulation:**
```bash
python blackholeaccretion.py
```

3. **Run Kerr simulation (Jupyter):**
```bash
jupyter notebook KerrBlackHole.ipynb
```

## 📊 Usage

### Single Image Rendering

```python
import numpy as np
from blackholeaccretion import render_image

# Camera setup
CAM_DIST = 30.0  # Distance from black hole
CAM_PITCH = 25.0  # Viewing angle (degrees)

# Render
cam_pos = np.array([0.0, CAM_DIST*np.sin(np.radians(CAM_PITCH)),
                    -CAM_DIST*np.cos(np.radians(CAM_PITCH))])
cam_target = np.array([0.0, 0.0, 0.0])

image = render_image(800, 600, cam_pos, cam_target, fov=60.0)
```

### Video Generation

The code includes animation loops that generate orbital flybys:

- **Polar orbit**: Camera rotates in Y-Z plane
- **Inclined orbit**: 45° inclination for dramatic views
- **Equatorial orbit**: View the accretion disk edge-on

## 🎨 Visualization Components

### Accretion Disk

The disk model includes:
- **Radial bounds**: 3Rs (ISCO) to 12Rs (outer edge)
- **Doppler beaming**: Approaching side appears brighter
- **Temperature gradient**: Hotter (yellow) near center, cooler (red) at edges
- **Procedural texture**: Ring patterns from disk turbulence

### Color Coding

- **Black**: Event horizon shadow (r < Rs)
- **Orange/Yellow**: Hot inner disk (near ISCO)
- **Red**: Cooler outer disk
- **Dark blue**: Escaped rays (background space)
- **White dots**: Background stars (simple starfield)

## 📐 Mathematical Details

### Geodesic Equations in Schwarzschild

Starting from the Schwarzschild metric and the geodesic equation:

```
d²x^μ/dλ² + Γ^μ_αβ (dx^α/dλ)(dx^β/dλ) = 0
```

For the equatorial plane (θ = π/2), this reduces to:

**Radial equation:**
```
d²r/dλ² = (r - 3M)(L/r³)²
```
where L is the conserved angular momentum.

**Angular equation:**
```
d²φ/dλ² = -(2/r)(dr/dλ)(dφ/dλ)
```

These are **exactly** what the code implements!

### Ray Tracing Algorithm

1. **Initialize**: Camera position (r₀, θ₀, φ₀) and ray direction
2. **Transform**: Convert 3D ray direction to 2D polar plane (planar geodesics)
3. **Integrate**: Use RK4 to evolve state [r, dr/dλ, φ, dφ/dλ]
4. **Check collisions**:
   - r < Rs → Event horizon (return black)
   - r > 50Rs → Escaped to infinity (return sky color)
   - Crossed equatorial plane → Check if hit accretion disk
5. **Texture lookup**: If disk hit, compute color based on radius and velocity

## 🎥 Example Videos

This repository includes example videos demonstrating the ray tracer's capabilities:

### Schwarzschild Black Hole
- **`BlackHoleGemini3.mp4`** (5.7 MB) - Schwarzschild black hole with accretion disk
  - Shows gravitational lensing and photon sphere effects
  - Demonstrates Doppler beaming on approaching side of disk

### Kerr (Rotating) Black Hole
- **`kerr_black_hole_HR_orbit.mp4`** (11 MB) - High-resolution Kerr black hole orbital animation
  - Frame-dragging effects visible in asymmetric shadow
  - D-shaped shadow characteristic of rotating black holes
  - 1000×750 resolution

These videos showcase:
- ✨ Gravitational lensing creating multiple images
- 🌀 Accretion disk with realistic brightness gradients
- 🔴 Doppler beaming (approaching side brighter)
- ⚫ Black hole shadow and photon sphere
- 🎬 Smooth orbital camera movements

## ⚙️ Configuration Parameters

### Physics Parameters

```python
RS = 1.0              # Schwarzschild radius (sets scale)
SPIN_A = 0.99         # Kerr spin (0 = Schwarzschild, 1 = extremal)
STEP_SIZE = 0.05      # Integration step size
MAX_STEPS = 5000      # Max steps before assuming ray escaped
```

### Rendering Parameters

```python
WIDTH = 800           # Image width (pixels)
HEIGHT = 600          # Image height (pixels)
FOV = 60.0            # Field of view (degrees)
CAM_DIST = 30.0       # Camera distance (in units of Rs)
```

### Accretion Disk

```python
R_ISCO = 3.0 * RS     # Inner edge (ISCO for Schwarzschild)
R_DISK_MAX = 12.0 * RS  # Outer edge
```

## 🔍 Scientific Accuracy

### What's Accurate:

✅ Geodesic equations from General Relativity
✅ Gravitational lensing (Einstein rings, multiple images)
✅ Event horizon size and location
✅ Photon sphere at 1.5Rs
✅ Frame dragging in Kerr metric
✅ Doppler beaming (relativistic boosting)

### Simplifications:

⚠️ Thin disk approximation (real disks have height)
⚠️ No radiative transfer (uses procedural textures)
⚠️ Simplified Doppler formula
⚠️ No redshift calculation
⚠️ No gravitational time dilation effects on photon energy

### Comparison to Real Observations

This simulation produces images similar to:
- **Event Horizon Telescope** images of M87* and Sgr A*
- **Interstellar** movie visualizations (though those used more detailed disk physics)
- **Scientific papers** on black hole shadows and photon rings

## 📚 References

### Academic Sources

1. **Chandrasekhar, S.** (1983). *The Mathematical Theory of Black Holes*
2. **Misner, Thorne, Wheeler** (1973). *Gravitation* - The classic GR textbook
3. **Luminet, J.P.** (1979). Image of a spherical black hole with thin accretion disk
4. **James et al.** (2015). Gravitational lensing by spinning black holes (Interstellar)
5. **Event Horizon Telescope Collaboration** (2019). First M87 Black Hole Image

### Implementation Resources

- **Video Tutorial**: [Schwarzschild Black Hole Implementation](https://www.youtube.com/watch?v=8-B6ryuBkCM) - Educational video that inspired the Schwarzschild ray tracer implementation

## 🤝 Contributing

Contributions welcome! Possible improvements:

- [ ] Add redshift calculations for spectral lines
- [ ] Implement thick disk models
- [ ] Add radiative transfer
- [ ] GPU acceleration with CUDA
- [ ] Interactive 3D visualization
- [ ] Schwarzschild-de Sitter (cosmological constant)
- [ ] Reissner-Nordström (charged black holes)

## 📄 License

MIT License - Feel free to use for educational or research purposes

## 🙏 Acknowledgments

This project was developed with AI assistance:
- **Schwarzschild implementation**: Created with Google Gemini, based on [this educational video](https://www.youtube.com/watch?v=8-B6ryuBkCM)
- **Documentation and Kerr enhancements**: Developed with Anthropic Claude

Based on the physics of General Relativity and inspired by:
- Kip Thorne's work on black hole visualization
- The Event Horizon Telescope project
- Jean-Pierre Luminet's pioneering 1979 raytracing work

---

**Note**: This is a scientific visualization tool. While the physics is accurate, real black hole observations require sophisticated radio telescope networks and years of data analysis!

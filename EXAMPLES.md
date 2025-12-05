# Usage Examples

This document provides practical examples for using the black hole ray tracer.

## Quick Start: Single Image

### Schwarzschild Black Hole

```python
python blackholeaccretion.py
```

This will:
1. Compile the Numba JIT kernels (~5-10 seconds)
2. Render an 800×600 image (~20-30 seconds)
3. Save as `black_hole.jpg`
4. Display using matplotlib

**Expected output:**
```
Initializing Black Hole Simulation...
Resolution: 800x600
Metric: Schwarzschild
Compiling JIT kernels (this might take a few seconds)...
Compilation finished in 8.34s
Rendering final image...
Render finished in 23.12s
Image saved as 'black_hole.jpg'
```

## Custom Parameters

### Changing Camera Position

```python
from blackholeaccretion import render_image, CAM_DIST, FOV
import numpy as np
import matplotlib.pyplot as plt

# Move camera further back
CAM_DIST = 50.0
CAM_PITCH = 30.0  # Look down from above

pitch_rad = np.radians(CAM_PITCH)
cam_pos = np.array([0.0,
                    CAM_DIST * np.sin(pitch_rad),
                    -CAM_DIST * np.cos(pitch_rad)])
cam_target = np.array([0.0, 0.0, 0.0])

# Render
img = render_image(1920, 1080, cam_pos, cam_target, FOV)

# Display
plt.figure(figsize=(16, 9))
plt.imshow(np.clip(img, 0, 1))
plt.axis('off')
plt.savefig('black_hole_hd.jpg', dpi=150, bbox_inches='tight')
plt.show()
```

### Edge-On View (Accretion Disk Profile)

```python
# Position camera in equatorial plane
CAM_PITCH = 0.0  # No vertical offset
cam_pos = np.array([0.0, 0.0, -CAM_DIST])

img = render_image(1600, 900, cam_pos, cam_target, 45.0)
```

## Video Generation

### Polar Orbit Animation

```python
import numpy as np
import cv2
from blackholeaccretion import render_image, CAM_DIST, FOV

NUM_FRAMES = 120
FPS = 24
WIDTH, HEIGHT = 640, 480

frames = []
for i in range(NUM_FRAMES):
    theta = (2.0 * np.pi * i) / NUM_FRAMES

    # Circular orbit in Y-Z plane
    cam_pos = np.array([0.0,
                        CAM_DIST * np.sin(theta),
                        -CAM_DIST * np.cos(theta)])
    target = np.array([0.0, 0.0, 0.0])

    # Render frame
    img = render_image(WIDTH, HEIGHT, cam_pos, target, FOV)
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    frames.append(frame_bgr)

    print(f"Frame {i+1}/{NUM_FRAMES}")

# Save video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('orbit.mp4', fourcc, FPS, (WIDTH, HEIGHT))
for frame in frames:
    out.write(frame)
out.release()
```

### Zoom-In Animation

```python
# Start far away, zoom toward photon sphere
START_DIST = 50.0
END_DIST = 5.0
NUM_FRAMES = 180

for i in range(NUM_FRAMES):
    # Exponential zoom for dramatic effect
    t = i / NUM_FRAMES
    dist = START_DIST * (END_DIST/START_DIST)**t

    cam_pos = np.array([0.0, dist * 0.3, -dist * 0.95])
    img = render_image(WIDTH, HEIGHT, cam_pos, cam_target, FOV)
    # ... save frame
```

## Kerr Black Hole (Jupyter)

### Basic Kerr Render

Open `KerrBlackHole.ipynb` in Jupyter and run all cells, or:

```python
from KerrBlackHole import render_frame_data, SPIN_A, CAM_DIST, FOV
import numpy as np

# Camera at 85° inclination (nearly edge-on)
cam_r = CAM_DIST
cam_theta = np.radians(85.0)
cam_phi = 0.0

# Render
img = render_frame_data(800, 600, cam_r, cam_theta, cam_phi, FOV)

import matplotlib.pyplot as plt
plt.imshow(np.clip(img, 0, 1))
plt.axis('off')
plt.title(f"Kerr Black Hole (a = {SPIN_A})")
plt.show()
```

### Varying Spin Parameter

```python
# Compare different spins
spins = [0.0, 0.5, 0.9, 0.99]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, a in zip(axes, spins):
    # Temporarily set spin
    global SPIN_A
    SPIN_A = a

    # Re-calculate horizon
    R_HORIZON = 1.0 + np.sqrt(1.0 - a**2)

    img = render_frame_data(400, 300, CAM_DIST,
                            np.radians(85), 0.0, FOV)

    ax.imshow(np.clip(img, 0, 1))
    ax.set_title(f'a = {a}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('spin_comparison.jpg')
```

**Expected result:** See the shadow change from circular (a=0) to D-shaped (a=0.99).

## Advanced Examples

### Multiple Views Grid

```python
# Grid of different viewing angles
angles = [0, 15, 30, 45, 60, 75, 90]
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for i, angle in enumerate(angles):
    ax = axes[i // 4, i % 4]

    pitch_rad = np.radians(angle)
    cam_pos = np.array([0.0,
                        CAM_DIST * np.sin(pitch_rad),
                        -CAM_DIST * np.cos(pitch_rad)])

    img = render_image(400, 300, cam_pos, cam_target, FOV)

    ax.imshow(np.clip(img, 0, 1))
    ax.set_title(f'Pitch: {angle}°')
    ax.axis('off')

axes[1, 3].axis('off')  # Hide last subplot
plt.tight_layout()
plt.savefig('viewing_angles.jpg', dpi=100)
```

### Accretion Disk Only (No Background)

Modify `ray_march` to return magenta for escaped rays:

```python
# In blackholeaccretion.py, change:
if new_r > 50.0:
    return np.array([1.0, 0.0, 1.0])  # Magenta for debugging

# After rendering:
img[img[:,:,0] > 0.9] = [0, 0, 0]  # Make magenta -> black
```

This isolates just the gravitationally lensed disk.

### High-Resolution Wallpaper

```python
# Ultra-HD render (4K)
WIDTH = 3840
HEIGHT = 2160

# Wider field of view for desktop wallpaper
FOV = 70.0

# Position for aesthetic composition
CAM_DIST = 35.0
CAM_PITCH = 20.0

pitch_rad = np.radians(CAM_PITCH)
cam_pos = np.array([0.0,
                    CAM_DIST * np.sin(pitch_rad),
                    -CAM_DIST * np.cos(pitch_rad)])

print("This will take several minutes...")
img = render_image(WIDTH, HEIGHT, cam_pos, cam_target, FOV)

plt.figure(figsize=(19.2, 10.8), dpi=200)
plt.imshow(np.clip(img, 0, 1))
plt.axis('off')
plt.savefig('black_hole_4k.png', dpi=200, bbox_inches='tight', pad_inches=0)
```

**Note:** A 4K image may take 5-10 minutes to render!

## Performance Tips

### Reduce Resolution for Testing

```python
# Quick preview (renders in ~2 seconds)
img = render_image(200, 150, cam_pos, cam_target, FOV)
```

### Adjust Integration Parameters

```python
# Faster but less accurate
MAX_STEPS = 2000  # Default: 5000
STEP_SIZE = 0.1   # Default: 0.05

# Slower but more accurate (captures fine details)
MAX_STEPS = 10000
STEP_SIZE = 0.02
```

### Parallel Rendering

The code already uses Numba's parallel mode. To control thread count:

```bash
# Use 8 threads
export NUMBA_NUM_THREADS=8
python blackholeaccretion.py
```

### Batch Rendering with Multiprocessing

```python
from multiprocessing import Pool
import numpy as np

def render_frame(i):
    theta = (2.0 * np.pi * i) / NUM_FRAMES
    cam_pos = np.array([0.0,
                        CAM_DIST * np.sin(theta),
                        -CAM_DIST * np.cos(theta)])
    img = render_image(WIDTH, HEIGHT, cam_pos, cam_target, FOV)

    # Save immediately to disk
    filename = f"frames/frame_{i:04d}.png"
    cv2.imwrite(filename, (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., ::-1])
    return filename

# Render in parallel
with Pool(processes=4) as pool:
    results = pool.map(render_frame, range(NUM_FRAMES))
```

## Troubleshooting

### "Compilation takes too long"

First run compiles JIT code. Subsequent runs are instant:

```python
# First run: ~10s compilation + 20s render
# Second run: ~20s render (no compilation!)
```

### "Image is all black"

- Camera might be inside event horizon (CAM_DIST < 1.0)
- Check camera is looking at origin
- Increase MAX_STEPS if rays are escaping too soon

### "Disk looks wrong"

- Ensure camera pitch is not 0° (looking along disk plane)
- Adjust R_ISCO and R_DISK_MAX for disk size
- Check texture_lookup function for color mapping

### "Memory error"

Reduce resolution:
```python
# Too large
WIDTH = 8000  # ❌

# More reasonable
WIDTH = 1920  # ✓
```

## Contributing Your Own Examples

Found a cool configuration? Submit a pull request with:

1. Code snippet
2. Output image
3. Brief description
4. Parameter values used

See `CONTRIBUTING.md` for guidelines.

## Next Steps

- Experiment with different camera paths
- Modify disk colors and patterns
- Try different integration accuracies
- Combine with background star fields
- Add post-processing (bloom, lens flares)

Have fun exploring curved spacetime! 🌌

# GeoPalette

<img src="https://raw.githubusercontent.com/igorpawelec/GeoPalette/main/www/geopalette_logo.png" align="right" width="200"/>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**Color space conversions for geospatial raster data.**

A pure-NumPy Python package for converting RGB raster bands into 14 color spaces commonly used in remote sensing, image segmentation, and forestry applications. Designed for 2-D arrays (GeoTIFF bands) — fully vectorized, no pixel-level loops.

## Background

Color space transformations are fundamental to object-based image analysis (OBIA) and superpixel segmentation in remote sensing. This package implements standard CIE colorimetry with proper sRGB linearization (IEC 61966-2-1) and D65 illuminant, ensuring mathematically correct conversions suitable for scientific applications.

Key references for the implemented color spaces:

- **CIELAB, CIELUV, xyY:** CIE 15:2004, *Colorimetry*. Commission Internationale de l'Éclairage.
- **DIN99:** DIN 6176:2003, *Farbmetrische Bestimmung von Farbabständen bei Körperfarben nach der DIN99-Formel*.
- **Oklab:** Ottosson, B. (2020). *A perceptual color space for image processing*. https://bottosson.github.io/posts/oklab/
- **Jzazbz / JzCzHz:** Safdar, M., Cui, G., Kim, Y.J., & Luo, M.R. (2017). Perceptually uniform color space for image signals including high dynamic range and wide gamut. *Optics Express*, 25(13), 15131–15151. https://doi.org/10.1364/OE.25.015131
- **sRGB linearization:** IEC 61966-2-1:1999. *Multimedia systems and equipment — Colour measurement and management*.

## Supported color spaces

| Space | Components | Category |
|-------|-----------|----------|
| **HSL** | H (0–360°), S, L (0–1) | Hue-based |
| **HSV** | H (0–360°), S, V (0–1) | Hue-based |
| **HSI** | H (0–360°), S (%), I (0–255) | Hue-based |
| **CIELAB** | L* (0–100), a*, b* | Perceptual (CIE) |
| **DIN99 (DLab)** | L*, a*, b*, L99, a99, b99 | Perceptual (DIN) |
| **Oklab** | L, a, b | Perceptual (modern) |
| **CIELUV** | L*, u*, v* | Perceptual (CIE) |
| **LCH(ab)** | L*, C, H (0–360°) | Cylindrical CIELAB |
| **LCH(uv)** | L*, C, H (0–360°) | Cylindrical CIELUV |
| **xyY** | x, y (chromaticity), Y (luminance) | CIE chromaticity |
| **JCH** | J, C, H (0–360°) | CIECAM02-like |
| **YCbCr** | Y (16–235), Cb, Cr (16–240) | Video (BT.601) |
| **Jzazbz** | Jz, az, bz | HDR perceptual |
| **JzCzHz** | Jz, Cz, hz (0–360°) | HDR cylindrical |

Inverse conversions: **CIELAB → RGB**, **Oklab → RGB**, **HSV → RGB**, **HSL → RGB**.

## Installation

**Recommended (conda + pip):**

```bash
# 1. Install native dependencies via conda
conda install -c conda-forge numpy rasterio

# 2. Install geopalette
pip install --no-deps .               # from cloned repo
# or
pip install --no-deps git+https://github.com/igorpawelec/geopalette.git
```

**Minimal (NumPy only, no GeoTIFF I/O):**

```bash
pip install numpy
pip install --no-deps .
```

> **Note:** The `--no-deps` flag prevents pip from overwriting conda packages. Rasterio is only needed for GeoTIFF I/O — all conversion functions work with plain NumPy arrays.

## Quick start

### Python API

```python
import numpy as np
from geopalette import convertbands, available_spaces

# Check available spaces
print(available_spaces())
# ['dlab', 'hsi', 'hsl', 'hsv', 'jch', 'jzazbz', 'jzczhz', 'lab', ...]

# Convert synthetic data
R = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
G = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
B = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

comps, names = convertbands(R, G, B, "lab")
print(names)  # ['L', 'a', 'b']
```

### With GeoTIFF (rasterio)

```python
from geopalette.io_utils import convert_raster

convert_raster(
    "ortho_rgb.tif",
    "results/",
    "lab",
    save_multiband=True,
    save_singlebands=True,
)
```

### Command line

```bash
geopalette -i ortho_rgb.tif -o results/ -s lab
geopalette -i ortho_rgb.tif -o results/ -s oklab --single-bands
```

### As Python module

```bash
python -m geopalette -i ortho_rgb.tif -o results/ -s lab
```

### Individual functions

```python
from geopalette import rgb_to_lab, rgb_to_oklab, rgb_to_jzazbz
from geopalette import lab_to_rgb, oklab_to_rgb, hsv_to_rgb, hsl_to_rgb

L, a, b = rgb_to_lab(R, G, B)
R2, G2, B2 = lab_to_rgb(L, a, b)  # inverse (sRGB [0,1])

from geopalette import rgb_to_hsv
H, S, V = rgb_to_hsv(R, G, B)
R2, G2, B2 = hsv_to_rgb(H, S, V)  # inverse (sRGB [0,1])
```

## Repository structure

```
geopalette/
├── geopalette/           # Package source
│   ├── __init__.py       # Public API
│   ├── __main__.py       # CLI entry point
│   ├── conversions.py    # All conversion functions
│   └── io_utils.py       # GeoTIFF read/write helpers
├── test_data/            # Sample rasters
├── www/                  # Logo & comparison images
├── geopalette_test.py    # Example usage script
├── test_conversions.py   # Validation tests
├── pyproject.toml
├── requirements.txt
├── environment.yaml
├── tests/                # Pytest suite
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE
```

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.21
- Rasterio ≥ 1.3 *(optional, for GeoTIFF I/O)*

## Citation

If you use this software in your research, please cite:

1. **This implementation:**

   > Pawelec, I. (2025). GeoPalette — Color space conversions for geospatial raster data [Software]. https://github.com/igorpawelec/geopalette

2. **For CIELAB/CIELUV conversions:**

   > CIE 15:2004. Colorimetry (3rd ed.). Commission Internationale de l'Éclairage.

3. **For Oklab:**

   > Ottosson, B. (2020). A perceptual color space for image processing. https://bottosson.github.io/posts/oklab/

4. **For Jzazbz/JzCzHz:**

   > Safdar, M., Cui, G., Kim, Y.J., & Luo, M.R. (2017). Perceptually uniform color space for image signals including high dynamic range and wide gamut. *Optics Express*, 25(13), 15131–15151.

See also [CITATION.cff](CITATION.cff).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

# geopalette

<img src="https://raw.githubusercontent.com/igorpawelec/GeoPalette/main/www/geopalette_logo.png" align="right" width="200"/>
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**Color space conversions for geospatial raster data.**

A pure-NumPy Python package for converting RGB raster bands into 13 color spaces commonly used in remote sensing, image segmentation, and forestry applications. Designed for 2-D arrays (GeoTIFF bands) — no loops, fully vectorised.

## Supported color spaces

| Space | Components | Category |
|-------|-----------|----------|
| **HSL** | H (0–360°), S, L (0–1) | Hue-based |
| **HSV** | H (0–360°), S, V (0–1) | Hue-based |
| **HSI** | H (0–360°), S (%), I (0–255) | Hue-based |
| **CIELAB** | L* (0–100), a*, b* | Perceptual |
| **DIN99 (DLab)** | L*, a*, b*, L99, a99, b99 | Perceptual |
| **Oklab** | L, a, b | Perceptual |
| **CIELUV** | L*, u*, v* | Perceptual |
| **LCH(ab)** | L*, C, H (0–360°) | Cylindrical |
| **LCH(uv)** | L*, C, H (0–360°) | Cylindrical |
| **xyY** | x, y (chromaticity), Y (luminance) | CIE |
| **JCH** | J, C, H (0–360°) | CIECAM02-like |
| **YCbCr** | Y (16–235), Cb, Cr (16–240) | Video |
| **JzCzHz** | Jz, Cz, hz (0–360°) | HDR perceptual |

Plus one inverse conversion: **CIELAB → RGB**.

## Installation

**Recommended (conda + pip):**

```bash
# 1. Install native dependencies via conda
conda install -c conda-forge numpy rasterio

# 2. Install geopalette
pip install .               # from cloned repo
# or
pip install --no-deps git+https://github.com/igorpawelec/geopalette.git
```

**Minimal (NumPy only, no I/O):**

```bash
pip install numpy
pip install --no-deps .
```

## Quick start

### Python API

```python
import numpy as np
from geopalette import convertbands, available_spaces

# Check available spaces
print(available_spaces())
# ['dlab', 'hsi', 'hsl', 'hsv', 'jch', 'jzczhz', 'lab', ...]

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
    "results/lab",
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
from geopalette import rgb_to_lab, rgb_to_oklab, lab_to_rgb

L, a, b = rgb_to_lab(R, G, B)
R2, G2, B2 = lab_to_rgb(L, a, b)  # inverse
```

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.21
- Rasterio ≥ 1.3 *(optional, for GeoTIFF I/O)*

## Repository structure

```
geopalette/
├── geopalette/           # Package source
│   ├── __init__.py       # Public API
│   ├── __main__.py       # CLI entry point
│   ├── conversions.py    # All conversion functions
│   └── io_utils.py       # GeoTIFF read/write helpers
├── tests/                # Pytest suite
├── test_data/            # Sample rasters
├── www/                  # Logo & assets
├── geopalette_test.py    # Example usage script
├── pyproject.toml
├── requirements.txt
├── environment.yaml
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE
```

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

## Citation

If you use this software in your research, please cite it.
See [CITATION.cff](CITATION.cff).

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

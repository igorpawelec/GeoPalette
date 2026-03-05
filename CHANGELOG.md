# Changelog

## [0.1.0] — 2025-XX-XX

### Added
- Initial release
- 13 forward color space conversions: HSL, HSV, HSI, CIELAB, DIN99 (DLab), 
  Oklab, CIELUV, LCH(ab), LCH(uv), xyY, JCH, YCbCr, JzCzHz
- 1 inverse conversion: CIELAB → RGB
- `convertbands()` dispatcher for easy single-call conversion
- `convert_raster()` I/O utility for GeoTIFF workflows
- CLI: `python -m geopalette` / `geopalette` command
- Test suite with synthetic data

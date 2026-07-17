# Changelog

## [0.3.0]

Three conversions produced wrong numbers. The unit tests all passed, because
they only ever checked shapes, dtypes and ranges — never a value against a
reference. `tests/test_reference.py` now compares every space to
`colour-science` and `scikit-image`, and each fix below is guarded by a test
that fails if it is reverted.

**Minkowski-free note for existing users:** `lab`, `luv`, `hsv`, `hsl`,
`xyY`, `oklab`, `lchab`, `lchuv` and `ycbcr` were already correct and are
unchanged. If you only used those, your results stand.

### Fixed
- **Jzazbz was wrong twice over.** It skipped the pre-scaling of X and Y that
  Safdar et al. (2017) define in eq. 8-9 (`X' = b·X − (b−1)·Z`,
  `Y' = g·Y − (g−1)·X`, with b=1.15, g=0.66) — the LMS matrix is defined
  against those primed values, not raw XYZ. It also used `d_0` (1.63e-11) in
  the Jz curve where the constant `d` (−0.56) belongs, which collapsed the
  formula to `Jz ≈ Iz`. Jz was out by ~0.17 against a range of 0.01-0.21.
  Both now match `colour.XYZ_to_Jzazbz` to float32 precision. `jzczhz` is
  derived from the same numbers and was wrong with it.
- **DIN99 dropped the 0.7 factor.** DIN 6176 defines
  `f = 0.7·(−a*·sin16° + b*·cos16°)`; without it a99/b99 were out by up to
  7 units. The whole point of DIN99 is that one unit is one perceptual step,
  so the scale has to be right.
- **RuntimeWarnings on achromatic pixels.** `hsl`, `hsv`, `jch`, `luv` and
  `dlab` divided by zero on every grey, white or black pixel, then discarded
  the NaN. The results were right and the warnings were noise — but grey,
  shadow and water are everywhere in a raster. Two causes: masking after
  dividing instead of indexing first, and `np.where`, which is not lazy and
  evaluates both branches.

### Changed
- **`ycbcr` docstring said "full-range" while implementing studio swing.**
  The formula is BT.601 legal range — black is Y=16, white is Y=235 — and
  matches `colour-science` with `out_legal=True`. Only the docstring was
  wrong, but it was wrong in a way that would silently mis-scale someone's
  data.
- **`jch` is documented for what it is.** It is not CIECAM02: no chromatic
  adaptation, no surround, no background. It tracks real CIECAM02 at r≈0.98
  on J and r≈0.89 on C, but hue can be off by 70°. Its H spans 0-324°, not
  the 0-360° previously documented, because it scales an HSV hue by 0.9.
- `test_conversions.py` moved to `tests/`, where its own header already said
  it lived. `geopalette_test.py`, which is an example rather than a test,
  moved to `examples/convert_geotiff.py`.
- `test_available_spaces_count` asserted a hardcoded 13 against 14 registered
  spaces, so it failed for whoever added the fourteenth. It now checks the
  properties that matter: sorted, unique, non-empty.
- Licence declared as an SPDX expression with `license-files` (PEP 639); the
  TOML table form is deprecated and stops working 2027-02-18. Needs
  setuptools >= 77.

- **`python -m geopalette` always exited 0.** `__main__.py` called `main()`
  without passing its return value to `sys.exit()` — `sys` was imported and
  unused, which was the tell — so a failed conversion reported success and
  any script checking `$?` missed it. It also printed raw tracebacks for a
  missing file; user mistakes now get one clean line on stderr and exit 1.
- Unused imports removed; `pyflakes` is clean outside the deliberate
  re-exports in `__init__.py`.

### Added
- GitHub Actions CI: tests and lint on Linux/macOS/Windows across Python
  3.9-3.12.
- `pip install geopalette[validate]` pulls the reference implementations.

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

# Contributing to geopalette

Contributions are welcome! Here's how to get started.

## Development setup

```bash
git clone https://github.com/igorpawelec/geopalette.git
cd geopalette
conda env create -f environment.yaml
conda activate geopalette
pip install -e .
```

## Running tests

```bash
pytest tests/
```

## Adding a new color space

1. Add the conversion function to `geopalette/conversions.py`
2. Register it in the `_CONVERSIONS` dictionary
3. Add it to the `__init__.py` imports
4. Add a test in `tests/test_conversions.py`
5. Update `README.md` with the new space

## Code style

- Follow PEP 8
- Use NumPy-style docstrings
- All functions operate on 2-D NumPy arrays

## Pull requests

- Fork the repo and create a feature branch
- Include tests for new functionality
- Update the CHANGELOG.md

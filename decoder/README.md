# Decoder Package

Neural network decoder experiments for analyzing and predicting properties of trained models.

## Quick Start

```bash
# Verify setup
python decoder/verify_setup.py

# Run experiments
python -m decoder.main
```

## Structure

```
decoder/
├── config.py              # Central configuration and path utilities
├── models.py              # Model registry
├── decoder_models.py      # Decoder architectures (FC, Transformer)
├── experiments.py         # Main experiment runners
├── setup/                 # Setup functions for different tasks
│   ├── class_id.py       # Output class prediction
│   ├── input_pixel.py    # Input pixel decoding
│   └── dataset_classification.py  # Dataset type classification
└── underlying_datasets/   # Data modules for loading models
```

## Configuration

All paths use absolute references via `decoder.config.get_underlying_path()`:

```python
from decoder.config import get_underlying_path

# Get path to underlying models
model_path = get_underlying_path('saved_models/my_model.pt')
```

Config dictionaries use safe `.get()` access with defaults:

```python
config = {
    'hidden_dim': [50, 50],
    'decoder_class': 'TransformerDecoder',
    # ... other options
}

# Layer index automatically calculated from architecture
layer_idx = len(config.get('hidden_dim', [50, 50]))
```

## Key Features

- **No relative imports**: All imports use `from decoder.X import Y`
- **No path manipulation**: No `sys.path` or `os.chdir()` calls
- **Dynamic layer indices**: Calculated from `hidden_dim` configuration
- **Safe config access**: All config reads use `.get()` with defaults
- **Absolute paths**: All file paths resolved from project root

## Validation

The package includes config validation:

```python
from decoder.config import validate_config

validate_config(my_config)  # Raises ValueError if invalid
```

## Tools & Utilities

### Experiment Results Analyzer 📊

Analyze and compare experiment results easily:

```bash
# List all available experiments
python decoder/analyze_results.py

# Show statistics for a specific experiment
python decoder/analyze_results.py --stats input-pixels

# Compare two configurations
python decoder/analyze_results.py --compare dropout no-dropout input-pixels

# Export summary to JSON
python decoder/analyze_results.py --export
```

**Features:**
- Works with or without pandas (stdlib fallback)
- Quick statistics for all numeric columns
- Side-by-side configuration comparisons
- JSON export for programmatic access
- Beautiful terminal output with emojis

### Setup Verification

Verify your decoder installation:

```bash
python decoder/verify_setup.py
```

This checks:
- All imports work correctly
- Directory structure is valid
- Model registry is properly configured
- Config validation works
- Helpful error messages if something's wrong

## Recent Improvements

- ✅ Fixed all relative imports to absolute
- ✅ Removed `sys.path` manipulation
- ✅ Removed `os.chdir()` calls
- ✅ Added `get_underlying_path()` utility
- ✅ Made layer indices dynamic based on architecture
- ✅ Added safe config access throughout
- ✅ Added validation and verification tools
- ✅ Added experiment results analyzer
- ✅ Enhanced error messages and debugging tools

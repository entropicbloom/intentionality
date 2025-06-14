# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run Commands
- Run an experiment: `python -m decoder.main`
- Run main class ID experiments: `python -m decoder.experiments.run_main_experiments_classid`
- Run input pixel experiments: `python -m decoder.experiments.run_main_experiments_inputpixels`
- Install dependencies: `pip install -r requirements.txt`

## Code Style Guidelines
- Imports: group standard library, third-party, and local imports
- Use absolute imports from project root (e.g. `from decoder.config import config`)
- Variable naming: snake_case for variables and functions
- Type hints: not consistently used but recommended for new code
- Error handling: use explicit try/except blocks
- Config management: use dictionary-based configuration
- Models follow PyTorch Lightning patterns
- Use seed parameters for reproducibility
- PEP 8 compliant formatting
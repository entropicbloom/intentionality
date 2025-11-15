#!/usr/bin/env python3
"""
Quick sanity check to verify decoder setup is correct.

Run this after installation or when debugging import/path issues.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def verify_setup():
    """Verify that the decoder package is set up correctly."""
    print("🔍 Verifying decoder setup...\n")

    # Check 1: Basic imports
    print("✓ Checking imports...")
    try:
        from decoder.config import PROJECT_ROOT, UNDERLYING_DIR, get_underlying_path, validate_config
        print("  ✓ Config module imported")
    except ImportError as e:
        print(f"  ✗ Config import failed: {e}")
        return False

    try:
        from decoder.models import decoder_dict
        print("  ✓ Models module imported")
    except ImportError as e:
        print(f"  ⚠ Models import failed (likely missing torch): {e}")
        print("    This is OK if dependencies aren't installed yet")
        decoder_dict = None

    try:
        from decoder.decoder_models import FCDecoder, TransformerDecoder
        print("  ✓ Decoder models imported")
    except ImportError as e:
        print(f"  ⚠ Decoder models import failed: {e}")
        print("    Install dependencies: pip install -r requirements.txt")

    # Check 2: Directory structure
    print("\n✓ Checking directory structure...")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Underlying dir: {UNDERLYING_DIR}")

    if not PROJECT_ROOT.exists():
        print(f"  ✗ Project root doesn't exist: {PROJECT_ROOT}")
        return False
    print("  ✓ Project root exists")

    if not UNDERLYING_DIR.exists():
        print(f"  ✗ Underlying directory doesn't exist: {UNDERLYING_DIR}")
        print("    Create it with: mkdir underlying")
        return False
    print("  ✓ Underlying directory exists")

    # Check 3: Model dictionary
    print("\n✓ Checking model registry...")
    if decoder_dict is not None:
        expected_models = ['FCDecoder', 'TransformerDecoder']
        for model_name in expected_models:
            if model_name in decoder_dict:
                print(f"  ✓ {model_name} registered")
            else:
                print(f"  ✗ {model_name} not found in decoder_dict")
                return False
    else:
        print("  ⚠ Skipped (models not imported due to missing dependencies)")

    # Check 4: Config validation
    print("\n✓ Checking config validation...")
    try:
        from decoder.config import config
        validate_config(config)
        print("  ✓ Default config is valid")
    except Exception as e:
        print(f"  ✗ Config validation failed: {e}")
        return False

    # Check 5: Path utility
    print("\n✓ Checking path utilities...")
    try:
        test_path = get_underlying_path("test")
        expected = str(UNDERLYING_DIR / "test")
        if test_path == expected:
            print(f"  ✓ get_underlying_path() working correctly")
        else:
            print(f"  ✗ Path mismatch: {test_path} != {expected}")
            return False
    except Exception as e:
        print(f"  ✗ Path utility failed: {e}")
        return False

    print("\n" + "="*50)
    print("✅ All checks passed! Decoder setup looks good.")
    print("="*50)
    return True


if __name__ == "__main__":
    import sys
    success = verify_setup()
    sys.exit(0 if success else 1)

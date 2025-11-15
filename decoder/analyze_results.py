#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════╗
║          Decoder Experiment Results Analyzer 📊               ║
║                                                               ║
║  Turn boring CSV files into beautiful insights!               ║
║  No pandas? No problem! Works with stdlib too.                ║
╚═══════════════════════════════════════════════════════════════╝

A fun utility to analyze and compare decoder experiment results.

Usage:
    python decoder/analyze_results.py                    # List all experiments
    python decoder/analyze_results.py --stats input-pixels
    python decoder/analyze_results.py --compare dropout no-dropout input-pixels
    python decoder/analyze_results.py --export           # Export JSON summary
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Add project root for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ResultsAnalyzer:
    """Analyze and compare decoder experiment results."""

    def __init__(self, data_dir: Path = None):
        """
        Initialize the analyzer.

        Args:
            data_dir: Path to data directory (defaults to project_root/data)
        """
        if data_dir is None:
            data_dir = project_root / "data"
        self.data_dir = data_dir
        self.results = {}

    def discover_experiments(self) -> Dict[str, List[Path]]:
        """
        Discover all available experiment results.

        Returns:
            Dictionary mapping experiment types to list of result files
        """
        experiments = {}

        if not self.data_dir.exists():
            print(f"⚠️  Data directory not found: {self.data_dir}")
            return experiments

        # Scan for experiment directories
        for exp_dir in self.data_dir.iterdir():
            if exp_dir.is_dir():
                csv_files = list(exp_dir.glob("*.csv"))
                if csv_files:
                    experiments[exp_dir.name] = csv_files

        return experiments

    def list_experiments(self):
        """Print a nice summary of available experiments."""
        experiments = self.discover_experiments()

        if not experiments:
            print("📭 No experiment results found.")
            print(f"   Expected location: {self.data_dir}")
            return

        print("📊 Available Experiments:\n")
        print("=" * 60)

        for exp_type, files in sorted(experiments.items()):
            print(f"\n🔬 {exp_type.upper()}")
            print(f"   Location: {self.data_dir / exp_type}")
            print(f"   Results: {len(files)} files\n")

            for f in sorted(files):
                # Try to extract meaningful info from filename
                name = f.stem
                size = f.stat().st_size
                print(f"   • {name:<40} ({size:>8,} bytes)")

        print("\n" + "=" * 60)
        print(f"Total: {sum(len(f) for f in experiments.values())} result files")

    def load_csv_safe(self, filepath: Path) -> Optional[object]:
        """
        Safely load a CSV file with helpful error messages.
        Falls back to standard library csv if pandas not available.

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame-like object or None if loading fails
        """
        try:
            import pandas as pd
            return pd.read_csv(filepath)
        except ImportError:
            # Fallback to standard library
            return self._load_csv_stdlib(filepath)
        except Exception as e:
            print(f"⚠️  Failed to load {filepath.name}: {e}")
            return None

    def _load_csv_stdlib(self, filepath: Path):
        """Load CSV using standard library (fallback when pandas not available)."""
        import csv

        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                data = list(reader)

            if not data:
                return None

            # Create a simple DataFrame-like object
            class SimpleDataFrame:
                def __init__(self, data):
                    self.data = data
                    self.columns = list(data[0].keys()) if data else []

                def __len__(self):
                    return len(self.data)

                def select_dtypes(self, include=None):
                    """Mimic pandas select_dtypes for numeric columns."""
                    numeric_cols = []
                    if self.data:
                        for col in self.columns:
                            try:
                                float(self.data[0][col])
                                numeric_cols.append(col)
                            except (ValueError, TypeError):
                                pass

                    class NumericColumns:
                        def __init__(self, cols):
                            self.columns = cols

                    return NumericColumns(numeric_cols)

                def __getitem__(self, col):
                    """Get column data."""
                    values = [row[col] for row in self.data]

                    class Column:
                        def __init__(self, values, name):
                            self.values = values
                            self.name = name
                            # Try to convert to numeric
                            try:
                                self.numeric_values = [float(v) for v in values if v]
                            except (ValueError, TypeError):
                                self.numeric_values = None

                        def mean(self):
                            if self.numeric_values:
                                return sum(self.numeric_values) / len(self.numeric_values)
                            return 0

                        def std(self):
                            if not self.numeric_values:
                                return 0
                            mean_val = self.mean()
                            variance = sum((x - mean_val) ** 2 for x in self.numeric_values) / len(self.numeric_values)
                            return variance ** 0.5

                        def min(self):
                            return min(self.numeric_values) if self.numeric_values else 0

                        def max(self):
                            return max(self.numeric_values) if self.numeric_values else 0

                    return Column(values, col)

            return SimpleDataFrame(data)

        except Exception as e:
            print(f"⚠️  Failed to load {filepath.name}: {e}")
            return None

    def quick_stats(self, experiment_type: str):
        """
        Show quick statistics for an experiment type.

        Args:
            experiment_type: Type of experiment (e.g., 'input-pixels')
        """
        experiments = self.discover_experiments()

        if experiment_type not in experiments:
            print(f"❌ Experiment type '{experiment_type}' not found.")
            print(f"   Available: {', '.join(experiments.keys())}")
            return

        print(f"\n📈 Quick Stats: {experiment_type}\n")
        print("=" * 60)

        files = experiments[experiment_type]

        for filepath in sorted(files):
            df = self.load_csv_safe(filepath)
            if df is None:
                continue

            print(f"\n{filepath.stem}")
            print("-" * 60)
            print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")

            # Show numeric column stats
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                print("\nNumeric Columns:")
                for col in numeric_cols:
                    mean = df[col].mean()
                    std = df[col].std()
                    min_val = df[col].min()
                    max_val = df[col].max()
                    print(f"  {col:>20}: {mean:>8.4f} ± {std:.4f}  "
                          f"[{min_val:.4f}, {max_val:.4f}]")

        print("\n" + "=" * 60)

    def compare_configs(self, config1: str, config2: str, experiment_type: str):
        """
        Compare two experimental configurations.

        Args:
            config1: First configuration name
            config2: Second configuration name
            experiment_type: Type of experiment
        """
        experiments = self.discover_experiments()

        if experiment_type not in experiments:
            print(f"❌ Experiment type '{experiment_type}' not found.")
            return

        # Find matching files
        files = experiments[experiment_type]
        file1 = next((f for f in files if config1 in f.stem), None)
        file2 = next((f for f in files if config2 in f.stem), None)

        if not file1 or not file2:
            print(f"❌ Could not find both configurations:")
            print(f"   Looking for: '{config1}' and '{config2}'")
            print(f"   Available files:")
            for f in files:
                print(f"   • {f.stem}")
            return

        df1 = self.load_csv_safe(file1)
        df2 = self.load_csv_safe(file2)

        if df1 is None or df2 is None:
            return

        print(f"\n🔬 Comparison: {config1} vs {config2}\n")
        print("=" * 60)

        # Compare numeric columns
        numeric_cols = set(df1.select_dtypes(include=['number']).columns) & \
                       set(df2.select_dtypes(include=['number']).columns)

        for col in sorted(numeric_cols):
            mean1 = df1[col].mean()
            mean2 = df2[col].mean()
            diff = mean2 - mean1
            pct_change = (diff / mean1 * 100) if mean1 != 0 else float('inf')

            symbol = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"

            print(f"\n{col}")
            print(f"  {config1:>20}: {mean1:.6f}")
            print(f"  {config2:>20}: {mean2:.6f}")
            print(f"  {symbol} Difference: {diff:+.6f} ({pct_change:+.2f}%)")

        print("\n" + "=" * 60)

    def export_summary(self, output_file: str = "experiment_summary.json"):
        """
        Export a JSON summary of all experiments.

        Args:
            output_file: Path to output JSON file
        """
        experiments = self.discover_experiments()
        summary = {}

        for exp_type, files in experiments.items():
            summary[exp_type] = {}

            for filepath in files:
                df = self.load_csv_safe(filepath)
                if df is None:
                    continue

                summary[exp_type][filepath.stem] = {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "numeric_stats": {}
                }

                # Add stats for numeric columns
                for col in df.select_dtypes(include=['number']).columns:
                    summary[exp_type][filepath.stem]["numeric_stats"][col] = {
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std()),
                        "min": float(df[col].min()),
                        "max": float(df[col].max())
                    }

        # Write to file
        output_path = project_root / output_file
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✅ Summary exported to: {output_path}")


def main():
    """Main entry point for the analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze decoder experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments
  python decoder/analyze_results.py --list

  # Show stats for input-pixels experiments
  python decoder/analyze_results.py --stats input-pixels

  # Compare dropout vs no-dropout
  python decoder/analyze_results.py --compare dropout no-dropout input-pixels

  # Export summary
  python decoder/analyze_results.py --export
        """
    )

    parser.add_argument('--list', action='store_true',
                        help='List all available experiments')
    parser.add_argument('--stats', metavar='EXPERIMENT',
                        help='Show quick statistics for an experiment type')
    parser.add_argument('--compare', nargs=3, metavar=('CONFIG1', 'CONFIG2', 'EXPERIMENT'),
                        help='Compare two configurations within an experiment')
    parser.add_argument('--export', action='store_true',
                        help='Export summary to JSON')

    args = parser.parse_args()

    analyzer = ResultsAnalyzer()

    # If no arguments, show list by default
    if len(sys.argv) == 1:
        analyzer.list_experiments()
        return

    if args.list:
        analyzer.list_experiments()

    if args.stats:
        analyzer.quick_stats(args.stats)

    if args.compare:
        config1, config2, exp_type = args.compare
        analyzer.compare_configs(config1, config2, exp_type)

    if args.export:
        analyzer.export_summary()


if __name__ == "__main__":
    main()

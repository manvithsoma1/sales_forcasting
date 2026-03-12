"""
model_migration.py — Keras Model Migration Utility
====================================================
Scans the models/ directory for old .h5 files saved with TF ≤ 2.15 and
re-saves them in the modern .keras format compatible with TF 2.16+.

Usage (from project root):
    python src/model_migration.py
    python src/model_migration.py --models-dir models/
    python src/model_migration.py --file models/lstm_grocery_v1.h5

This resolves the error:
    "Unrecognized keyword arguments: ['batch_shape']"
"""

import os
import sys
import glob
import argparse

# Allow running from project root or from src/
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def migrate_file(h5_path: str, overwrite: bool = False) -> str:
    """
    Load a single .h5 model with the compatibility patch and re-save as .keras.

    Parameters
    ----------
    h5_path : str  Path to the .h5 model file.
    overwrite : bool  If True, overwrites an existing .keras file.

    Returns
    -------
    str  Path of the newly saved .keras file, or empty string on failure.
    """
    from src.model import robust_load_keras_model

    keras_path = os.path.splitext(h5_path)[0] + ".keras"

    if os.path.exists(keras_path) and not overwrite:
        print(f"  ⏭️  Skipping (already migrated): {keras_path}")
        return keras_path

    print(f"  🔄 Migrating: {h5_path}")
    try:
        model = robust_load_keras_model(h5_path)
        model.save(keras_path)
        print(f"  ✅ Saved: {keras_path}")
        return keras_path
    except Exception as e:
        print(f"  ❌ Failed to migrate {h5_path}: {e}")
        return ""


def migrate_all(models_dir: str = "models", overwrite: bool = False) -> dict:
    """
    Scan a directory (recursively) for .h5 files and migrate each one.

    Parameters
    ----------
    models_dir : str  Directory to scan.
    overwrite : bool  Overwrite existing .keras files.

    Returns
    -------
    dict  {'success': [...], 'failed': [...], 'skipped': [...]}
    """
    pattern = os.path.join(models_dir, "**", "*.h5")
    h5_files = glob.glob(pattern, recursive=True)

    if not h5_files:
        print(f"ℹ️  No .h5 model files found in '{models_dir}'. Nothing to migrate.")
        return {"success": [], "failed": [], "skipped": []}

    print(f"Found {len(h5_files)} .h5 file(s) to process:\n")

    results = {"success": [], "failed": [], "skipped": []}

    for h5_path in h5_files:
        keras_path = os.path.splitext(h5_path)[0] + ".keras"
        if os.path.exists(keras_path) and not overwrite:
            print(f"  ⏭️  Skipping (already migrated): {keras_path}")
            results["skipped"].append(h5_path)
            continue

        out = migrate_file(h5_path, overwrite=overwrite)
        if out:
            results["success"].append(out)
        else:
            results["failed"].append(h5_path)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"  ✅ Migrated : {len(results['success'])}")
    print(f"  ⏭️  Skipped  : {len(results['skipped'])}")
    print(f"  ❌ Failed   : {len(results['failed'])}")
    if results["failed"]:
        print("\nFailed files:")
        for f in results["failed"]:
            print(f"    {f}")
    print("=" * 60)

    return results


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Migrate Keras .h5 models to .keras format (fixes batch_shape error)."
    )
    parser.add_argument(
        "--models-dir", "-d",
        default="models",
        help="Directory to scan for .h5 files (default: models/)"
    )
    parser.add_argument(
        "--file", "-f",
        default=None,
        help="Migrate a single specific .h5 file instead of scanning a directory."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-migrate even if .keras file already exists."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🔧 Keras Model Migration Utility — batch_shape → shape fix")
    print("=" * 60 + "\n")

    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            sys.exit(1)
        migrate_file(args.file, overwrite=args.overwrite)
    else:
        migrate_all(models_dir=args.models_dir, overwrite=args.overwrite)

    print("\nDone. You can now load .keras files without compatibility errors.")
    print("Update your app to load .keras files instead of .h5 files.")


if __name__ == "__main__":
    main()

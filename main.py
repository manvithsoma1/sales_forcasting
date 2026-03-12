"""
main.py — v2.0
Multi-model training with ensemble.
Usage:
    python main.py                    # Train all models (LightGBM + LSTM + Ensemble)
    python main.py --model lstm       # Train LSTM only
    python main.py --model lgbm       # Train LightGBM only
    python main.py --store 1 --family "GROCERY I"  # Train on specific store/family
"""

import argparse
from src.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description='Store Sales Forecasting — Model Training')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'lstm', 'lgbm'],
                        help='Which model to train')
    parser.add_argument('--store', type=int, default=None,
                        help='Filter to specific store number')
    parser.add_argument('--family', type=str, default=None,
                        help='Filter to specific product family')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Store Sales Forecasting — Training Pipeline v2.0")
    print("=" * 60)

    pipeline = Pipeline(config_path=args.config)

    train_lstm = args.model in ('all', 'lstm')
    train_lgbm = args.model in ('all', 'lgbm')

    results = pipeline.run(
        store_nbr=args.store,
        family=args.family,
        train_lstm=train_lstm,
        train_lgbm=train_lgbm,
    )

    # Summary
    print("\n" + "=" * 60)
    print("📊 TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Data shape:    {results.get('data_shape', 'N/A')}")
    print(f"  Feature shape: {results.get('feature_shape', 'N/A')}")
    print(f"  Model version: v{results.get('version', '?')}")
    print()

    for model_name, metrics in results.get('metrics', {}).items():
        print(f"  [{model_name.upper()}]")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
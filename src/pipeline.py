"""
Pipeline Module — v2.0
End-to-end automated pipeline: data → features → train → evaluate → serve.
"""

import os
import time
import yaml
import joblib
import numpy as np
import pandas as pd

from .data_loader import DataLoader
from .features import FeatureEngineer
from .preprocessing import Preprocessor
from .model import AttentionLSTMModel, LightGBMModel, EnsembleModel, ModelRegistry
from .evaluation import Evaluator
from .anomaly_detection import AnomalyDetector


class Pipeline:
    """
    Orchestrates the full ML pipeline.
    Can be run daily for fresh predictions.
    """

    def __init__(self, config_path='config/config.yaml'):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.loader = DataLoader(config_path)
        self.engineer = FeatureEngineer()
        self.preprocessor = Preprocessor(config_path)
        self.evaluator = Evaluator()
        self.registry = ModelRegistry(config_path)

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    def run(self, store_nbr=None, family=None, train_lstm=True, train_lgbm=True,
            progress_callback=None):
        """
        Run the full pipeline.

        Parameters
        ----------
        store_nbr : optional store filter (None = all stores)
        family : optional family filter (None = all families)
        train_lstm : whether to train LSTM
        train_lgbm : whether to train LightGBM
        progress_callback : callable(step, total, message) for progress updates

        Returns
        -------
        dict with trained models, metrics, and predictions
        """
        total_steps = 6
        step = 0

        def progress(msg):
            nonlocal step
            step += 1
            print(f"\n{'='*60}")
            print(f"[{step}/{total_steps}] {msg}")
            print(f"{'='*60}")
            if progress_callback:
                progress_callback(step, total_steps, msg)

        t0 = time.time()
        results = {}

        # 1. Data Ingestion
        progress("Loading & merging data...")
        raw = self.loader.load_raw_data()
        df = self.loader.merge_data(raw)
        holidays_raw = self.loader.get_holidays_raw()

        if store_nbr or family:
            df = self.loader.filter_subset(df, store_nbr, family)
            print(f"  Filtered to {len(df)} rows")

        results['data_shape'] = df.shape

        # 2. Feature Engineering
        progress("Engineering features...")
        df_feat = self.engineer.create_features(df, holidays_df=holidays_raw, include_lags=True)

        # Drop rows with NaN from lags
        initial_len = len(df_feat)
        df_feat = df_feat.dropna(subset=[c for c in df_feat.columns if 'lag' in c or 'roll' in c])
        print(f"  Dropped {initial_len - len(df_feat)} rows with NaN lags")
        results['feature_shape'] = df_feat.shape

        # 3. Train LightGBM
        metrics = {}
        models = {}

        if train_lgbm:
            progress("Training LightGBM...")
            lgbm = LightGBMModel(self.config_path)
            feature_cols = self.engineer.get_feature_columns(mode='lgbm')

            if family:
                # Single family mode
                X, y = self.preprocessor.prepare_lgbm_data(df_feat, feature_cols)
                split = int(len(X) * 0.85)
                lgbm.train(X.iloc[:split], y.iloc[:split],
                          X.iloc[split:], y.iloc[split:],
                          family_name=family or 'global')
            else:
                lgbm.train_per_family(df_feat, feature_cols)

            models['lgbm'] = lgbm

            # Evaluate
            test_X, test_y = self.preprocessor.prepare_lgbm_data(df_feat.tail(1000), feature_cols)
            first_model_name = list(lgbm.models.keys())[0] if lgbm.models else 'global'
            try:
                pred = lgbm.predict(test_X, family_name=first_model_name)
                metrics['lgbm'] = self.evaluator.calculate_metrics(test_y.values, pred, label='LightGBM')
            except Exception as e:
                print(f"  LightGBM eval failed: {e}")
                metrics['lgbm'] = {}

        # 4. Train LSTM
        if train_lstm:
            progress("Training Attention LSTM...")
            lstm_features = self.engineer.get_feature_columns(mode='lstm')
            lstm_cols = ['sales'] + [c for c in lstm_features if c in df_feat.columns]
            df_lstm = df_feat[lstm_cols].copy()

            df_scaled, scaler = self.preprocessor.scale_data(df_lstm)
            X_seq, y_seq = self.preprocessor.create_sequences(df_scaled.values)

            if len(X_seq) > 0:
                input_shape = (X_seq.shape[1], X_seq.shape[2])
                lstm = AttentionLSTMModel(input_shape, self.config_path)
                history = lstm.train(X_seq, y_seq)
                models['lstm'] = lstm
                results['lstm_scaler'] = scaler

                # Evaluate
                y_pred = lstm.predict(X_seq)
                metrics['lstm'] = self.evaluator.calculate_metrics(y_seq, y_pred, label='LSTM')
                self.evaluator.plot_loss(history)

                joblib.dump(scaler, os.path.join(self.registry.base_path, 'scaler.pkl'))

        # 5. Build Ensemble (if both models trained)
        if 'lstm' in models and 'lgbm' in models:
            progress("Building ensemble...")
            try:
                ensemble = EnsembleModel(self.config_path)
                # Simple: use tail predictions
                preds = {}
                test_X, test_y = self.preprocessor.prepare_lgbm_data(
                    df_feat.tail(500), self.engineer.get_feature_columns(mode='lgbm')
                )
                first_model_name = list(models['lgbm'].models.keys())[0]
                preds['lgbm'] = models['lgbm'].predict(test_X, family_name=first_model_name)

                # For LSTM, use the last chunk
                lstm_preds = models['lstm'].predict(X_seq[-len(test_y):])
                preds['lstm'] = lstm_preds[:len(test_y)]

                min_len = min(len(v) for v in preds.values())
                preds = {k: v[:min_len] for k, v in preds.items()}
                y_ens = test_y.values[:min_len]

                ensemble.train(preds, y_ens)
                models['ensemble'] = ensemble

                ens_pred = ensemble.predict(preds)
                metrics['ensemble'] = self.evaluator.calculate_metrics(y_ens, ens_pred, label='Ensemble')
            except Exception as e:
                print(f"  Ensemble build failed: {e}")
        else:
            progress("Skipping ensemble (need both LSTM + LightGBM)...")

        # 6. Save
        progress("Saving models & artifacts...")
        flat_metrics = {}
        for model_name, m in metrics.items():
            for k, v in m.items():
                flat_metrics[f'{model_name}_{k}'] = v

        version = self.registry.save_version(models, flat_metrics,
                                             feature_cols=self.engineer.get_feature_columns())

        elapsed = time.time() - t0
        print(f"\n🏁 Pipeline complete in {elapsed:.1f}s — version v{version}")

        results['metrics'] = metrics
        results['models'] = models
        results['version'] = version
        return results

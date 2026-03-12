"""
Evaluation Module — v2.0
RMSLE, SHAP explainability, residual analysis, backtesting, per-family metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Evaluator:
    def __init__(self):
        os.makedirs('plots', exist_ok=True)

    # ------------------------------------------------------------------
    # Core Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def rmsle(y_true, y_pred):
        """Root Mean Squared Logarithmic Error — Kaggle competition metric."""
        y_true = np.maximum(y_true, 0)
        y_pred = np.maximum(y_pred, 0)
        return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

    def calculate_metrics(self, y_true, y_pred, label=''):
        """Calculate and print RMSE, MAE, RMSLE."""
        r = self.rmse(y_true, y_pred)
        m = self.mae(y_true, y_pred)
        rl = self.rmsle(y_true, y_pred)

        prefix = f"[{label}] " if label else ""
        print(f"\n--- {prefix}Evaluation Metrics ---")
        print(f"  RMSE:  {r:.4f}")
        print(f"  MAE:   {m:.4f}")
        print(f"  RMSLE: {rl:.4f}")
        return {'rmse': r, 'mae': m, 'rmsle': rl}

    def per_family_metrics(self, df, pred_col='predicted', actual_col='sales'):
        """
        Calculate metrics per product family.

        Parameters
        ----------
        df : DataFrame with 'family', actual_col, pred_col columns
        """
        if 'family' not in df.columns:
            return pd.DataFrame()

        results = []
        for fam in df['family'].unique():
            sub = df[df['family'] == fam]
            if len(sub) == 0:
                continue
            results.append({
                'family': fam,
                'count': len(sub),
                'rmse': self.rmse(sub[actual_col], sub[pred_col]),
                'mae': self.mae(sub[actual_col], sub[pred_col]),
                'rmsle': self.rmsle(sub[actual_col], sub[pred_col]),
            })

        return pd.DataFrame(results).sort_values('rmsle')

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_loss(self, history, save_path='plots/training_loss.png'):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss', color='#00D4AA')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss', color='#FF6B6B')
        plt.title('Model Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Loss plot saved to {save_path}")

    def plot_predictions(self, y_true, y_pred, start_idx=0, length=100,
                         save_path='plots/prediction_sample.png'):
        plt.figure(figsize=(14, 6))
        end = start_idx + length
        plt.plot(y_true[start_idx:end], label='Actual', color='#00D4AA', linewidth=2)
        plt.plot(y_pred[start_idx:end], label='Predicted', color='#FF6B6B',
                 linestyle='--', linewidth=2)
        plt.title('Actual vs Predicted Sales')
        plt.xlabel('Time Steps')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Prediction plot saved to {save_path}")

    def plot_residuals(self, y_true, y_pred, save_path='plots/residual_analysis.png'):
        """Residual analysis: scatter + distribution."""
        residuals = np.array(y_true) - np.array(y_pred)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color='#00D4AA')
        axes[0].axhline(y=0, color='#FF6B6B', linestyle='--')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Residual')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)

        # 2. Residual histogram
        axes[1].hist(residuals, bins=50, color='#00D4AA', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Residual')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        axes[1].grid(True, alpha=0.3)

        # 3. QQ plot
        try:
            from scipy import stats
            stats.probplot(residuals, plot=axes[2])
            axes[2].set_title('Q-Q Plot')
        except ImportError:
            axes[2].text(0.5, 0.5, 'scipy not available', ha='center', va='center')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Residual analysis saved to {save_path}")

    # ------------------------------------------------------------------
    # SHAP Explainability
    # ------------------------------------------------------------------

    def shap_importance(self, model, X, feature_names=None, max_display=20,
                        save_path='plots/shap_importance.png'):
        """
        SHAP feature importance for tree-based models (LightGBM).

        Parameters
        ----------
        model : trained LightGBM model (or any tree model)
        X : feature matrix (DataFrame or ndarray)
        """
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values, X,
                feature_names=feature_names,
                max_display=max_display,
                show=False
            )
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"SHAP importance saved to {save_path}")
            return shap_values
        except ImportError:
            print("⚠️ shap not installed — skipping SHAP analysis")
            return None

    def shap_waterfall(self, model, X_single, feature_names=None,
                       save_path='plots/shap_waterfall.png'):
        """SHAP waterfall chart for a single prediction explanation."""
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            sv = explainer(X_single)

            plt.figure(figsize=(12, 8))
            shap.waterfall_plot(sv[0], show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"SHAP waterfall saved to {save_path}")
            return sv
        except ImportError:
            print("⚠️ shap not installed — skipping waterfall")
            return None

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    def backtest(self, df, model_predict_fn, feature_cols, target_col='sales',
                 window_size=30, step_size=7):
        """
        Rolling-window backtesting.

        Parameters
        ----------
        df : DataFrame sorted by date
        model_predict_fn : callable(X) -> y_pred
        feature_cols : feature column list
        target_col : target column name
        window_size : training window size in days
        step_size : step between evaluation windows

        Returns
        -------
        DataFrame with per-window metrics
        """
        results = []
        available = [c for c in feature_cols if c in df.columns]

        for start in range(0, len(df) - window_size - step_size, step_size):
            val_start = start + window_size
            val_end = min(val_start + step_size, len(df))

            X_val = df.iloc[val_start:val_end][available].fillna(0)
            y_val = df.iloc[val_start:val_end][target_col].values

            if len(X_val) == 0:
                continue

            try:
                y_pred = model_predict_fn(X_val)
                results.append({
                    'window_start': val_start,
                    'window_end': val_end,
                    'n_samples': len(y_val),
                    'rmse': self.rmse(y_val, y_pred),
                    'mae': self.mae(y_val, y_pred),
                    'rmsle': self.rmsle(y_val, y_pred),
                })
            except Exception as e:
                print(f"  Backtest window {start} failed: {e}")

        bt = pd.DataFrame(results)
        if len(bt) > 0:
            print(f"\nBacktest over {len(bt)} windows:")
            print(f"  Mean RMSLE: {bt['rmsle'].mean():.4f} ± {bt['rmsle'].std():.4f}")
        return bt
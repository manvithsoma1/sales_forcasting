"""
Anomaly Detection Module — v2.0
Detects sales anomalies using Isolation Forest.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import yaml


class AnomalyDetector:
    """
    Detects days where actual sales deviates significantly from predicted.
    Uses Isolation Forest + statistical thresholding.
    """

    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        ad_cfg = self.config.get('anomaly_detection', {})
        self.contamination = ad_cfg.get('contamination', 0.05)
        self.threshold_std = ad_cfg.get('threshold_std', 2.5)
        self.model = None

    def fit(self, df, features=None):
        """
        Fit Isolation Forest on historical data.

        Parameters
        ----------
        df : DataFrame with sales and feature columns
        features : list of feature columns. If None, uses ['sales', 'transactions']
        """
        if features is None:
            features = [c for c in ['sales', 'transactions', 'onpromotion'] if c in df.columns]
            if not features:
                features = ['sales']

        X = df[features].fillna(0).values
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X)
        self.features = features
        print(f"Anomaly detector fitted on {len(X)} samples, features: {features}")

    def detect(self, df):
        """
        Detect anomalies in the data.

        Returns
        -------
        DataFrame with 'is_anomaly' column (-1 = anomaly, 1 = normal)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = df[self.features].fillna(0).values
        labels = self.model.predict(X)
        scores = self.model.decision_function(X)

        result = df.copy()
        result['anomaly_label'] = labels
        result['anomaly_score'] = scores
        result['is_anomaly'] = (labels == -1).astype(int)

        n_anomalies = result['is_anomaly'].sum()
        print(f"  Detected {n_anomalies} anomalies out of {len(df)} days ({n_anomalies/len(df)*100:.1f}%)")
        return result

    def detect_residual_anomalies(self, y_true, y_pred, dates=None):
        """
        Flag days where actual significantly deviates from predicted.

        Parameters
        ----------
        y_true : array of actual sales
        y_pred : array of predicted sales
        dates : optional array of dates

        Returns
        -------
        DataFrame with anomaly flags
        """
        residuals = np.array(y_true) - np.array(y_pred)
        mean_r = np.mean(residuals)
        std_r = np.std(residuals)

        threshold_upper = mean_r + self.threshold_std * std_r
        threshold_lower = mean_r - self.threshold_std * std_r

        is_anomaly = (residuals > threshold_upper) | (residuals < threshold_lower)

        result = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred,
            'residual': residuals,
            'is_anomaly': is_anomaly.astype(int),
            'anomaly_type': np.where(
                residuals > threshold_upper, 'demand_spike',
                np.where(residuals < threshold_lower, 'unexpected_drop', 'normal')
            )
        })
        if dates is not None:
            result['date'] = dates

        n = result['is_anomaly'].sum()
        print(f"  Residual anomalies: {n} days flagged")
        return result

    def get_alerts(self, anomaly_df, top_n=5):
        """
        Generate human-readable alerts for recent anomalies.

        Parameters
        ----------
        anomaly_df : output from detect() or detect_residual_anomalies()
        top_n : number of alerts to return

        Returns
        -------
        list of alert dictionaries
        """
        anomalies = anomaly_df[anomaly_df['is_anomaly'] == 1]
        if len(anomalies) == 0:
            return [{'level': 'info', 'message': '✅ No anomalies detected — all sales within expected range.'}]

        alerts = []
        for _, row in anomalies.tail(top_n).iterrows():
            atype = row.get('anomaly_type', 'unknown')
            date_str = row['date'].strftime('%Y-%m-%d') if 'date' in row and hasattr(row['date'], 'strftime') else 'recent'

            if atype == 'demand_spike':
                alerts.append({
                    'level': 'warning',
                    'message': f"🔴 {date_str}: Demand spike — actual {row['actual']:.0f} vs predicted {row['predicted']:.0f} (+{row['residual']:.0f} units)"
                })
            elif atype == 'unexpected_drop':
                alerts.append({
                    'level': 'error',
                    'message': f"🟡 {date_str}: Unexpected drop — actual {row['actual']:.0f} vs predicted {row['predicted']:.0f} ({row['residual']:.0f} units)"
                })
            else:
                alerts.append({
                    'level': 'warning',
                    'message': f"⚠️ {date_str}: Anomalous sales pattern detected (score: {row.get('anomaly_score', 'N/A')})"
                })

        return alerts

"""
Model Module — v2.0
AttentionLSTM, LightGBM per-family, optional TFT, Ridge Ensemble, and ModelRegistry.
"""

import os
import json
import numpy as np
import joblib
import yaml
from datetime import datetime

# ======================================================================
# TensorFlow / Keras (LSTM)
# ======================================================================
import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout,
    BatchNormalization, Layer, Permute, Multiply, Flatten,
    RepeatVector, Lambda
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ======================================================================
# LightGBM
# ======================================================================
import lightgbm as lgb

# ======================================================================
# Ensemble
# ======================================================================
from sklearn.linear_model import Ridge


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ATTENTION LAYER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class AttentionLayer(Layer):
    """Simple self-attention for sequence outputs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight', shape=(input_shape[-1], 1),
            initializer='glorot_uniform', trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias', shape=(input_shape[1], 1),
            initializer='zeros', trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch, timesteps, features)
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

    def get_config(self):
        return super().get_config()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ATTENTION LSTM MODEL
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class AttentionLSTMModel:
    """
    Bidirectional LSTM with self-attention mechanism.
    Uses Huber loss for robustness to outliers.
    """

    def __init__(self, input_shape, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.input_shape = input_shape
        lstm_cfg = self.config['model']['lstm']
        self.units = lstm_cfg.get('units', [128, 64])
        self.dropout = lstm_cfg.get('dropout', 0.3)
        self.use_attention = lstm_cfg.get('attention', True)
        self.use_bidirectional = lstm_cfg.get('bidirectional', True)
        self.lr = lstm_cfg.get('learning_rate', 0.001)
        self.loss = lstm_cfg.get('loss', 'huber')

        self.model = self._build_model()

    def _build_model(self):
        print(f"Building Attention LSTM — input shape: {self.input_shape}")

        inputs = Input(shape=self.input_shape)
        x = inputs

        # Stacked BiLSTM layers
        for i, units in enumerate(self.units):
            return_seq = True  # Always return sequences (attention needs full output)
            if self.use_bidirectional:
                x = Bidirectional(LSTM(units, return_sequences=return_seq))(x)
            else:
                x = LSTM(units, return_sequences=return_seq)(x)
            x = Dropout(self.dropout)(x)
            x = BatchNormalization()(x)

        # Attention
        if self.use_attention:
            x = AttentionLayer()(x)
        else:
            x = Lambda(lambda t: t[:, -1, :])(x)  # Take last timestep

        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)
        outputs = Dense(1)(x)

        model = KerasModel(inputs=inputs, outputs=outputs)

        loss_fn = tf.keras.losses.Huber() if self.loss == 'huber' else 'mse'
        model.compile(
            optimizer=Adam(learning_rate=self.lr, clipnorm=1.0),
            loss=loss_fn,
            metrics=['mae']
        )
        return model

    def train(self, X_train, y_train, epochs=None, batch_size=None):
        e = epochs or self.config['model']['epochs']
        b = batch_size or self.config['model']['batch_size']
        val_split = self.config['model'].get('validation_split', 0.15)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
        ]

        print(f"Training Attention LSTM for up to {e} epochs...")
        history = self.model.fit(
            X_train, y_train,
            epochs=e, batch_size=b,
            validation_split=val_split,
            callbacks=callbacks, verbose=1
        )
        return history

    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"LSTM model saved to {path}")

    def load(self, path):
        self.model = tf.keras.models.load_model(
            path, custom_objects={'AttentionLayer': AttentionLayer}
        )
        print(f"LSTM model loaded from {path}")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LIGHTGBM MODEL (per-family)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class LightGBMModel:
    """
    LightGBM gradient boosted trees.
    Can train one global model or per-family models.
    """

    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        lgb_cfg = self.config['model']['lightgbm']
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': lgb_cfg.get('n_estimators', 1000),
            'learning_rate': lgb_cfg.get('learning_rate', 0.05),
            'max_depth': lgb_cfg.get('max_depth', 8),
            'num_leaves': lgb_cfg.get('num_leaves', 63),
            'min_child_samples': lgb_cfg.get('min_child_samples', 20),
            'subsample': lgb_cfg.get('subsample', 0.8),
            'colsample_bytree': lgb_cfg.get('colsample_bytree', 0.8),
            'verbose': -1,
            'n_jobs': -1,
            'random_state': 42,
        }
        self.early_stopping = lgb_cfg.get('early_stopping_rounds', 50)
        self.models = {}  # family_name -> model

    def train(self, X_train, y_train, X_val=None, y_val=None, family_name='global'):
        """Train a single LightGBM model."""
        print(f"Training LightGBM [{family_name}] — {X_train.shape[0]} samples, {X_train.shape[1]} features...")

        callbacks = [lgb.early_stopping(self.early_stopping, verbose=False), lgb.log_evaluation(period=0)]
        eval_set = [(X_val, y_val)] if X_val is not None else None

        model = lgb.LGBMRegressor(**self.params)
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )

        self.models[family_name] = model
        print(f"  → Best iteration: {model.best_iteration_}")
        return model

    def train_per_family(self, df, feature_cols, target_col='sales', val_ratio=0.15):
        """Train a separate LightGBM for each product family."""
        families = df['family'].unique() if 'family' in df.columns else ['global']
        print(f"Training LightGBM for {len(families)} families...")

        for fam in families:
            subset = df[df['family'] == fam] if 'family' in df.columns else df
            if len(subset) < 100:
                print(f"  ⚠️ Skipping {fam} — too few samples ({len(subset)})")
                continue

            available = [c for c in feature_cols if c in subset.columns]
            X = subset[available].fillna(0)
            y = subset[target_col].fillna(0)

            split_idx = int(len(X) * (1 - val_ratio))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

            self.train(X_train, y_train, X_val, y_val, family_name=fam)

        print(f"Trained {len(self.models)} family models.")

    def predict(self, X, family_name='global'):
        if family_name in self.models:
            return self.models[family_name].predict(X)
        elif 'global' in self.models:
            return self.models['global'].predict(X)
        else:
            raise ValueError(f"No model found for family '{family_name}'")

    def feature_importance(self, family_name='global'):
        model = self.models.get(family_name, self.models.get('global'))
        if model is None:
            return {}
        return dict(zip(model.feature_name_, model.feature_importances_))

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        for name, model in self.models.items():
            safe_name = name.replace(' ', '_').replace('/', '_')
            joblib.dump(model, os.path.join(path, f'lgbm_{safe_name}.pkl'))
        print(f"LightGBM models saved to {path}")

    def load(self, path):
        self.models = {}
        if not os.path.exists(path):
            raise FileNotFoundError(f"LightGBM model path not found: {path}")
        for fname in os.listdir(path):
            if fname.startswith('lgbm_') and fname.endswith('.pkl'):
                family = fname.replace('lgbm_', '').replace('.pkl', '').replace('_', ' ')
                self.models[family] = joblib.load(os.path.join(path, fname))
        print(f"Loaded {len(self.models)} LightGBM models from {path}")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ENSEMBLE MODEL
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class EnsembleModel:
    """
    Stacking ensemble with Ridge regression meta-learner.
    Combines out-of-fold predictions from LSTM + LightGBM (+ optional TFT).
    """

    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.meta_model = Ridge(alpha=1.0)
        self.weights = None
        self.model_names = []

    def train(self, predictions_dict, y_true):
        """
        Train meta-learner on out-of-fold predictions.

        Parameters
        ----------
        predictions_dict : dict of {model_name: np.array of predictions}
        y_true : np.array of true values
        """
        self.model_names = list(predictions_dict.keys())
        X_meta = np.column_stack([predictions_dict[k] for k in self.model_names])

        print(f"Training ensemble meta-learner ({len(self.model_names)} models)...")
        self.meta_model.fit(X_meta, y_true)

        self.weights = dict(zip(self.model_names, self.meta_model.coef_))
        print(f"  Ensemble weights: {self.weights}")
        print(f"  Intercept: {self.meta_model.intercept_:.4f}")

    def predict(self, predictions_dict):
        """Blend predictions using learned weights."""
        X_meta = np.column_stack([predictions_dict[k] for k in self.model_names])
        return self.meta_model.predict(X_meta)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'meta_model': self.meta_model,
            'model_names': self.model_names,
            'weights': self.weights
        }, path)
        print(f"Ensemble saved to {path}")

    def load(self, path):
        data = joblib.load(path)
        self.meta_model = data['meta_model']
        self.model_names = data['model_names']
        self.weights = data['weights']
        print(f"Ensemble loaded — models: {self.model_names}")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MODEL REGISTRY
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ModelRegistry:
    """
    Save/load/version model artifacts with MLflow experiment tracking.
    Versioned storage: models/v{N}/
    """

    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        reg_cfg = self.config['model']['registry']
        self.base_path = reg_cfg.get('base_path', 'models')
        self.experiment_name = reg_cfg.get('experiment_name', 'store-sales-forecasting')

    def get_next_version(self):
        """Returns the next version number."""
        existing = []
        if os.path.exists(self.base_path):
            for d in os.listdir(self.base_path):
                if d.startswith('v') and d[1:].isdigit():
                    existing.append(int(d[1:]))
        return max(existing, default=0) + 1

    def save_version(self, models, metrics, feature_cols=None):
        """
        Save a versioned snapshot of all models + metadata.

        Parameters
        ----------
        models : dict of {name: model_object}
        metrics : dict of {metric_name: value}
        feature_cols : list of feature column names
        """
        version = self.get_next_version()
        version_path = os.path.join(self.base_path, f'v{version}')
        os.makedirs(version_path, exist_ok=True)

        # Save models
        for name, model in models.items():
            if hasattr(model, 'save'):
                if isinstance(model, AttentionLSTMModel):
                    model.save(os.path.join(version_path, f'{name}.h5'))
                elif isinstance(model, LightGBMModel):
                    model.save(os.path.join(version_path, name))
                elif isinstance(model, EnsembleModel):
                    model.save(os.path.join(version_path, f'{name}.pkl'))

        # Save metadata
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'model_names': list(models.keys()),
            'feature_columns': feature_cols or []
        }
        with open(os.path.join(version_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Version v{version} saved to {version_path}")

        # Log to MLflow if available
        self._mlflow_log(version, metrics, version_path)

        return version

    def load_version(self, version=None):
        """Load models from a specific version (latest if None)."""
        if version is None:
            version = self.get_next_version() - 1
        if version < 1:
            raise ValueError("No model versions found.")

        version_path = os.path.join(self.base_path, f'v{version}')
        meta_path = os.path.join(version_path, 'metadata.json')

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Version v{version} not found at {version_path}")

        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        print(f"Loading model version v{version} (from {metadata['timestamp']})")
        return metadata, version_path

    def _mlflow_log(self, version, metrics, artifact_path):
        """Log to MLflow (silently skips if not installed/configured)."""
        try:
            import mlflow
            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run(run_name=f'v{version}'):
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
                mlflow.log_param('version', version)
                mlflow.log_artifacts(artifact_path)
            print(f"  📊 MLflow run logged for v{version}")
        except Exception:
            pass  # MLflow optional
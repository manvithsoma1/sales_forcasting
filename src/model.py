import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import yaml
import os

class LSTMModel:
    def __init__(self, input_shape, config_path='config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        """
        Constructs an improved LSTM architecture for sales forecasting.
        Uses deeper stacked LSTMs with BatchNorm for stability.
        """
        print(f"Building LSTM Model with input shape: {self.input_shape}...")
        
        model = Sequential()
        
        # Layer 1: LSTM
        model.add(LSTM(128, return_sequences=True, input_shape=self.input_shape))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        
        # Layer 2: LSTM
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        
        # Layer 3: Final LSTM
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Output Layer
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001, clipnorm=1.0), 
            loss='mse',
            metrics=['mae']
        )        
        return model

    def train(self, X_train, y_train, epochs=None, batch_size=None):
        """
        Trains the model with early stopping and learning rate reduction.
        """
        e = epochs if epochs else self.config['model']['epochs']
        b = batch_size if batch_size else self.config['model']['batch_size']
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print(f"Starting training for up to {e} epochs (early stopping enabled)...")
        history = self.model.fit(
            X_train, y_train,
            epochs=e,
            batch_size=b,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def save_model(self, path='models/best_model_v1.h5'):
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
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
        Constructs the LSTM Architecture.
        """
        print(f"Building LSTM Model with input shape: {self.input_shape}...")
        
        model = Sequential()
        
        # Layer 1: LSTM with Return Sequences (pass data to next LSTM layer)
        model.add(LSTM(64, return_sequences=True, input_shape=self.input_shape))
        model.add(Dropout(0.2)) # Prevent Overfitting
        
        # Layer 2: Standard LSTM
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Output Layer: Predicts 1 value (Sales)
        model.add(Dense(1))
        
        # Compile
# OLD
        # model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # NEW (Add clipnorm)
        model.compile(
            optimizer=Adam(learning_rate=0.001, clipnorm=1.0), 
            loss='mse'
        )        
        return model

    def train(self, X_train, y_train, epochs=None, batch_size=None):
        """
        Trains the model.
        """
        # Use config values if arguments not provided
        e = epochs if epochs else self.config['model']['epochs']
        b = batch_size if batch_size else self.config['model']['batch_size']
        
        print(f"Starting training for {e} epochs...")
        history = self.model.fit(
            X_train, y_train,
            epochs=e,
            batch_size=b,
            validation_split=0.1, # Use 10% of data to check accuracy during training
            verbose=1
        )
        return history

    def save_model(self, path='models/best_model_v1.h5'):
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")
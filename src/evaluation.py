import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

class Evaluator:
    def __init__(self):
        # Create a folder for plots if it doesn't exist
        os.makedirs('plots', exist_ok=True)

    def plot_loss(self, history):
        """
        Plots the training loss curve.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/training_loss.png')
        print("plot saved to plots/training_loss.png")
        # plt.show() # Uncomment if running in Jupyter

    def plot_predictions(self, y_true, y_pred, start_idx=0, length=100):
        """
        Plots a zoom-in of Actual vs Predicted sales.
        """
        plt.figure(figsize=(12, 6))
        # Plot a slice of data (e.g., first 100 days)
        plt.plot(y_true[start_idx:start_idx+length], label='Actual Sales', color='blue')
        plt.plot(y_pred[start_idx:start_idx+length], label='Predicted Sales', color='orange', linestyle='--')
        plt.title(f'Actual vs Predicted Sales (First {length} Time Steps)')
        plt.xlabel('Time Steps')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/prediction_sample.png')
        print("plot saved to plots/prediction_sample.png")

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculates RMSE and MAE.
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        print(f"\n--- Evaluation Metrics ---")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE:  {mae:.2f}")
        return rmse, mae
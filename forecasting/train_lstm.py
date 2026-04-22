# forecasting/train_lstm.py

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from forecasting.data_loader import LSTMDataLoader
from forecasting.lstm_model  import ThermalLSTM
from configs.default_config  import CoolSyncConfig


def train_lstm(
    csv_path:      str   = 'data/merged_lstm_core.csv',
    epochs:        int   = 50,
    learning_rate: float = 1e-3,
    batch_size:    int   = 32,
    lookback:      int   = 8,
    save_path:     str   = 'results/checkpoints/lstm_best.pth',
):
    """
    Trains the ThermalLSTM on merged_lstm_core.csv.

    Steps:
        1. Load and prepare data
        2. Build model
        3. Train with early stopping
        4. Save best model checkpoint
        5. Plot training and validation loss curves
    """

    config = CoolSyncConfig()
    torch.manual_seed(config.seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device         : {device}")

    # Step 1: Load data
    print("\n--- Loading Data ---")
    loader = LSTMDataLoader(csv_path, lookback=lookback)
    train_loader, val_loader = loader.get_loaders(
        batch_size=batch_size
    )

    # Step 2: Build model
    print("\n--- Building Model ---")
    model = ThermalLSTM(
        input_size  = config.lstm_input_size,
        hidden_size = config.lstm_hidden,
        num_layers  = config.lstm_layers,
        dropout     = config.lstm_dropout,
    ).to(device)

    print(f"Parameters     : {model.count_parameters():,}")

    # Loss function and optimizer
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # Reduce learning rate when validation plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        factor=0.5
        
    )

    # Step 3: Training loop
    print("\n--- Training ---")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_losses   = []
    val_losses     = []
    best_val_loss  = float('inf')
    patience_count = 0
    early_stop_at  = 10

    for epoch in range(1, epochs + 1):

        # Training phase
        model.train()
        epoch_train = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(X_batch)
            loss  = loss_fn(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train += loss.item()

        avg_train = epoch_train / len(train_loader)

        # Validation phase
        model.eval()
        epoch_val = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds   = model(X_batch)
                loss    = loss_fn(preds, y_batch)
                epoch_val += loss.item()

        avg_val = epoch_val / len(val_loader)

        # Save best model
        if avg_val < best_val_loss:
            best_val_loss  = avg_val
            patience_count = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_count += 1

        # Early stopping
        if patience_count >= early_stop_at:
            print(f"Early stopping at epoch {epoch}")
            break

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {avg_train:.4f} | "
                f"Val Loss: {avg_val:.4f} | "
                f"Best: {best_val_loss:.4f}"
            )

    # Step 4: Plot training curves
    os.makedirs('results/plots', exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses,   label='Val Loss',   color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('LSTM Training - Thermal Load Prediction')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/plots/lstm_training_curve.png')
    plt.close()

    print(f"\nBest Val Loss  : {best_val_loss:.4f}")
    print(f"Model saved    : {save_path}")
    print(f"Plot saved     : results/plots/lstm_training_curve.png")

    return best_val_loss
# forecasting/lstm_model.py

import torch
import torch.nn as nn


class ThermalLSTM(nn.Module):
    """
    LSTM network for predicting future thermal load
    from historical token volume and GPU metrics.

    Input  : (batch, lookback=8, features=6)
    Output : (batch,) predicted TLHC value
    """

    def __init__(
        self,
        input_size:  int   = 6,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        dropout:     float = 0.2,
    ):
        """
        input_size  : number of input features (6)
        hidden_size : number of LSTM memory units (64)
        num_layers  : number of stacked LSTM layers (2)
        dropout     : fraction of neurons to randomly zero (0.2)
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # LSTM processes the sequence of feature vectors
        # batch_first=True means input shape is
        # (batch, sequence, features)
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        # Final layer maps 64 LSTM outputs to 1 prediction
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        x shape : (batch, lookback, input_size)

        Process:
            1. LSTM processes all 8 timesteps
            2. Take output from final timestep only
               because it has seen all previous steps
            3. Apply dropout
            4. Pass through fully connected layer
            5. Return single prediction value
        """
        # lstm_out shape: (batch, lookback, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take only the last timestep output
        # shape: (batch, hidden_size)
        last_step = lstm_out[:, -1, :]

        # Apply dropout
        last_step = self.dropout(last_step)

        # Predict thermal load
        # shape: (batch, 1)
        output = self.fc(last_step)

        # Remove last dimension
        # shape: (batch,)
        return output.squeeze(-1)

    def count_parameters(self) -> int:
        """
        Returns total number of trainable parameters.
        Useful to report model complexity.
        """
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )
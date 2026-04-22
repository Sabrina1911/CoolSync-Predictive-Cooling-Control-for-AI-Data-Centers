# configs/default_config.py
from dataclasses import dataclass

@dataclass
class CoolSyncConfig:

    # Temperature limits (ASHRAE standard)
    safe_temp_min:  float = 18.0
    safe_temp_max:  float = 27.0
    critical_temp:  float = 35.0

    # Cooling system
    min_cooling_level:     int = 0
    max_cooling_level:     int = 10
    initial_cooling_level: int = 5

    # Starting conditions
    initial_temperature:  float = 24.0
    initial_workload:     float = 0.50
    initial_ambient_temp: float = 22.0
    ambient_temp_min:     float = 20.0
    ambient_temp_max:     float = 24.0

    # Physics equation
    alpha:     float = 1.8
    beta:      float = 1.2
    noise_std: float = 0.15

    # Episode
    episode_length:     int = 200
    time_delta_minutes: int = 5

    # Reward weights
    w_energy:      float = 0.20
    w_overheat:    float = 10.0
    w_overcool:    float = 2.0
    w_instability: float = 1.0
    w_safe_bonus:  float = 5.0

    # Termination
    terminate_on_critical: bool = True

    # LSTM
    lstm_lookback:   int   = 8
    lstm_hidden:     int   = 64
    lstm_layers:     int   = 2
    lstm_dropout:    float = 0.2
    lstm_input_size: int   = 10

    # Normalization
    temp_min_for_norm:   float = 0.0
    temp_max_for_norm:   float = 50.0
    action_min_for_norm: int   = 0
    action_max_for_norm: int   = 2

    # Seed
    seed: int = 42
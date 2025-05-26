# === Shared configuration ===

# Dataset generation
N_SAMPLES = 3000         # Number of samples per option type (Asian, Barrier, Lookback)
M_MC = 5000              # Monte Carlo simulation paths per sample
S0 = 100                  # Underlying asset price
r = 0.02                  # Risk-free rate
v0 = 0.04  # Initial variance, required by Bates models

# Option parameters
K_VALUES = [80, 90, 100, 110, 120]
T_VALUES = [1/12, 3/12, 6/12, 9/12, 1.0]
BARRIER_MIN = 100
BARRIER_MAX = 200

# Training parameters
EPOCHS = 100
BATCH_SIZE = 32           # Adjust based on your hardware capabilities

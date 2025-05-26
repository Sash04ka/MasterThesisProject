# === Shared configuration ===

# Dataset generation
N_SAMPLES = 30000             # 10x total base simulations (each produces 3 rows)
M_MC = 10000                  # 10x Monte Carlo simulations per sample for higher precision
S0 = 100                     # Spot price remains unchanged
r = 0.02                     # Risk-free rate for pricing under Q measure

# Option specification (discrete values used)
K_VALUES = [80, 90, 100, 110, 120]              # Strike prices remain same
T_VALUES = [1/12, 3/12, 6/12, 1.0]              # Maturities remain same
BARRIER_MIN = 100
BARRIER_MAX = 200

# Bates models default/initial variance — retained for compatibility
v0 = 0.04

# Training settings
EPOCHS = 100                 # Training epochs unchanged; can increase if needed
BATCH_SIZE = 128             # Increase batch size for faster training with more data

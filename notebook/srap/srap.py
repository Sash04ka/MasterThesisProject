import pandas as pd
import numpy as np

# Load dataset
file_path = r"/data/option_dataset_asian.csv"
df = pd.read_csv(file_path)

# Prices to find (from your log)
target_prices = [
    2.3055, 3.3280, 12.9333, 16.9814, 17.8167, 21.7810,
    6.5324, 3.6752, 2.7938, 21.3027, 13.3543, 0.5353,
    21.5065, 22.0989, 13.0801, 6.0997, 4.1737, 17.5138,
    16.9329, 3.1773, 3.0070, 3.5688, 13.2300, 6.4435,
    5.6442, 22.0750, 16.9837, 2.1490, 0.8531, 21.3501,
    21.4062, 4.2446, 8.9078, 17.0008, 21.3518, 6.1602,
    2.5101, 1.6250, 4.1426, 17.5532, 9.1519, 3.1697,
    9.7112, 16.9654, 13.1079, 9.6896, 17.1654, 21.3341,
    1.7046, 21.4098, 3.3021, 22.2839, 0.8742, 2.5629, 9.9151
]

tolerance = 0.01

# Find closest matching samples for each price
matched_samples = []
for target in target_prices:
    matches = df[np.abs(df['price'] - target) < tolerance]
    if not matches.empty:
        matched_samples.append(matches)

# Concatenate all found samples
matched_df = pd.concat(matched_samples).drop_duplicates().reset_index(drop=True)
matched_df.to_csv("matched_samples.csv", index=False)

# Show summary info
print(f"Found {len(matched_df)} matching samples for {len(target_prices)} target prices.")

# Display key columns for manual inspection
display_cols = ['type', 'K', 'T', 'r', 'price'] + \
               [col for col in df.columns if col.startswith('sigma_')]

print(matched_df[display_cols].head(15))  # show first 15 matches for inspection

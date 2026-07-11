import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the data safely
csv_file = "output.csv"  # Ensure this matches your file path
print(f"Reading {csv_file}...")

# Read using semicolon separator as produced by your C code
df = pd.read_csv(csv_file, sep=';')

# Calculate Absolute Error or Squared Error for a cleaner loss curve
df['ABS_ERROR'] = df['ERROR'].abs()

print("Processing visualization data...")

# Create a figure with two subplots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ==========================================
# SUBPLOT 1: THE TRAINING LOSS OVER TIME
# ==========================================
# Since 1,000,000 points will crash the renderer, we take a rolling mean 
# or chunk average to see the macroscopic training trend.
window_size = 5000
loss_smooth = df['ABS_ERROR'].rolling(window=window_size).mean()

# Downsample for faster rendering
step = 100
ax1.plot(df['EPOCH'][::step], loss_smooth[::step], color='#e74c3c', label='MAE (Rolling Avg)')
ax1.set_title("Training Convergence (Loss Curve)")
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Mean Absolute Error")
ax1.set_yscale('log') # Log scale helps see micro-adjustments at the end
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend()

# ==========================================
# SUBPLOT 2: CURVE FITTING VERIFICATION
# ==========================================
# We grab the final 2,000 iterations to check the fully trained model.
# We must sort by X, otherwise the line plot will look like a chaotic web.
final_points = df.tail(2000).copy()
final_points_sorted = final_points.sort_values(by='X')

ax2.plot(final_points_sorted['X'], final_points_sorted['TARGET'], 
         color='#2c3e50', linewidth=2.5, label='True Function $f(x)$')
ax2.scatter(final_points_sorted['X'], final_points_sorted['PREDICTION'], 
            color='#3498db', s=15, alpha=0.6, label='C Library Prediction')

ax2.set_title("Universal Approximation Verification ($[-5, 5]$)")
ax2.set_xlabel("Input X")
ax2.set_ylabel("Output Y")
ax2.grid(True, ls="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.savefig('network_results.png', dpi=300)
print("Plot successfully saved to network_results.png!")
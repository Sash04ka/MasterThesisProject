import matplotlib.pyplot as plt

# Training data
epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
losses = [78.6599, 1.5298, 1.2921, 0.6330, 0.1380, 0.5414, 0.1296, 0.9431, 0.1166, 0.1117]

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, marker='o', linestyle='-', color='b')

# Titles and labels
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.grid(True)

# Save or show the plot
plt.tight_layout()
plt.savefig('training_loss_plot.png', dpi=300)  # Save as high-quality PNG
plt.show()


import matplotlib.pyplot as plt

# Data
methods = ['Monte Carlo', 'Neural Network']
times = [109.02, 0.2]  # Total time in seconds for batch

# Create the bar plot
plt.figure(figsize=(7, 5))
plt.bar(methods, times, color=['red', 'green'])

# Titles and labels
plt.title('Total Pricing Time: Monte Carlo vs Neural Network')
plt.ylabel('Total Time (seconds)')
plt.grid(axis='y')

# Annotate exact values
for i, time in enumerate(times):
    plt.text(i, time + 2, f"{time:.2f} s", ha='center', va='bottom', fontweight='bold')

# Save or show the plot
plt.tight_layout()
plt.savefig('mc_vs_mlp_speed.png', dpi=300)  # Save as high-quality PNG
plt.show()

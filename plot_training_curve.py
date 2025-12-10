import matplotlib.pyplot as plt
import numpy as np

# Data points based on our experiments
epochs = [0, 10, 100]
accuracies = [0.02, 0.5641, 0.7700] # Random chance ~2%, 10e 56%, 100e 77%

# Interpolate for a smooth curve
x = np.linspace(0, 100, 100)
# Log-like growth curve fitting these points roughly
# A simple log curve: y = a + b * ln(x + 1)
# or just interpolation
y = np.interp(x, epochs, accuracies)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='GENERanno Validation Accuracy')
plt.scatter(epochs, accuracies, color='red', s=100, zorder=5)

# Annotate points
plt.annotate('Random Chance (~2%)', xy=(0, 0.02), xytext=(5, 0.1), arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('Epoch 10: 56.41%', xy=(10, 0.5641), xytext=(15, 0.5), arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('Epoch 100: 77.00%', xy=(100, 0.77), xytext=(70, 0.7), arrowprops=dict(facecolor='black', shrink=0.05))

plt.title('GENERanno Training Progress (Zymo Dataset)', fontsize=16)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right')
plt.ylim(0, 1.0)

plt.savefig('outputs/zymo_generanno_100e/training_curve.png', dpi=300)
print("Plot saved to outputs/zymo_generanno_100e/training_curve.png")

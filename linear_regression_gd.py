"""
Linear Regression using Gradient Descent (from scratch)

Features:
- Multi-variable regression
- Feature normalization
- Vectorized implementation (NumPy)
- Visualization (2D + 3D)
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Data (2 features: size, bedrooms)
# -----------------------------
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
], dtype=float)

y = np.array([300, 500, 700, 900, 1100], dtype=float)

# Save original for plotting
X_orig = X.copy()

# -----------------------------
# 2. Feature Normalization
# -----------------------------
# Makes all features comparable → faster convergence
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

# -----------------------------
# 3. Initialize parameters
# -----------------------------
m, n = X.shape
w = np.zeros(n)
b = 0.0
alpha = 0.01
iterations = 1000

cost_history = []

# -----------------------------
# 4. Gradient Descent
# -----------------------------
for i in range(iterations):
    # Prediction (vectorized)
    y_pred = X @ w + b

    # Error
    error = y_pred - y

    # Gradients
    dj_dw = (X.T @ error) / m
    dj_db = np.sum(error) / m

    # Update parameters
    w -= alpha * dj_dw
    b -= alpha * dj_db

    # Cost (MSE)
    cost = np.sum(error**2) / (2 * m)
    cost_history.append(cost)

print("Final Weights:", w)
print("Final Bias:", b)

# -----------------------------
# 5. Visualization
# -----------------------------
fig = plt.figure(figsize=(12,5))

# --- 3D Plot ---
ax = fig.add_subplot(1,2,1, projection='3d')
ax.scatter(X_orig[:,0], X_orig[:,1], y, color="blue")

# Create grid
x_surf = np.linspace(X_orig[:,0].min(), X_orig[:,0].max(), 10)
y_surf = np.linspace(X_orig[:,1].min(), X_orig[:,1].max(), 10)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)

# Normalize grid before prediction
X_grid = np.c_[x_surf.ravel(), y_surf.ravel()]
X_grid_norm = (X_grid - mean) / std

z_surf = X_grid_norm @ w + b
z_surf = z_surf.reshape(x_surf.shape)

ax.plot_surface(x_surf, y_surf, z_surf, color="cyan", alpha=0.5)

ax.set_xlabel("Size")
ax.set_ylabel("Bedrooms")
ax.set_zlabel("Price")
ax.set_title("3D Regression Plane")

# --- Cost Plot ---
plt.subplot(1,2,2)
plt.plot(cost_history, color="green")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.grid(True)

plt.tight_layout()

# Save output for README
plt.savefig("output.png")

plt.show()

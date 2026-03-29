# 📈 Linear Regression from Scratch (Gradient Descent)

This project implements **Linear Regression using Gradient Descent from scratch** using NumPy, without relying on high-level ML libraries like sklearn.

---

## 🚀 Key Highlights

* ⚡ Fully vectorized implementation using NumPy
* 📉 Gradient Descent optimization
* 🧠 Multi-variable regression support
* 🎨 Visualization using Matplotlib
* 🧊 3D regression plane (for 2 features)

---

## 🧠 Problem Setup

We model:

y = w₁x₁ + w₂x₂ + ... + b

We minimize the cost function:

J(w,b) = (1 / 2m) Σ (prediction - actual)²

Using Gradient Descent:

w = w - α * ∂J/∂w
b = b - α * ∂J/∂b

---

## ⚙️ Features

### ✅ Vectorized Gradient Descent

* Predictions: `X @ w`
* Gradients: `X.T @ error`

### ✅ Feature Normalization

```python
X = (X - X.mean(axis=0)) / X.std(axis=0)
```

Ensures faster and more stable convergence.

### ✅ Visualizations

* 3D regression plane
* Cost vs iterations graph

---

## 📊 Sample Output

![Output](output.png)

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
python linear_regression_gd.py
```

---

## 📌 Key Learnings

* Gradient Descent converges when:

  * Cost stabilizes
  * Gradients approach zero
* Feature scaling is critical for convergence
* Vectorization improves performance significantly

---

## 🚀 Future Improvements

* Add animation of gradient descent
* Extend to polynomial regression
* Compare with sklearn implementation
* Use real-world datasets

---

## 👨‍💻 Author

Arya S

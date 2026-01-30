# LinearRegression1D_Bias

> **A self-evolving 1D linear regression model where bias learns context instead of remaining constant.**

---

## 🚀 Motivation

Classical linear regression assumes:

```
y = wx + b
```

where the **bias (b)** is a constant scalar.

This assumption breaks down in real-world systems where:

* regime shifts exist
* offsets change with input context
* asymmetric behavior appears across domains

This project introduces the **context-aware bias**:

```
y = wx + b(x)
```

where bias itself as  **learned as a smooth function of input**.

---

## 🧠 Core Idea

* Keep the **linear slope (w)** global and totally interpretable
* Let the **bias evolve with input context**
* Preserve simplicity while capturing non-linear offsets

No neural networks.
No feature engineering.
No piecewise heuristics.

Just a better modeling assumption.

---

## 📐 Mathematical Formulation

The model is defined as:

```
ŷ = w·x + α·tanh(βx + γ)
```

Where:

* `w` → global linear slope
* `α` → bias amplitude
* `β` → bias sharpness
* `γ` → bias shift

The **tanh-based bias** allows smooth, differentiable transitions across regimes.

---

## 📊 What the Visualization Shows

The generated plot decomposes the model into:

* 🔵 **Linear Component** → `w·x`
* 🟠 **Bias Function** → `α·tanh(βx + γ)`
* 🟢 **Final Prediction** → sum of both

This explicit decomposition is critical for:

* interpretability
* debugging
* research analysis

---

## 🗂 Project Structure

```
Self-Evolving-1D-Linear-Regression/
│
├── main.py                  # Training + visualization entry point
│
├── core/
│   ├── model.py             # LinearRegression1D_Bias definition
│   ├── optimizer.py         # Gradient descent updates
│
├── visuals/
│   └── plot_components.py   # Component-wise visualization
│
└── README.md
```

---

## ▶️ How to Run

### 1. Install dependencies

```
pip install numpy matplotlib
```

### 2. Run the experiment

```
python main.py
```

This will:

* generate synthetic data with regime shifts
* train the model via gradient descent
* plot model components

---

## 📈 Example Output

You will see a plot with:

* a straight line (global trend)
* a smooth bias curve
* a final prediction adapting across input space

This behavior **cannot be achieved with standard linear regression**.

---

## 🔬 Research Direction

This project serves as a foundation for an upcoming **research paper** exploring:

* context-aware bias in linear models
* interpretability-preserving alternatives to neural networks
* regime-aware regression for real-world data

---

## 🧪 Use Cases

* Manufacturing & sensor calibration
* Economics & policy modeling
* System drift correction
* Explainable AI pipelines

---

## 🤝 Contributions

Ideas, critiques, and extensions are welcome.

If you build on this concept, please cite or reference the project.

---

## 📜 License

MIT License — free to use, modify, and build upon.

---

⭐ If this idea challenges how you think about linear models, consider starring the repo.

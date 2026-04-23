# 📊 Self-Pruning Neural Network using Stochastic Gates

## 📌 Overview
This project implements a self-pruning neural network using stochastic gating mechanisms to automatically remove less important neurons or weights during training. The model learns which parameters are useful and eliminates redundant ones, improving efficiency without manual pruning.

---

## 🎯 Objectives
- Reduce model complexity dynamically during training  
- Achieve sparsity automatically  
- Improve computational efficiency  
- Maintain model accuracy  

---

## ⚙️ Methodology

### 🔹 Stochastic Gating
Each neuron or weight is assigned a learnable gate value sampled between **0 and 1**:
- Values close to **0 → pruned (inactive)**
- Values close to **1 → retained (active)**

---

### 🔹 Training Process
1. Initialize model with stochastic gates  
2. Perform forward and backward propagation  
3. Apply sparsity regularization  
4. Gradually push unnecessary weights toward zero  

---

## 📈 Stochastic Gate Distribution

![Stochastic Gate Distribution](Stochastic%20Gradient%20Distribution.png)

---

## 🔍 Observations
- Majority of gate values are concentrated near **0**
- Indicates **high sparsity** in the network  
- Only a small subset of neurons remain active  

---

## 🧠 Implementation Details
- Language: **Python**
- Core file: `stochastic_self_pruning.py`
- Techniques used:
  - Stochastic sampling  
  - Regularization for sparsity  
  - Gated forward propagation  

---

## 📊 Results

| Metric            | Observation |
|------------------|------------|
| Sparsity Level   | High |
| Model Size       | Reduced |
| Performance      | Comparable to baseline |

---

## 🚀 Advantages
- Automatic pruning without manual intervention  
- Reduced computation and memory usage  
- Suitable for large-scale neural networks  

---

## ⚠️ Limitations
- Requires careful tuning of hyperparameters  
- Risk of over-pruning  
- Training instability in early stages  

---

## 🔮 Future Work
- Adaptive pruning thresholds  
- Integration with GPU acceleration  
- Testing on real-world datasets  
- Combining with model quantization  

---

## 📂 Repository Structure
├── README.md
├── report.md
├── stochastic_self_pruning.py
├── Stochastic Gradient Distribution.png


---

## 🧾 Conclusion
The stochastic self-pruning neural network effectively reduces unnecessary parameters while maintaining performance. The gate distribution clearly demonstrates that the model learns to eliminate redundant components automatically, making it a powerful approach for efficient deep learning.

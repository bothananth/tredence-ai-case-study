# Stochastic Self-Pruning Neural Network

## 1. Overview

Deep neural networks often contain a large number of redundant parameters, which increases computational cost and memory usage. To address this, this project explores a **stochastic self-pruning mechanism**, where the model learns sparse connectivity during training.

Unlike traditional pruning approaches that remove weights after training, this method integrates pruning directly into the learning process using **probabilistic gating**.

---

## 2. Core Idea

Each weight in the network is associated with a **stochastic gate** that determines whether the connection is active or inactive.

Instead of deterministic gating, this implementation uses a **Hard Concrete distribution**, which enables:
- Differentiable sampling
- Efficient gradient-based optimization
- Better exploration of sparse structures

---

## 3. Methodology

### 3.1 Stochastic Prunable Layer

A custom layer (`StochasticPrunableLinear`) was designed with:
- Learnable weights and biases  
- Additional parameters (`log_alpha`) controlling gate behavior  

During forward propagation:
- Gates are sampled using a stochastic function  
- Sampled gates are applied to weights  
- This results in dynamic pruning during training  

---

### 3.2 Sparsity Regularization

The training objective combines classification and sparsity:
Total Loss = Classification Loss + λ × Sparsity Loss

Where:
- Sparsity Loss is computed as the sum of gate activations  
- This encourages many gates to approach zero  

Unlike mean-based penalties, the use of an L1-style formulation provides stronger pressure for sparsity.

---

### 3.3 Model Enhancements

To improve training stability and generalization:
- Batch Normalization layers were introduced  
- Dropout was added to prevent overfitting  
- A lower temperature parameter was used for sharper gating behavior  

---

### 3.4 Training Configuration

- Dataset: CIFAR-10  
- Optimizer: Adam  
- Epochs: 15  
- Batch Size: 128  
- Device: GPU (if available)  

---

## 4. Experimental Results

| Lambda (λ) | Accuracy (%) | Sparsity (%) |
|------------|-------------|--------------|
| 0.05       | 72.1        | 44.8         |
| 0.1        | 68.3        | 63.5         |
| 0.2        | 60.7        | 82.9         |

---

## 5. Analysis

### 5.1 Effect of Regularization Strength

- Lower λ values maintain higher accuracy but produce limited pruning  
- Moderate λ values achieve a balanced trade-off  
- Higher λ values enforce aggressive pruning but reduce accuracy  

---

### 5.2 Sparsity Behavior

The gate values exhibit a **bimodal distribution**, where:
- A large number of gates collapse toward zero  
- A smaller subset remains active  

This indicates successful separation between essential and redundant connections.

---

## 6. Visualization

![Stochastic Gate Distribution](stochastic_gate_distribution.png)

**Figure 1:** Distribution of stochastic gate values after training.

The histogram demonstrates that:
- Many connections are effectively deactivated  
- Remaining active connections contribute to model performance  

---

## 7. Key Observations

- Stochastic gating improves flexibility in pruning decisions  
- L1-style sparsity loss is critical for achieving high sparsity  
- Proper tuning of λ significantly impacts model behavior  

---

## 8. Conclusion

This work presents a stochastic self-pruning neural network capable of dynamically learning sparse structures during training.

The approach successfully reduces model complexity while maintaining reasonable predictive performance, making it suitable for deployment in resource-constrained environments.

---

## 9. Future Directions

- Extend to convolutional architectures  
- Explore structured pruning techniques  
- Introduce annealing strategies for λ  
- Combine with quantization for further optimization  

---


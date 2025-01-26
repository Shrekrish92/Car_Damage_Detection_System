# Fuzzy Integral Based Ensemble Approach for Car Damage Detection

This project presents an AI-based solution to enhance car damage evaluation for insurance companies. Using an ensemble learning approach with fuzzy ranking, this system integrates multiple transfer learning models to provide robust, explainable, and accurate damage classifications.

---

## Table of Contents
- [Introduction](#introduction)
- [Methodologies](#methodologies)
  - [Transfer Learning Models](#transfer-learning-models)
  - [Fuzzy Ranking and Ensemble Framework](#fuzzy-ranking-and-ensemble-framework)
  - [Explainability](#explainability)
- [Results](#results)
- [Future Scope](#future-scope)

---

## Introduction
Insurance claims for car accidents often involve lengthy and biased evaluation processes. This project proposes an AI-driven approach that combines transfer learning and fuzzy ranking in an ensemble model. This method ensures precise, unbiased, and efficient evaluations, benefiting both car owners and insurance companies.

---

## Methodologies

### Transfer Learning Models
We trained three pre-trained models—ResNet18, GoogLeNet, and Swin-Transformer—on a car damage dataset with six damage categories:
- **Crack**
- **Dent**
- **Shattered Glass**
- **Scratch**
- **Flat Tire**
- **Broken Lamp**

Each model has unique architectural features:
1. **ResNet18**: Uses residual connections to mitigate vanishing gradient issues.
2. **GoogLeNet**: Features inception modules for multi-scale feature extraction.
3. **Swin-Transformer**: Implements window-based attention for computational efficiency.

### Fuzzy Ranking and Ensemble Framework
Unlike traditional ensembles, this approach dynamically evaluates prediction scores for each test sample using a modified Gompertz function. This ensures improved classification without manually adjusting weights.

### Explainability
Explainability was implemented using GradCAM to visualize the decision-making areas of the models. This aids in understanding predictions and building trust, especially for stakeholders like insurance companies.

---

## Results
The ensemble model outperformed individual models in classification accuracy:

| Model       | Accuracy |
|-------------|----------|
| ResNet18    | 90.5%    |
| GoogLeNet   | 93.1%    |
| Swin        | 74.6%    |
| **Ensemble** | **94.0%** |

### ROC-AUC Curve
The ensemble model demonstrated strong class differentiation, with results visualized via ROC-AUC curves.

---

## Future Scope
- **Explainability for Ensembles**: Further research on interpretability for ensemble models.
- **Multi-Label Predictions**: Extending the framework to handle multi-label classifications for real-world applications.


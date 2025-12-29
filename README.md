# AMLS Assignment 25/26 – BreastMNIST Benchmark

This repository contains the coursework for **ELEC0134 Applied Machine Learning Systems (AMLS)**.The objective of this assignment is to benchmark the performance of **classical machine learning models**and **deep learning models** on the **BreastMNIST** medical image dataset, and to analyze the impact of model capacity, data augmentation, and training budget.

---

## Project Structure
AMLS_25_26_SN24100833/
├── Code/
│   ├── model_A/        # Classical ML models (SVM, feature extraction, training & evaluation)
│   ├── model_B/        # Deep learning models (CNN architectures, training & evaluation)
├── Datasets/           # Dataset folder (left empty for submission)
├── main.py             # Entry point to run all experiments
├── complexity.py
├── plot_results.py
└── README.md           # Project documentation


---

## Models Implemented
### Model A – Classical Machine Learning
- Linear / Kernel Support Vector Machine (SVM)
- Feature pipelines including:
  - Raw pixel flattening
  - Dimensionality reduction (e.g. PCA)
- Performance comparison between different feature pipelines
---

### Model B – Deep Learning
- Convolutional Neural Network (CNN)
- Comparison of different model capacities
- Analysis of training budget and data augmentation strategies
---

## Data Augmentation
The following augmentation techniques are applied where appropriate:
- Random rotations and flips
- Intensity-based transformations (e.g. noise or contrast adjustments)


```bash
python main.py
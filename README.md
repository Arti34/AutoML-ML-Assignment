# AutoML, RAPIDS and PyCaret Assignment

This repository contains my work for the assignment covering K-Means, AutoGluon, NVIDIA RAPIDS, and PyCaret.

I kept each notebook focused on the important concepts and added short notes before the main code cells so I can also use them later for revision.

## Contents

### Part 1 - K-Means and Its Variations
Notebook: `Part1_KMeans_and_Variations_Notes.ipynb`

Topics covered:
- Lloyd's K-Means from scratch
- scikit-learn KMeans
- K-Means++ initialization
- Bisecting K-Means
- MiniBatch K-Means
- K-Means limitations on non-convex data

Main takeaway: K-Means repeatedly assigns points to the nearest centroid and updates the centroid using the mean of the assigned points. Different variations improve initialization, scalability, or the way clusters are created.

### Part 2 - AutoGluon Capabilities
Notebook: `Part2_AutoGluon_Capabilities_FINAL_v2.ipynb`

Topics covered:
- Binary classification
- Multiclass classification
- Regression
- Time-series forecasting
- AutoGluon capability overview

Main takeaway: AutoGluon provides a common AutoML workflow where we define the target and metric, fit a predictor, compare models, and generate predictions.

### Part 3 - AutoGluon End-to-End ML
Notebook: `Part3_AutoGluon_End_to_End_ML_FINAL_FAST.ipynb`

Topics covered:
- Train/test split
- Baseline model
- AutoGluon training
- Leaderboard
- Accuracy, Precision, Recall, F1, ROC-AUC and PR-AUC
- Confusion matrix
- Decision thresholds
- Feature importance

Main takeaway: AutoML reduces model-training code, but choices such as the target, metric, data split, threshold, and interpretation still require ML understanding.

### Part 4 - NVIDIA RAPIDS: CPU vs GPU
Notebook: `Part4_NVIDIA_RAPIDS_CPU_vs_GPU_FINAL_FAST.ipynb`

Topics covered:
- NumPy vs CuPy
- pandas vs cuDF
- scikit-learn K-Means vs cuML K-Means
- CPU-to-GPU transfer overhead
- Large vs small workloads

Main takeaway: GPU acceleration is most useful when the workload is large and parallel enough to justify GPU startup and transfer overhead.

| CPU | GPU |
| --- | --- |
| NumPy | CuPy |
| pandas | cuDF |
| scikit-learn | cuML |

Runtime: use a **T4 GPU** in Google Colab.

### Part 5 - PyCaret Capabilities
Notebook: `Part5_PyCaret_3_3_2_FINAL_STABLE.ipynb`

Topics covered:
- Classification
- Regression
- Clustering
- Anomaly Detection
- Time-Series Forecasting

Main takeaway: PyCaret provides a similar low-code workflow for different machine-learning problems.

Supervised flow:

```python
setup(...)
best = compare_models(...)
predictions = predict_model(best)
```

Unsupervised flow:

```python
setup(...)
model = create_model(...)
result = assign_model(model)
```

Runtime: this notebook uses **PyCaret 3.3.2** with **Python 3.11**. In Colab, use runtime version **2025.07**.

### Part 6 - PyCaret MLOps
Notebook: `Part6_PyCaret_MLOps_FINAL_FAST.ipynb`

Topics covered:
- Model training and selection
- Honest test evaluation
- Finalizing a model
- Saving and loading the pipeline
- Predicting on new data
- Drift monitoring using PSI
- Monitoring model performance
- Retraining

Main takeaway:

**Train -> Evaluate -> Finalize -> Save/Load -> Predict -> Monitor -> Retrain**

Runtime: use the Colab **2025.07** runtime with Python 3.11.

## Revision Notes

Before the important code cells, I added short notes using this format:

- **What we saw before**
- **How this cell is different**
- **What this cell does**
- **What we achieve after running it**
- **Remember**

I used this format so the notebooks can work both as assignment submissions and as quick revision material later.

## Running the Notebooks

The notebooks are designed to run in Google Colab.

1. Open the `.ipynb` file in Colab.
2. Run the cells from top to bottom.
3. Save the notebook with the generated outputs.

Special runtime requirements:
- **Part 4:** T4 GPU runtime
- **Part 5:** Colab 2025.07 / Python 3.11
- **Part 6:** Colab 2025.07 / Python 3.11

## Video Walkthrough

| Part | Video |
| --- | --- |
| Part 1 - K-Means | https://youtu.be/XW5lx2QGLCI |
| Part 2 - AutoGluon Capabilities | https://youtu.be/MSIo6TtnsmI |
| Part 3 - AutoGluon End-to-End ML | https://youtu.be/F8EYwiBPeDw |
| Part 4 - NVIDIA RAPIDS | https://youtu.be/sE0OlGrMDPU |
| Part 5 - PyCaret Capabilities | https://youtu.be/T_kA3HzJ6ns |
| Part 6 - PyCaret MLOps | https://youtu.be/-Q6bJ9gPRgM |

## Overall Takeaway

This assignment helped me understand the difference between implementing ML concepts directly and using higher-level AutoML tools.

K-Means helped me understand the clustering logic itself. AutoGluon and PyCaret showed how AutoML can reduce repetitive model-development work, while RAPIDS demonstrated how GPU acceleration can improve performance for suitable workloads.

The main learning for me was that tools can automate many steps, but understanding the data, selecting the correct metric, evaluating the model properly, and interpreting the results are still important parts of the machine-learning workflow.
